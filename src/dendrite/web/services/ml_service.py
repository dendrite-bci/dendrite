"""
ML Workbench service  - model listing, training jobs, decoder saving,
and online training loop.
"""

import asyncio
import json
import logging
import os
import queue
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np

from dendrite.data.loaders import (
    EpochedData,
    load_epochs,
    load_moabb_for_training,
    load_study_history,
    merge_recordings,
)
from dendrite.data.storage.database import TrainingJobRepository
from dendrite.ml.decoders.registry import (
    DECODER_REGISTRY,
    get_available_decoders,
    get_decoder_entry,
)
from dendrite.ml.evaluation import benchmark_cv, enrich_per_trial, evaluate_epochs
from dendrite.ml.models.registry import MODEL_REGISTRY
from dendrite.ml.training.runner import run_training
from dendrite.utils.serialization import jsonify
from dendrite.web.services.data_service import DataService

logger = logging.getLogger(__name__)

_MIN_CLASSES = 2


def _check_enough_classes(epoched: EpochedData) -> None:
    if len(set(epoched.y.tolist())) < _MIN_CLASSES:
        raise ValueError(f"Need at least {_MIN_CLASSES} classes")


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class MLService:
    """Orchestrates ML model listing, training jobs, and decoder management."""

    def __init__(self, data_service: DataService) -> None:
        self._data_service = data_service
        self._job_repo = TrainingJobRepository(data_service.db)
        self._bridge: Any | None = None
        self._active_jobs: dict[int, asyncio.Task] = {}
        self._job_progress: dict[int, dict] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loaded_data: EpochedData | None = None
        self._eval_data: EpochedData | None = None
        self._executor = None
        self._mp_manager = None
        self._stop_events: dict[int, Any] = {}  # job_id → multiprocessing.Event

    def _get_executor(self):
        if self._executor is None:
            from concurrent.futures import ProcessPoolExecutor
            self._executor = ProcessPoolExecutor(max_workers=1)
        return self._executor

    def _get_manager(self):
        if self._mp_manager is None:
            import multiprocessing
            self._mp_manager = multiprocessing.Manager()
        return self._mp_manager

    def _shutdown_resources(self) -> None:
        """Shut down process pool executor and multiprocessing manager."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        if self._mp_manager is not None:
            try:
                self._mp_manager.shutdown()
            except (OSError, RuntimeError):
                pass
            self._mp_manager = None

    def set_bridge(self, bridge: Any) -> None:
        self._bridge = bridge
        self._loop = asyncio.get_running_loop()

    def _broadcast_from_thread(self, channel: str, data: dict) -> None:
        if self._bridge and self._loop:
            asyncio.run_coroutine_threadsafe(
                self._broadcast(channel, data), self._loop,
            )

    def _broadcast_step(self, step: str) -> None:
        self._broadcast_from_thread("training", {"type": "data_loading_step", "step": step})

    # ------------------------------------------------------------------ #
    # Job helpers (spawn, run, result retrieval)
    # ------------------------------------------------------------------ #

    def _spawn_job(
        self, study_id, model_type: str, config: dict, job_type: str, coro_fn,
    ) -> dict[str, Any]:
        """Create a job record, spawn an async task, return {job_id, status}."""
        job_id = self._job_repo.create_job(
            study_id=study_id, model_type=model_type,
            config_json=json.dumps(config), job_type=job_type,
        )
        task = asyncio.create_task(coro_fn(job_id))
        self._active_jobs[job_id] = task
        return {"job_id": job_id, "status": "running"}

    async def _run_job(self, job_id: int, work_fn, result_type: str = "complete") -> None:
        """Generic async job runner — status, broadcast, error handling."""
        self._job_repo.update_status(job_id, "running", started_at=_now_iso())
        await self._broadcast("training", {"type": "started", "job_id": job_id})
        try:
            result = jsonify(await work_fn())
            self._job_repo.set_result(job_id, json.dumps(result))
            self._job_repo.update_status(job_id, "completed", completed_at=_now_iso())
            await self._broadcast("training", {
                "type": result_type, "job_id": job_id, "result": result,
            })
        except asyncio.CancelledError:
            logger.info(f"Job {job_id} cancelled")
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}", exc_info=True)
            self._job_repo.update_status(
                job_id, "failed", error_message=str(e), completed_at=_now_iso(),
            )
            await self._broadcast("training", {
                "type": "failed", "job_id": job_id, "error": str(e),
            })
        finally:
            self._cleanup_job(job_id)
            self._shutdown_resources()

    def _get_job_result(self, job_id: int) -> dict | None:
        """Get a job's parsed result from the database."""
        job = self._job_repo.get_by_id(job_id)
        if job and job.get("result_json"):
            return json.loads(job["result_json"])
        return None

    # ------------------------------------------------------------------ #
    # Data Loading (delegates to data/loaders)
    # ------------------------------------------------------------------ #

    async def discover_moabb_datasets(self) -> list[dict[str, Any]]:
        from dendrite.data.loaders.moabb_discovery import discover_moabb_datasets as _discover

        return await asyncio.wait_for(asyncio.to_thread(_discover), timeout=30)

    async def load_moabb_dataset(self, config: dict[str, Any]) -> dict[str, Any]:
        def _sync():
            self._loaded_data = load_moabb_for_training(config, self._broadcast_step)
            eval_ratio = config.get("eval_split", 0.2)
            if eval_ratio > 0:
                self._eval_data = self._loaded_data.split_eval(eval_ratio)
            else:
                self._eval_data = None
            result = self._loaded_data.info()
            if self._eval_data is not None:
                result["eval"] = self._eval_data.info()
            return result

        return await asyncio.wait_for(asyncio.to_thread(_sync), timeout=120)

    async def load_recording(self, config: dict[str, Any]) -> dict[str, Any]:
        recording_ids = config.get("recording_ids")
        eval_recording_ids = config.get("eval_recording_ids")

        if recording_ids:
            def _sync():
                self._loaded_data = merge_recordings(
                    recording_ids, config, self._data_service, self._broadcast_step,
                )
                if eval_recording_ids:
                    self._broadcast_step("Loading eval recordings...")
                    self._eval_data = merge_recordings(
                        eval_recording_ids, config, self._data_service,
                        self._broadcast_step,
                    )
                else:
                    # Auto-split: hold out eval_split% of train data for evaluation
                    eval_ratio = config.get("eval_split", 0.2)
                    if eval_ratio > 0:
                        self._broadcast_step("Auto-splitting eval data...")
                        self._eval_data = self._loaded_data.split_eval(eval_ratio)
                    else:
                        self._eval_data = None
                result = self._loaded_data.info()
                if self._eval_data is not None:
                    result["eval"] = self._eval_data.info()
                return result

            timeout = 120 + 60 * len(recording_ids)
            return await asyncio.wait_for(asyncio.to_thread(_sync), timeout=timeout)

        def _sync():
            recording = self._data_service.recordings.get_by_id(config["recording_id"])
            if not recording:
                raise ValueError(f"Recording {config['recording_id']} not found")
            self._loaded_data = load_epochs(config, recording["hdf5_file_path"])
            self._eval_data = None
            return self._loaded_data.info()

        return await asyncio.wait_for(asyncio.to_thread(_sync), timeout=120)

    def get_loaded_data_info(self) -> dict[str, Any] | None:
        if self._loaded_data is None:
            return None
        info = self._loaded_data.info()
        if self._eval_data is not None:
            info["eval"] = self._eval_data.info()
        return info

    # ------------------------------------------------------------------ #
    # Model listing
    # ------------------------------------------------------------------ #

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {
                "model_type": name,
                "description": entry.get("description", name),
                "modalities": entry.get("modalities", ["any"]),
                "default_steps": entry.get("default_steps", []),
                "step_types": entry.get("step_types", {}),
            }
            for name in get_available_decoders()
            if (entry := get_decoder_entry(name))
        ]

    def get_model_config_schema(self, model_type: str) -> dict[str, Any] | None:
        registry_entry = MODEL_REGISTRY.get(model_type)
        if not registry_entry:
            if model_type in DECODER_REGISTRY:
                return {"properties": {}, "title": model_type, "type": "object"}
            return None
        config_class = registry_entry.get("config")
        if config_class is None:
            return {"properties": {}, "title": model_type, "type": "object"}
        return config_class.model_json_schema()

    # ------------------------------------------------------------------ #
    # Workbench training jobs
    # ------------------------------------------------------------------ #

    def _attach_progress(self, job: dict) -> None:
        job_id = job["job_id"]
        if job["status"] == "running" and job_id in self._job_progress:
            job["progress"] = self._job_progress[job_id]

    def _cleanup_job(self, job_id: int) -> None:
        self._active_jobs.pop(job_id, None)
        self._job_progress.pop(job_id, None)
        self._stop_events.pop(job_id, None)

    def list_jobs(self, study_id: int | None = None, job_type: str | None = None) -> list[dict]:
        jobs = self._job_repo.list_jobs(study_id, job_type=job_type)
        for job in jobs:
            self._attach_progress(job)
        return jobs

    def get_job(self, job_id: int) -> dict[str, Any] | None:
        job = self._job_repo.get_by_id(job_id)
        if not job:
            return None
        self._attach_progress(job)
        return job

    async def start_training(self, config: dict[str, Any]) -> dict[str, Any]:
        study_id = config.get("study_id")
        model_type = config["model_type"]

        if study_id is not None:
            study = self._data_service.studies.get_by_id(study_id)
            if not study:
                raise ValueError(f"Study {study_id} not found")
        if config.get("use_loaded_data") and self._loaded_data is None:
            raise ValueError("No data loaded - load data first via the Data tab")
        if model_type not in DECODER_REGISTRY:
            raise ValueError(f"Unknown model type: {model_type}")

        return self._spawn_job(
            study_id, model_type, config, "training",
            lambda jid: self._run_training_async(jid, config),
        )

    async def cancel_training(self, job_id: int) -> bool:
        task = self._active_jobs.get(job_id)
        if task and not task.done():
            # Signal subprocess to stop before shutting down executor
            stop_event = self._stop_events.pop(job_id, None)
            if stop_event:
                stop_event.set()
            self._shutdown_resources()
            task.cancel()
            self._job_repo.update_status(job_id, "cancelled", completed_at=_now_iso())
            self._cleanup_job(job_id)
            await self._broadcast("training", {"type": "cancelled", "job_id": job_id})
            return True
        return False

    async def save_decoder(
        self, job_id: int, decoder_name: str, description: str | None = None,
    ) -> dict[str, Any] | None:
        job = self._job_repo.get_by_id(job_id)
        if not job or job["status"] != "completed":
            return None

        result = self._get_job_result(job_id)
        if not result or "path" not in result:
            return None

        from dendrite.ml.decoders import load_decoder

        decoder = await asyncio.to_thread(load_decoder, result["path"])

        study_id = job.get("study_id")
        study_name = None
        if study_id:
            study = self._data_service.studies.get_by_id(study_id)
            study_name = study["study_name"] if study else None

        json_path = await asyncio.to_thread(decoder.save, decoder_name, study_name)

        metrics = decoder.get_training_metrics() or {}
        first_metrics = next(iter(metrics.values()), {})
        training_rec_ids = None
        if hasattr(decoder, "config") and decoder.config:
            training_rec_ids = decoder.config.training_recording_ids
        decoder_id = self._data_service.decoders.add_decoder(
            study_id=study_id, decoder_name=decoder_name,
            decoder_path=json_path, model_type=job["model_type"],
            num_classes=decoder.num_classes,
            training_accuracy=first_metrics.get("final_train_acc"),
            validation_accuracy=first_metrics.get("final_val_acc"),
            description=description,
            training_recording_ids=training_rec_ids,
        )
        if decoder_id:
            self._job_repo.link_decoder(job_id, decoder_id)
        return {"decoder_id": decoder_id, "decoder_path": json_path}

    def delete_job(self, job_id: int) -> bool:
        task = self._active_jobs.get(job_id)
        if task and not task.done():
            task.cancel()
        self._cleanup_job(job_id)
        return self._job_repo.delete_job(job_id)

    # ------------------------------------------------------------------ #
    # Training config
    # ------------------------------------------------------------------ #

    def _enrich_training_config(self, config: dict[str, Any]) -> None:
        """Populate decoder-describing metadata on the training config dict.

        Propagates channel labels, sample rate, event/label mappings, and
        preprocessing normalization from loaded data into the config so the
        resulting decoder is self-describing.
        """
        modality = config.get("modality", "eeg")

        # Propagate loaded-data metadata
        if self._loaded_data:
            if not config.get("event_id"):
                config["event_id"] = self._loaded_data.metadata.get("event_id", {})
            if not config.get("channel_labels") and self._loaded_data.channel_names:
                selected_ch = config.get("selected_channels")
                if selected_ch:
                    names = [
                        self._loaded_data.channel_names[i]
                        for i in selected_ch
                        if i < len(self._loaded_data.channel_names)
                    ]
                else:
                    names = list(self._loaded_data.channel_names)
                config["channel_labels"] = {modality: names}
            if not config.get("sample_rate"):
                config["sample_rate"] = self._loaded_data.sample_rate
            if not config.get("recording_ids"):
                rec_ids = self._loaded_data.metadata.get("recording_ids")
                rec_id = self._loaded_data.metadata.get("recording_id")
                if rec_ids:
                    config["recording_ids"] = rec_ids
                elif rec_id:
                    config["recording_id"] = rec_id

        # Normalize loose lowcut/highcut/apply_rereferencing into mode_preprocessing
        if not config.get("mode_preprocessing") and (
            config.get("lowcut") is not None or config.get("highcut") is not None
        ):
            config["mode_preprocessing"] = {
                modality: {
                    "lowcut": config.get("lowcut"),
                    "highcut": config.get("highcut"),
                    "apply_rereferencing": config.get("apply_rereferencing", False),
                    "filter_order": config.get("filter_order", 4),
                },
            }

        # Build event_mapping/label_mapping so the decoder is self-describing
        if not config.get("event_mapping"):
            event_id = config.get("event_id") or {}
            selected = config.get("selected_events")
            if event_id:
                pairs = (
                    [(name, code) for name, code in event_id.items() if name in selected]
                    if selected else list(event_id.items())
                )
                config["event_mapping"] = {int(code): name for name, code in pairs}
                config["label_mapping"] = {
                    name: i for i, (name, _) in enumerate(
                        sorted(pairs, key=lambda p: p[1])
                    )
                }
        # NOTE: "rest" / background class is NOT added here — it's derived
        # from the actual y data in _run_training_async after loading.

    # ------------------------------------------------------------------ #
    # Training execution
    # ------------------------------------------------------------------ #

    async def _run_training_async(self, job_id: int, config: dict[str, Any]) -> None:
        async def work():
            if config.get("use_loaded_data") and self._loaded_data is not None:
                X, y = self._loaded_data.X.copy(), self._loaded_data.y.copy()
                selected = config.get("selected_channels")
                if selected and X.ndim == 3:
                    valid = [i for i in selected if i < X.shape[1]]
                    if valid:
                        X = X[:, valid, :]
                selected_events = config.get("selected_events")
                event_id = self._loaded_data.metadata.get("event_id")
                label_map = self._loaded_data.metadata.get("label_map", {})
                if selected_events and event_id:
                    keep = {label_map[c] for n, c in event_id.items()
                            if n in selected_events and c in label_map}
                    if keep:
                        mask = np.isin(y, list(keep))
                        X, y = X[mask], y[mask]
            else:
                rid = config.get("recording_id")
                if rid:
                    rec = self._data_service.recordings.get_by_id(rid)
                    if not rec:
                        raise ValueError(f"Recording {rid} not found")
                    file_path = rec["hdf5_file_path"]
                else:
                    file_path = config.get("file_path")
                if not file_path:
                    raise ValueError("Either file_path or recording_id required")
                epoched = await asyncio.to_thread(load_epochs, config, file_path)
                X, y = epoched.X, epoched.y

            self._enrich_training_config(config)

            loop = asyncio.get_event_loop()
            workbench_config = {**config, "max_threads": os.cpu_count() or 4}
            manager = self._get_manager()
            progress_q = manager.Queue()
            stop_event = manager.Event()
            self._stop_events[job_id] = stop_event

            if config.get("optuna_enabled"):
                from dendrite.ml.search import run_optuna_search
                msg_type, target = "optuna_trial", run_optuna_search
                args = (X, y, workbench_config, progress_q, stop_event)
            else:
                msg_type, target = "epoch", run_training
                save_name = f"workbench_{job_id}_{int(time.time())}"
                args = (X, y, workbench_config, save_name, progress_q, stop_event)

            drain_task = asyncio.create_task(
                self._drain_progress(job_id, progress_q, msg_type=msg_type),
            )
            try:
                result = await loop.run_in_executor(
                    self._get_executor(), target, *args,
                )
            finally:
                self._stop_events.pop(job_id, None)
                drain_task.cancel()

            # Compute honest eval metrics on pre-split held-out data
            if self._eval_data is not None and isinstance(result, dict) and result.get("path"):
                result["eval_metrics"] = await asyncio.to_thread(
                    self._compute_eval_metrics, result["path"], self._eval_data, config,
                )

            return result

        await self._run_job(job_id, work)

    async def _drain_progress(
        self, job_id: int, q: Any, msg_type: str = "epoch", timeout: float = 0.5,
    ) -> None:
        """Drain a thread-safe queue and broadcast each item via WS."""
        loop = asyncio.get_event_loop()
        try:
            while True:
                try:
                    msg = await loop.run_in_executor(None, q.get, True, timeout)
                except queue.Empty:
                    continue
                if msg is None:
                    break
                self._job_progress[job_id] = {
                    "type": msg_type, "job_id": job_id, **msg,
                }
                await self._broadcast("training", self._job_progress[job_id])
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"Progress drain error for job {job_id}: {e}")

    @staticmethod
    def _compute_eval_metrics(
        decoder_path: str, eval_data: Any, config: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate a trained decoder on held-out eval data (runs in thread)."""
        from sklearn.metrics import classification_report, confusion_matrix

        from dendrite.ml.decoders import load_decoder

        try:
            decoder = load_decoder(decoder_path)
            X_eval, y_eval = eval_data.X, eval_data.y
            y_pred = decoder.predict(X_eval)
            if isinstance(y_pred, int):
                y_pred = np.array([y_pred])

            class_names = eval_data.metadata.get("class_names", [])
            if class_names:
                class_labels = class_names
            elif config.get("label_mapping"):
                class_labels = [name for name, _ in sorted(
                    config["label_mapping"].items(), key=lambda x: x[1],
                )]
            elif config.get("event_mapping"):
                class_labels = [
                    config["event_mapping"][c]
                    for c in sorted(config["event_mapping"])
                ]
            else:
                class_labels = [str(c) for c in sorted(np.unique(y_eval))]

            return {
                "confusion_matrix": confusion_matrix(y_eval, y_pred).tolist(),
                "classification_report": classification_report(
                    y_eval, y_pred, output_dict=True, zero_division=0,
                ),
                "val_samples": len(y_eval),
                "class_labels": class_labels,
            }
        except Exception:
            logger.warning("Failed to compute held-out eval metrics", exc_info=True)
            return {}

    # ------------------------------------------------------------------ #
    # Online training loop
    # ------------------------------------------------------------------ #

    async def run_online_training_loop(self, training_queue, shared_state) -> None:
        loop = asyncio.get_event_loop()
        logger.info("Online training loop started")
        history_cache: dict[str, tuple] = {}  # mode_name → (X, y)

        while True:
            # Queue read — OSError here means the queue pipe is closed (pipeline stopped)
            try:
                try:
                    request = await loop.run_in_executor(
                        None, lambda q=training_queue: q.get(timeout=0.5),
                    )
                except queue.Empty:
                    continue
                except OSError:
                    logger.info("Online training loop: queue closed (pipeline stopped)")
                    return
            except asyncio.CancelledError:
                logger.info("Online training loop cancelled")
                return

            if request is None:
                continue

            # Process request — errors here must not kill the loop
            try:
                mode_name = request["mode_name"]

                session_file = shared_state.get("recording_file")
                if not session_file:
                    logger.warning(f"No session file for {mode_name}")
                    continue

                logger.info(f"Online training request from {mode_name}")

                try:
                    epoched = await asyncio.to_thread(
                        load_epochs, request, session_file, swmr=True,
                    )
                    _check_enough_classes(epoched)
                    X, y = epoched.X, epoched.y
                except ValueError as e:
                    logger.info(f"Live session has no epochs yet: {e}")
                    X, y = None, None

                # Augment (or bootstrap) with historical recordings (cached)
                if request.get("use_study_history") and request.get("study_name"):
                    if mode_name not in history_cache:
                        try:
                            hist_X, hist_y = await asyncio.to_thread(
                                load_study_history, request, request["study_name"],
                                self._data_service,
                                X.shape[1:] if X is not None else None,
                                request.get("study_history_recording_ids"),
                            )
                            history_cache[mode_name] = (hist_X, hist_y)
                        except Exception as e:
                            logger.warning(f"Historical data load failed: {e}")
                            history_cache[mode_name] = (None, None)
                    hist_X, hist_y = history_cache[mode_name]
                    if hist_X is not None:
                        if X is not None:
                            X = np.concatenate([hist_X, X], axis=0)
                            y = np.concatenate([hist_y, y], axis=0)
                        else:
                            X, y = hist_X, hist_y
                        logger.info(
                            f"Study history: {len(hist_y)} epochs (total: {len(y)})"
                        )

                if X is None or len(X) == 0:
                    logger.warning(f"No training data available for {mode_name}")
                    continue

                model_config = request.get("decoder_config", {}).get("model_config", {})
                modalities = request.get("modalities")
                if not modalities:
                    logger.warning(
                        f"Training request for {mode_name} missing modalities, skipping"
                    )
                    continue
                modality = modalities[0]
                train_config = {
                    **model_config, "modality": modality,
                    "mode_preprocessing": request.get("mode_preprocessing", {}),
                    "event_mapping": request.get("event_mapping"),
                    "label_mapping": request.get("label_mapping"),
                }
                save_name = f"online_{mode_name}_{int(time.time())}"

                result = await loop.run_in_executor(
                    self._get_executor(), run_training, X, y, train_config, save_name,
                )

                decoder_info = {
                    "path": result["path"],
                    "timestamp": time.time(),
                    "n_epochs": result.get("n_epochs", 0),
                    "elapsed": result.get("elapsed", 0.0),
                    "source_mode": mode_name,
                }
                shared_state.set(f"{mode_name}:trained_decoder", decoder_info)
                shared_state.set("latest_trained_decoder", decoder_info)

                logger.info(f"Online training done for {mode_name}: {result['path']}")

            except asyncio.CancelledError:
                logger.info("Online training loop cancelled")
                return
            except Exception as e:
                logger.error(f"Online training error: {e}", exc_info=True)

    # ------------------------------------------------------------------ #
    # Evaluation (epoch-by-epoch decoder eval on loaded data)
    # ------------------------------------------------------------------ #

    async def start_evaluation(self, config: dict[str, Any]) -> dict[str, Any]:
        source_job_id = config.get("job_id")
        if not source_job_id:
            raise ValueError("job_id is required")
        source_job = self._job_repo.get_by_id(source_job_id)
        if not source_job or source_job["status"] != "completed":
            raise ValueError(f"Job {source_job_id} not found or not completed")

        data = self._eval_data or self._loaded_data
        if data is None:
            raise ValueError("No data loaded - load data first")

        result = self._get_job_result(source_job_id)
        if not result or "path" not in result:
            raise ValueError("Decoder path not found for this job")

        decoder_path = result["path"]

        # For sliding window mode, get recording path from loaded data metadata
        recording_path = None
        if config.get("mode") == "sliding_window":
            meta = data.metadata if data else {}
            rec_id = meta.get("recording_id")
            rec_ids = meta.get("recording_ids")
            rid = rec_id or (rec_ids[0] if rec_ids else None)
            if rid:
                rec = self._data_service.recordings.get_by_id(rid)
                if rec:
                    recording_path = rec["hdf5_file_path"]
            if not recording_path:
                raise ValueError("Sliding window eval requires recording-based data")

        return self._spawn_job(
            source_job.get("study_id"), source_job["model_type"], config, "evaluation",
            lambda jid: self._run_eval(jid, config, decoder_path, data, recording_path),
        )

    @staticmethod
    def _prepare_sliding_window_data(
        recording_path: str,
        source_config: dict[str, Any],
        step_size_ms: int = 100,
        preprocessing_config: Any | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Load recording, apply preprocessing, build sliding window config."""
        from dendrite.data.loaders import load_file
        from dendrite.data.loaders._types import build_preprocessing_config

        loaded = load_file(recording_path)
        # Get modality from decoder's preprocessing config (authoritative)
        modality = ""
        if preprocessing_config and preprocessing_config.modality_preprocessing:
            modality = next(iter(preprocessing_config.modality_preprocessing))
        if not modality:
            modality = source_config.get("modality", "")
        if modality:
            loaded.filter_modality(modality)

        if preprocessing_config is not None:
            primary = next(iter(preprocessing_config.modality_preprocessing.values()), None)
            if primary:
                loaded.preprocess(primary)
        else:
            loaded.preprocess(build_preprocessing_config(source_config))

        selected = source_config.get("selected_events")
        event_id = loaded.event_id or {}
        if selected:
            ev_map = {code: name for name, code in event_id.items() if name in selected}
        else:
            ev_map = {code: name for name, code in event_id.items()}

        return loaded, {
            "step_ms": step_size_ms,
            "epoch_tmin": source_config.get("epoch_tmin", 0.0),
            "epoch_tmax": source_config.get("epoch_tmax", 2.0),
            "event_mapping": ev_map,
        }

    async def _run_eval(
        self, job_id: int, config: dict, decoder_path: str, data: Any, recording_path: str | None = None,
    ) -> None:
        async def work():
            from dendrite.ml.decoders import load_decoder
            from dendrite.ml.evaluation import evaluate_sliding_window

            decoder = await asyncio.to_thread(load_decoder, decoder_path)
            ch_idx = config.get("channel_indices")

            progress_q: queue.Queue = queue.Queue()
            drain_task = asyncio.create_task(
                self._drain_progress(job_id, progress_q, "eval_step", timeout=0.1),
            )

            def _on_progress(step, total, info, trial_timeline=None):
                progress_q.put({
                    "step": step, "total": total,
                    "trial": jsonify(info),
                    "timeline": jsonify(trial_timeline) if trial_timeline else [],
                })

            try:
                if config.get("mode") == "sliding_window" and recording_path:
                    src_cfg = {}
                    src_job = self._job_repo.get_by_id(config.get("job_id", 0))
                    if src_job and src_job.get("config_json"):
                        src_cfg = json.loads(src_job["config_json"])

                    loaded, sw_config = await asyncio.to_thread(
                        self._prepare_sliding_window_data,
                        recording_path, src_cfg,
                        config.get("step_size_ms", 100),
                        decoder.config.preprocessing_config,
                    )
                    sw_config["detection_strategy"] = config.get("detection_strategy", "dwell")
                    sw_config["dwell_n"] = config.get("dwell_n", 3)
                    sw_config["confidence_threshold"] = config.get("confidence_threshold", 0.0)
                    # Pass training label_map so eval uses the same class indices
                    if data and data.metadata.get("label_map"):
                        sw_config["code_to_class"] = data.metadata["label_map"]
                    logger.info(
                        f"Sliding window eval: {len(sw_config['event_mapping'])} event types, "
                        f"step={sw_config['step_ms']}ms, "
                        f"epoch=[{sw_config['epoch_tmin']}, {sw_config['epoch_tmax']}]s"
                    )
                    result = await asyncio.to_thread(
                        evaluate_sliding_window,
                        decoder, loaded.data, loaded.events, loaded.sample_rate,
                        sw_config, ch_idx, _on_progress,
                    )
                    logger.info(
                        f"Sliding window eval done: {result.get('n_trials')} trials, "
                        f"acc={result.get('accuracy')}, "
                        f"timeline={len(result.get('timeline', []))} steps"
                    )
                else:
                    X, y = data.X, data.y
                    if ch_idx:
                        X = X[:, ch_idx, :]
                    result = await asyncio.to_thread(
                        evaluate_epochs, decoder, X, y, _on_progress,
                    )
            finally:
                progress_q.put(None)
                drain_task.cancel()

            return result

        await self._run_job(job_id, work, result_type="eval_metrics")

    # ------------------------------------------------------------------ #
    # Re-aggregate eval results with a different gate
    # ------------------------------------------------------------------ #

    def reaggregate_eval(self, job_id: int, config: dict[str, Any]) -> dict[str, Any]:
        """Re-run metric aggregation on stored eval results with a new gate."""
        from statistics import mean, median

        from dendrite.ml.decision_gate import DecisionGate
        from dendrite.ml.metrics_utils import calculate_itr, compute_trial_metrics

        result = self._get_job_result(job_id)
        if not result or result.get("mode") != "sliding_window":
            raise ValueError("Job has no sliding-window eval results")

        gate = DecisionGate.from_config(config)
        per_trial = result["per_trial"]
        bg_preds = result.get("background_preds", [])
        bg_confs = result.get("background_confs")
        n_classes = len({t["label"] for t in per_trial})
        eval_cfg = result.get("config", {})

        agg = compute_trial_metrics(
            per_trial, bg_preds, gate,
            num_classes=n_classes,
            step_duration_ms=eval_cfg.get("step_ms", 100),
            label_mapping=eval_cfg.get("label_mapping", {}),
            background_confs=bg_confs,
        )

        outcomes = agg.pop("trial_outcomes", [])
        for t, o in zip(per_trial, outcomes, strict=True):
            t.update(o)
        enrich_per_trial(per_trial)
        agg["per_trial"] = per_trial
        agg["gate"] = gate.to_dict()

        # Update event_markers correctness + TTD/ITR
        trial_correct = {t["trial"]: t["correct"] for t in per_trial}
        agg["event_markers"] = [
            {**m, "correct": trial_correct.get(i + 1, False)}
            for i, m in enumerate(result.get("event_markers", []))
        ]
        ttds = [t["ttd_ms"] for t in per_trial if t.get("ttd_ms") is not None]
        agg["ttd"] = {
            "mean_ms": round(mean(ttds), 1), "median_ms": round(median(ttds), 1),
            "min_ms": round(min(ttds), 1), "max_ms": round(max(ttds), 1),
            "n_detected": len(ttds), "n_total": len(per_trial),
        } if ttds else None
        mean_sel_s = mean(ttds) / 1000 if ttds else eval_cfg.get("trial_window_sec", 2.0)
        agg["itr_bits_per_min"] = round(calculate_itr(n_classes, agg["accuracy"], mean_sel_s), 2)
        return agg

    # ------------------------------------------------------------------ #
    # Benchmark (k-fold CV across multiple models)
    # ------------------------------------------------------------------ #

    async def start_benchmark(self, config: dict[str, Any]) -> dict[str, Any]:
        model_types = config.get("model_types", [])
        if not model_types:
            raise ValueError("No models selected")
        if self._loaded_data is None:
            raise ValueError("No data loaded - load data first")

        data = self._loaded_data
        return self._spawn_job(
            None, ",".join(model_types), config, "benchmark",
            lambda jid: self._run_bench(jid, config, data),
        )

    async def _run_bench(self, job_id: int, config: dict, data: Any) -> None:
        async def work():
            X, y = data.X, data.y
            ch_idx = config.get("channel_indices")
            if ch_idx:
                X = X[:, ch_idx, :]

            def _on_model(result):
                self._broadcast_from_thread("training", {
                    "type": "bench_model_complete", "job_id": job_id, "result": result,
                })

            all_results = await asyncio.to_thread(
                benchmark_cv, X, y, config["model_types"], config, _on_model,
            )
            return {"results": all_results, "n_folds": config.get("n_folds", 5)}

        await self._run_job(job_id, work)

    # ------------------------------------------------------------------ #
    # Utils
    # ------------------------------------------------------------------ #

    async def _broadcast(self, channel: str, data: dict) -> None:
        if self._bridge:
            await self._bridge.broadcast(channel, data)

    def cleanup_sync(self) -> None:
        for job_id, task in list(self._active_jobs.items()):
            task.cancel()
            self._job_repo.update_status(job_id, "cancelled", completed_at=_now_iso())
        self._active_jobs.clear()
        self._job_progress.clear()
        self._loaded_data = None
        self._eval_data = None
        self._shutdown_resources()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
