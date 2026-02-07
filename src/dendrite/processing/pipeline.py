"""
Processing Pipeline for Dendrite System

This module contains the main processing pipeline that runs in a separate process
to manage all data processing, including data acquisition, preprocessing, mode processing,
and data saving.
"""

import json
import logging
import signal
import time

from dendrite.constants import (
    DEFAULT_RECORDING_NAME,
    DEFAULT_STUDY_NAME,
    TIMEOUT_DATA_ACQUISITION,
    TIMEOUT_DATA_PROCESSOR,
    TIMEOUT_DATA_SAVER,
    TIMEOUT_METRICS_SAVER,
    TIMEOUT_MODE_PROCESS,
    VERSION,
    get_study_paths,
)
from dendrite.data.acquisition import DataAcquisition, DataRecord
from dendrite.data.storage.data_saver import DataSaver
from dendrite.data.storage.metrics_saver import MetricsSaver
from dendrite.processing.modes import create_mode
from dendrite.processing.pipeline_config import PipelineConfig
from dendrite.processing.processor import DataProcessor
from dendrite.utils.logger_central import get_logger, setup_logger

# Module logger
logger = get_logger("MainProcessor")


def _cleanup_processes(processes: dict, logger) -> None:
    """Clean up processes with appropriate timeouts, terminating if necessary."""
    for name, (process, timeout) in processes.items():
        if process and process.is_alive():
            process.join(timeout=timeout)
            if process.is_alive():
                logger.warning(f"{name} did not stop cleanly within {timeout}s timeout")
                process.terminate()
                process.join(timeout=1)


def run_pipeline(config: PipelineConfig) -> None:
    """
    Start the Dendrite processing pipeline in a separate process.

    Sets up data acquisition, preprocessing, mode processing, and data saving.
    Uses synchronized distribution to ensure all modes receive identical samples.
    """
    setup_logger("MainProcessor", level=logging.DEBUG)
    logger.info("Pipeline starting")

    # Get stop_event early for signal handlers
    stop_event = config.stop_event

    # Set up signal handlers for clean shutdown on SIGTERM/SIGINT
    def _signal_handler(signum, frame):
        try:
            signal_name = signal.Signals(signum).name
        except ValueError:
            signal_name = f"SIGNAL({signum})"
        logger.info(f"Received {signal_name}, initiating shutdown...")
        stop_event.set()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    try:
        sample_rate = config.sample_rate
        mode_instance_configs = config.mode_instances
        file_identifier = config.file_identifier
        data_queue = config.data_queue
        save_queue = config.save_queue
        plot_queue = config.plot_queue
        prediction_queue = config.prediction_queue
        mode_output_queues = config.mode_output_queues
        pid_queue = config.pid_queue
        shared_state = config.shared_state

        preprocess_data = config.preprocess_data
        preprocessing_config = (
            {
                "preprocess_data": config.preprocess_data,
                "modality_preprocessing": config.modality_preprocessing,
                "quality_control": config.quality_control,
            }
            if preprocess_data
            else {}
        )

        # Prepare directories
        study_name = config.study_name
        paths = get_study_paths(study_name)
        paths["raw"].mkdir(parents=True, exist_ok=True)
        paths["metrics"].mkdir(parents=True, exist_ok=True)
        raw_data_filename = str(paths["raw"] / f"{file_identifier}_eeg.h5")
        metrics_filename = str(paths["metrics"] / f"{file_identifier}_metrics.h5")

        # Metadata for saving
        script_metadata = {
            "version": VERSION,
            "sample_rate": sample_rate,
            "recording_name": file_identifier,
            "mode_instances": {},
        }

        # Log config summary
        modality_count = len(preprocessing_config.get("modality_preprocessing", {}))
        logger.info(f"Preprocessing: {modality_count} modalities configured")

        # Spawn DataAcquisition & DataSaver
        stream_configs = config.stream_configs
        logger.debug(f"Using stream configurations: {stream_configs}")

        data_acquisition = DataAcquisition(
            data_queue=data_queue,
            save_queue=save_queue,
            stop_event=stop_event,
            stream_configs=stream_configs,
            shared_state=shared_state,
        )
        data_acquisition.daemon = True
        data_acquisition.start()

        data_saver = DataSaver(
            filename=raw_data_filename, save_queue=save_queue, stop_event=stop_event
        )
        data_saver.daemon = True
        data_saver.start()

        # Send global metadata (BIDS fields) to saver
        global_metadata = {
            "version": VERSION,
            "sample_rate": sample_rate,
            "study_name": config.study_name,
            "recording_name": config.recording_name,
            "subject_id": config.subject_id,
            "session_id": config.session_id,
            "run_number": config.run_number,
            "experiment_description": config.experiment_description,
        }
        now = time.time()
        metadata_record = DataRecord(
            modality="Metadata",
            sample=json.dumps(global_metadata),
            timestamp=now,
            local_timestamp=now,
        )
        save_queue.put(metadata_record)
        logger.debug("Global BIDS metadata sent to saver")

        data_processor = DataProcessor(
            data_queue=data_queue,
            plot_queue=plot_queue,
            stop_event=stop_event,
            mode_configs=mode_instance_configs,
            stream_configs=stream_configs,
            preprocessing_config=preprocessing_config,
            shared_state=shared_state,
        )
        data_processor_proc = data_processor.start()
        logger.debug("DataProcessor started - directly distributing samples to all mode queues")

        processing_processes = []

        for instance_name, instance_config in mode_instance_configs.items():
            mode = instance_config["mode"]

            event_mapping_for_mode = instance_config.get("event_mapping", {})

            script_metadata["mode_instances"][instance_name] = {
                "model_type": instance_config.get("model_type", "EEGNet"),
                "event_mapping": event_mapping_for_mode,
                "mode_settings": instance_config,
            }

            mode_input_queue = data_processor.get_queue_for_mode(instance_name)
            if not mode_input_queue:
                logger.error(f"No input queue found for mode instance '{instance_name}'. Skipping.")
                continue

            logger.info(f"[{instance_name}] Creating mode '{mode}'")
            logger.debug(f"[{instance_name}] Event Mapping: {event_mapping_for_mode}")

            try:
                # Prepare enhanced instance config with runtime parameters
                enhanced_config = instance_config.copy()
                enhanced_config.update(
                    {
                        "file_identifier": file_identifier,
                        "study_name": config.study_name,
                        "preprocessing_config": preprocessing_config if preprocess_data else None,
                    }
                )

                # Add sync-to-async sharing for synchronous modes
                if mode.lower() == "synchronous":
                    linked_async_modes = config.sync_to_async_sharing.get(
                        instance_name, []
                    )
                    enhanced_config["linked_async_modes"] = linked_async_modes
                    logger.debug(
                        f"[{instance_name}] Sync mode will share models with: {linked_async_modes}"
                    )

                mode_params = {
                    "data_queue": mode_input_queue,
                    "output_queue": mode_output_queues[instance_name],
                    "stop_event": stop_event,
                    "instance_config": enhanced_config,
                    "sample_rate": sample_rate,
                    "prediction_queue": prediction_queue,
                    "shared_state": shared_state,
                }

                logger.debug(
                    f"[{instance_name}] Using channel_selection: {enhanced_config.get('channel_selection')}"
                )
                proc = create_mode(mode, **mode_params)
                proc.daemon = True
                proc.start()

                if proc.is_alive():
                    logger.info(f"Processing mode started successfully for {instance_name}")
                    processing_processes.append(proc)

                    # Send PID back to main window for resource monitoring
                    if pid_queue:
                        try:
                            pid_queue.put({"mode_name": instance_name, "pid": proc.pid})
                        except Exception as e:
                            logger.warning(f"Failed to send PID for {instance_name}: {e}")
                else:
                    logger.error(f"FAILED - Processing mode failed to start for {instance_name}")

            except Exception as e:
                error_msg = (
                    f"CRITICAL ERROR: Failed to start mode '{mode}' for {instance_name}: {e!s}"
                )
                logger.error(error_msg, exc_info=True)

        # Metrics data saver
        metrics_queues = {}
        for instance_name, fanout_queue in mode_output_queues.items():
            metrics_queues[instance_name] = fanout_queue.primary_queue

        metrics_saver = MetricsSaver(
            filename=metrics_filename,
            stop_event=stop_event,
            script_metadata=script_metadata,
            mode_metrics_queues=metrics_queues,
            shared_state=shared_state,
        )
        metrics_saver.daemon = True
        metrics_saver.start()

        stop_event.wait()

        processes_to_clean = {
            "Data acquisition": (data_acquisition, TIMEOUT_DATA_ACQUISITION),
            "Data saver": (data_saver, TIMEOUT_DATA_SAVER),
            "Data processor": (data_processor_proc, TIMEOUT_DATA_PROCESSOR),
            "Metrics saver": (metrics_saver, TIMEOUT_METRICS_SAVER),
        }

        for p in processing_processes:
            processes_to_clean[f"Mode {p.name}"] = (p, TIMEOUT_MODE_PROCESS)

        _cleanup_processes(processes_to_clean, logger)
        logger.info("Pipeline stopped")

    except Exception as e:
        logger.error(f"CRITICAL ERROR in pipeline: {e}", exc_info=True)
