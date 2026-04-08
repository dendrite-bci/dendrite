"""Regression & benchmark tests for the ML pipeline.

Guards against silent accuracy regressions and evaluates the full
pipeline end-to-end: offline CV, sliding-window evaluation, online
learning curves, and live sync-mode processing.

Datasets
--------
Swarm study (local H5):
    ~99 epochs/session, 60 channels, 500 Hz, 2-class MI (circle/triangle).
    4 sessions parametrized independently.

MOABB BNCI2014_001 subject 1:
    576 epochs, 22 channels, 250 Hz, 4-class MI.

Run
---
    uv run pytest tests/test_ml/test_regression.py -v -s -m slow       # CV + learning curves
    uv run pytest tests/test_ml/test_regression.py -v -s -m benchmark  # eval pipeline
    uv run pytest tests/test_ml/test_regression.py::TestDwellDetectAny -v  # fast unit tests
"""

import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from dendrite.ml.decision_gate import dwell_detect_any
from dendrite.ml.decoders.decoder import Decoder
from dendrite.ml.decoders.decoder_schemas import DecoderConfig
from dendrite.ml.evaluation import evaluate_epochs, evaluate_sliding_window
from tests.conftest import SWARM_STUDY_ROOT

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED = 42

SWARM_DIR = SWARM_STUDY_ROOT / "data" / "raw" / "sub-01" / "training"
SWARM_FILES = sorted(SWARM_DIR.glob("*_eeg.h5")) if SWARM_DIR.exists() else []
SWARM_H5 = SWARM_DIR / "sub-01_ses-01_task-training_run-01_20260304_112150_eeg.h5"

MOABB_DATASET = "BNCI2014_001"
MOABB_SUBJECT = 1

EVENT_MAPPING = {1: "circle", 2: "triangle"}
EPOCH_TMIN = 0.5
EPOCH_TMAX = 4.5
BANDPASS = (8, 30)

MODEL_TYPES = ["CSP+LDA", "EEGNet"]
MODEL_TYPES_FULL = ["CSP+LDA", "EEGNet", "BDEEGNet"]
NEURAL_EPOCHS = {"CSP+LDA": 1, "EEGNet": 200, "BDEEGNet": 200}

# Thresholds (conservative — above chance)
ACC_SWARM = 0.30
ACC_MOABB = 0.40
ACC_STEP_MIN = 0.15
STEP_EPOCH_RATIO_MIN = 0.2
TIME_CLASSICAL = 10.0
TIME_NEURAL = 120.0

TRAINING_INTERVAL = 10
BENCHMARK_DIR = Path(__file__).parent.parent / "benchmarks"
PLOT_DIR = Path(__file__).parent / "regression_plots"

# Module-level collectors — filled during test run, plotted at session end
_cv_results: list[dict] = []
_preq_curves: dict[str, list[tuple[int, float]]] = {}
_eval_results: dict[str, list[dict]] = {}  # dataset_name → list of per-model rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_all():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _session_id(path: Path) -> str:
    m = re.search(r"ses-(\d+)", path.name)
    return f"ses-{m.group(1)}" if m else path.stem


def _load_swarm_session(path: Path):
    """Load a single swarm session with full preprocessing (matches production).

    Bad channel detection + interpolation, bandpass, CAR, epoch extraction.
    """
    from dendrite.data.loaders import load_file
    from dendrite.data.quality import detect_bad_channels

    loaded = load_file(str(path))
    loaded.filter_modality("eeg")

    bad_result = detect_bad_channels(loaded.data)
    bad_indices = bad_result["bad_channels"]

    loaded.preprocess(
        {"lowcut": BANDPASS[0], "highcut": BANDPASS[1], "apply_rereferencing": True},
        bad_channels={"eeg": bad_indices} if bad_indices else None,
    )
    return loaded


def _epoch_loaded(loaded, event_mapping=None, tmin=None, tmax=None):
    """Epoch a preprocessed loaded recording. Returns (X, y_encoded)."""
    epoched = loaded.epoch({
        "epoch_tmin": tmin or EPOCH_TMIN,
        "epoch_tmax": tmax or EPOCH_TMAX,
        "event_mapping": event_mapping or EVENT_MAPPING,
    })
    X, y = epoched.X, epoched.y
    label_map = {code: i for i, code in enumerate(sorted(set(y.tolist())))}
    y_encoded = np.array([label_map[label] for label in y], dtype=np.int64)
    return X, y_encoded


def _cv_score(model_type, X, y, n_folds=5, epochs=50, pipeline_steps=None):
    """Stratified k-fold CV. Returns (mean_acc, std_acc, elapsed, fold_accs)."""
    num_classes = int(np.max(y) + 1)
    input_shapes = {"eeg": list(X.shape[1:])}

    _seed_all()
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_accs = []
    start = time.time()

    for train_idx, test_idx in cv.split(X, y):
        config = DecoderConfig(
            model_type=model_type, num_classes=num_classes,
            input_shapes=input_shapes, epochs=epochs, batch_size=32,
            learning_rate=0.001, validation_split=0.0,
            use_early_stopping=False, pipeline_steps=pipeline_steps,
        )
        decoder = Decoder(config)
        decoder.input_shapes = input_shapes
        decoder.fit(X[train_idx], y[train_idx])
        fold_accs.append(float(decoder.score(X[test_idx], y[test_idx])))

    elapsed = time.time() - start
    return np.mean(fold_accs), np.std(fold_accs), elapsed, fold_accs


def _train_decoder(model_type, X_train, y_train, X_val, y_val, full_config=False):
    """Train a single decoder, return (decoder, val_acc, elapsed)."""
    num_classes = int(np.max(y_train) + 1)
    input_shapes = {"eeg": list(X_train.shape[1:])}
    epochs = NEURAL_EPOCHS.get(model_type, 30)

    config = DecoderConfig(
        model_type=model_type, num_classes=num_classes,
        input_shapes=input_shapes, epochs=epochs, batch_size=32,
        learning_rate=0.001,
        validation_split=0.2 if full_config else 0.0,
        use_early_stopping=full_config, early_stopping_patience=25,
        use_lr_scheduler=full_config, lr_scheduler_type="OneCycleLR",
        use_class_weights=full_config,
    )
    decoder = Decoder(config)
    decoder.input_shapes = input_shapes

    start = time.time()
    decoder.fit(X_train, y_train)
    elapsed = time.time() - start
    val_acc = float(decoder.score(X_val, y_val))
    return decoder, val_acc, elapsed


def _run_sliding_eval(decoder, loaded, event_mapping, sample_rate, code_to_class=None):
    """Run sliding window eval with dwell + majority. Returns (dwell, majority, elapsed)."""
    base = {
        "step_ms": 50, "epoch_tmin": EPOCH_TMIN, "epoch_tmax": EPOCH_TMAX,
        "event_mapping": event_mapping, "dwell_n": 10,
    }
    if code_to_class:
        base["code_to_class"] = code_to_class

    start = time.time()
    dwell = evaluate_sliding_window(
        decoder, loaded.data, loaded.events, sample_rate,
        {**base, "detection_strategy": "dwell"},
    )
    majority = evaluate_sliding_window(
        decoder, loaded.data, loaded.events, sample_rate,
        {**base, "detection_strategy": "majority"},
    )
    elapsed = time.time() - start

    for r in (dwell, majority):
        per_trial = r.get("per_trial", [])
        r["mean_step_acc"] = round(
            float(np.mean([t["step_accuracy"] for t in per_trial])) if per_trial else 0.0, 4,
        )
    return dwell, majority, elapsed


def _save_benchmark(dataset: str, results: list[dict]) -> Path:
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    path = BENCHMARK_DIR / f"eval_{dataset}_{ts}.json"
    path.write_text(json.dumps({
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset": dataset, "epoch_window": [EPOCH_TMIN, EPOCH_TMAX],
        "results": results,
    }, indent=2))
    return path


def _print_table(dataset: str, rows: list[dict]):
    header = (
        f"{'Model':<12} {'Val Acc':>8} {'Epoch Acc':>10} {'Step Acc':>9} "
        f"{'Dwell Acc':>10} {'Maj Acc':>9} {'TTD ms':>8} {'Train s':>8} {'Eval s':>8}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}\n  Eval Benchmark: {dataset}  (epoch {EPOCH_TMIN}–{EPOCH_TMAX}s)\n{sep}")
    print(header)
    print(sep)
    for r in rows:
        ttd_s = f"{r['ttd_mean_ms']:.0f}" if r.get("ttd_mean_ms") else "—"
        step_s = f"{r['step_acc']:.1%}" if r.get("step_acc") is not None else "—"
        dwell_s = f"{r['dwell_acc']:.1%}" if r.get("dwell_acc") is not None else "—"
        maj_s = f"{r['majority_acc']:.1%}" if r.get("majority_acc") is not None else "—"
        eval_t = r.get("eval_epoch_time_s", 0) + r.get("eval_sliding_time_s", 0)
        print(
            f"{r['model_type']:<12} {r['val_acc']:>7.1%} {r['epoch_acc']:>9.1%} "
            f"{step_s:>9} {dwell_s:>10} {maj_s:>9} {ttd_s:>8} "
            f"{r['train_time_s']:>7.1f} {eval_t:>7.1f}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", params=SWARM_FILES, ids=[_session_id(f) for f in SWARM_FILES])
def swarm_data(request):
    """Load one swarm session, parametrized over all available H5 files."""
    if not SWARM_FILES:
        pytest.skip(f"No swarm files in {SWARM_DIR}")
    loaded = _load_swarm_session(request.param)
    return _epoch_loaded(loaded)


@pytest.fixture(scope="module")
def moabb_data():
    """Load MOABB BNCI2014_001 subject 1."""
    try:
        from dendrite.data.loaders._training_data import load_moabb_for_training
    except ImportError:
        pytest.skip("MOABB not available")
    try:
        data = load_moabb_for_training({
            "dataset_code": MOABB_DATASET, "subject": MOABB_SUBJECT,
            "paradigm": "MotorImagery",
        })
    except Exception as e:
        pytest.skip(f"MOABB loading failed: {e}")
    return data.X, data.y


@pytest.fixture(scope="module")
def swarm_context():
    """Single swarm session with 80/20 train-test split for eval benchmarks."""
    if not SWARM_H5.exists():
        pytest.skip(f"Swarm file not found: {SWARM_H5}")
    loaded = _load_swarm_session(SWARM_H5)
    X, y = _epoch_loaded(loaded)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(splitter.split(X, y))
    return {
        "X_train": X[train_idx], "y_train": y[train_idx],
        "X_test": X[test_idx], "y_test": y[test_idx],
        "loaded": loaded, "event_mapping": EVENT_MAPPING,
        "sample_rate": loaded.sample_rate,
    }


@pytest.fixture(scope="module")
def swarm_pooled_context():
    """Cross-session: train on sessions 1..N-1, eval on session N."""
    if not SWARM_DIR.exists():
        pytest.skip(f"Swarm dir not found: {SWARM_DIR}")
    h5_files = sorted(SWARM_DIR.glob("*_eeg.h5"))
    if len(h5_files) < 2:
        pytest.skip(f"Need ≥2 sessions, found {len(h5_files)}")

    label_map = {code: i for i, code in enumerate(sorted(EVENT_MAPPING.keys()))}
    code_to_class = label_map

    all_X, all_y = [], []
    for h5 in h5_files[:-1]:
        loaded = _load_swarm_session(h5)
        X_sess, y_sess = _epoch_loaded(loaded)
        all_X.append(X_sess)
        all_y.append(y_sess)

    X_train = np.concatenate(all_X)
    y_train = np.concatenate(all_y)

    eval_loaded = _load_swarm_session(h5_files[-1])
    X_test, y_test = _epoch_loaded(eval_loaded)

    print(f"Cross-session: train {len(h5_files) - 1} sessions ({len(y_train)} epochs), "
          f"eval session {len(h5_files)} ({len(y_test)} epochs)")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "eval_loaded": eval_loaded, "event_mapping": EVENT_MAPPING,
        "code_to_class": code_to_class, "sample_rate": eval_loaded.sample_rate,
    }


# ---------------------------------------------------------------------------
# 1. CV Accuracy Regression (per-session)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestRegressionSwarm:

    def test_csp_lda(self, swarm_data, request):
        X, y = swarm_data
        mean_acc, std_acc, elapsed, folds = _cv_score("CSP+LDA", X, y, n_folds=5, epochs=1)
        print(f"CSP+LDA swarm: {mean_acc:.1%} +/- {std_acc:.1%} ({elapsed:.1f}s)")
        _cv_results.append({"dataset": request.node.callspec.id, "model": "CSP+LDA",
                            "mean": mean_acc, "std": std_acc, "folds": folds})
        assert mean_acc > ACC_SWARM
        assert elapsed < TIME_CLASSICAL * 5

    def test_eegnet(self, swarm_data, request):
        X, y = swarm_data
        mean_acc, std_acc, elapsed, folds = _cv_score("EEGNet", X, y, n_folds=5, epochs=30)
        print(f"EEGNet swarm: {mean_acc:.1%} +/- {std_acc:.1%} ({elapsed:.1f}s)")
        _cv_results.append({"dataset": request.node.callspec.id, "model": "EEGNet",
                            "mean": mean_acc, "std": std_acc, "folds": folds})
        assert mean_acc > ACC_SWARM
        assert elapsed < TIME_NEURAL * 5


@pytest.mark.slow
class TestRegressionMOABB:

    def test_csp_lda(self, moabb_data):
        X, y = moabb_data
        mean_acc, std_acc, elapsed, folds = _cv_score("CSP+LDA", X, y, n_folds=5, epochs=1)
        print(f"CSP+LDA MOABB: {mean_acc:.1%} +/- {std_acc:.1%} ({elapsed:.1f}s)")
        _cv_results.append({"dataset": "MOABB", "model": "CSP+LDA",
                            "mean": mean_acc, "std": std_acc, "folds": folds})
        assert mean_acc > ACC_MOABB
        assert elapsed < TIME_CLASSICAL * 5

    def test_eegnet(self, moabb_data):
        X, y = moabb_data
        mean_acc, std_acc, elapsed, folds = _cv_score("EEGNet", X, y, n_folds=5, epochs=30)
        print(f"EEGNet MOABB: {mean_acc:.1%} +/- {std_acc:.1%} ({elapsed:.1f}s)")
        _cv_results.append({"dataset": "MOABB", "model": "EEGNet",
                            "mean": mean_acc, "std": std_acc, "folds": folds})
        assert mean_acc > ACC_MOABB
        assert elapsed < TIME_NEURAL


# ---------------------------------------------------------------------------
# 2. Scaler Ablation
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestScalerAblation:

    def test_eegnet_scaler_comparison(self, swarm_data):
        X, y = swarm_data
        acc_on, _, _, _ = _cv_score("EEGNet", X, y, n_folds=5, epochs=30, pipeline_steps=["scaler", "classifier"])
        acc_off, _, _, _ = _cv_score("EEGNet", X, y, n_folds=5, epochs=30, pipeline_steps=["classifier"])
        print(f"EEGNet  scaler=True: {acc_on:.1%}  scaler=False: {acc_off:.1%}  diff: {(acc_off - acc_on):+.1%}")

    def test_fgmdm_baseline(self, swarm_data):
        X, y = swarm_data
        acc, std, elapsed, _ = _cv_score("FgMDM", X, y, n_folds=5, epochs=1)
        print(f"FgMDM: {acc:.1%} +/- {std:.1%} ({elapsed:.1f}s)")
        assert acc > ACC_SWARM


# ---------------------------------------------------------------------------
# 3. Eval Pipeline Benchmark (epoch + sliding window)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
class TestEvalBenchmarkSwarm:

    def test_benchmark(self, swarm_context):
        ctx = swarm_context
        rows = []
        for model_type in MODEL_TYPES:
            decoder, val_acc, train_time = _train_decoder(
                model_type, ctx["X_train"], ctx["y_train"],
                ctx["X_test"], ctx["y_test"], full_config=True,
            )
            epoch_result = evaluate_epochs(decoder, ctx["X_test"], ctx["y_test"])
            dwell, majority, sliding_time = _run_sliding_eval(
                decoder, ctx["loaded"], ctx["event_mapping"], ctx["sample_rate"],
            )
            ttd = dwell.get("ttd", {})
            row = {
                "model_type": model_type, "val_acc": val_acc,
                "epoch_acc": epoch_result["accuracy"],
                "step_acc": dwell["mean_step_acc"],
                "dwell_acc": dwell["accuracy"],
                "majority_acc": majority["accuracy"],
                "ttd_mean_ms": ttd.get("mean_ms"),
                "train_time_s": round(train_time, 2),
                "eval_epoch_time_s": 0, "eval_sliding_time_s": round(sliding_time, 2),
            }
            rows.append(row)

        _print_table("Swarm", rows)
        _eval_results["Swarm"] = rows
        print(f"Results saved to: {_save_benchmark('swarm', rows)}")

        for row in rows:
            assert row["epoch_acc"] > ACC_SWARM
            assert row["step_acc"] > ACC_STEP_MIN
            ratio = row["step_acc"] / max(row["epoch_acc"], 0.01)
            assert ratio > STEP_EPOCH_RATIO_MIN
            if row["ttd_mean_ms"] is not None:
                assert row["ttd_mean_ms"] > 0


@pytest.mark.benchmark
class TestEvalBenchmarkMOABB:

    def test_benchmark(self, moabb_data):
        X, y = moabb_data
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        train_idx, test_idx = next(splitter.split(X, y))
        rows = []
        for model_type in MODEL_TYPES:
            decoder, val_acc, train_time = _train_decoder(
                model_type, X[train_idx], y[train_idx], X[test_idx], y[test_idx],
            )
            epoch_result = evaluate_epochs(decoder, X[test_idx], y[test_idx])
            rows.append({
                "model_type": model_type, "val_acc": val_acc,
                "epoch_acc": epoch_result["accuracy"],
                "step_acc": None, "dwell_acc": None, "majority_acc": None,
                "ttd_mean_ms": None, "train_time_s": round(train_time, 2),
                "eval_epoch_time_s": 0, "eval_sliding_time_s": 0,
            })

        _print_table("MOABB", rows)
        _eval_results["MOABB"] = rows
        print(f"Results saved to: {_save_benchmark('moabb', rows)}")
        for row in rows:
            assert row["epoch_acc"] > ACC_SWARM


# ---------------------------------------------------------------------------
# 4. Cross-Session Eval (pooled)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
class TestEvalPooled:

    def test_benchmark(self, swarm_pooled_context):
        ctx = swarm_pooled_context
        rows = []
        for model_type in MODEL_TYPES_FULL:
            decoder, val_acc, train_time = _train_decoder(
                model_type, ctx["X_train"], ctx["y_train"],
                ctx["X_test"], ctx["y_test"], full_config=True,
            )
            epoch_result = evaluate_epochs(decoder, ctx["X_test"], ctx["y_test"])
            dwell, majority, sliding_time = _run_sliding_eval(
                decoder, ctx["eval_loaded"], ctx["event_mapping"],
                ctx["sample_rate"], code_to_class=ctx["code_to_class"],
            )
            ttd = dwell.get("ttd", {})
            rows.append({
                "model_type": model_type, "val_acc": val_acc,
                "epoch_acc": epoch_result["accuracy"],
                "step_acc": dwell["mean_step_acc"],
                "dwell_acc": dwell["accuracy"],
                "majority_acc": majority["accuracy"],
                "ttd_mean_ms": ttd.get("mean_ms"),
                "train_time_s": round(train_time, 2),
                "eval_epoch_time_s": 0, "eval_sliding_time_s": round(sliding_time, 2),
            })

        _print_table("Swarm Cross-Session", rows)
        _eval_results["Cross-Session"] = rows
        print(f"Results saved to: {_save_benchmark('swarm_cross_session', rows)}")
        for row in rows:
            assert row["epoch_acc"] > ACC_SWARM


# ---------------------------------------------------------------------------
# 5. Online Learning Curves
# ---------------------------------------------------------------------------


def _prequential_curve(path: Path) -> list[tuple[int, float]]:
    """Prequential accuracy curve using the production training path."""
    from dendrite.data.loaders import load_file
    from dendrite.data.loaders._training_data import load_epochs
    from dendrite.data.quality import detect_bad_channels
    from dendrite.ml.training.runner import decoder_config_from_dict, train_decoder
    loaded = load_file(str(path))
    loaded.filter_modality("eeg")
    bad_indices = detect_bad_channels(loaded.data)["bad_channels"]

    request = {
        "modalities": ["eeg"],
        "event_mapping": EVENT_MAPPING,
        "label_mapping": {"circle": 0, "triangle": 1},
        "epoch_tmin": EPOCH_TMIN, "epoch_tmax": EPOCH_TMAX,
        "mode_preprocessing": {
            "eeg": {"lowcut": BANDPASS[0], "highcut": BANDPASS[1], "apply_rereferencing": True},
        },
        "effective_bad": {"eeg": bad_indices} if bad_indices else {},
    }

    epoched = load_epochs(request, str(path))
    X, y = epoched.X, epoched.y
    num_classes = int(np.max(y) + 1)
    input_shapes = {"eeg": list(X.shape[1:])}

    curve = []
    for n_train in range(TRAINING_INTERVAL, len(y), TRAINING_INTERVAL):
        if len(set(y[:n_train].tolist())) < 2:
            continue
        _seed_all()
        config = decoder_config_from_dict({"model_type": "CSP+LDA"}, num_classes, input_shapes)
        decoder = train_decoder(X[:n_train], y[:n_train], config)
        acc = float(decoder.score(X[n_train:], y[n_train:]))
        curve.append((n_train, acc))
    return curve


@pytest.mark.slow
class TestOnlineLearningCurve:

    def test_learning_curves(self):
        if not SWARM_FILES:
            pytest.skip(f"No swarm files in {SWARM_DIR}")
        for path in SWARM_FILES:
            sid = _session_id(path)
            curve = _prequential_curve(path)
            _preq_curves[sid] = curve
            print(f"\n{sid} (CSP+LDA):")
            for n, acc in curve:
                print(f"  {n:3d} trials: {acc:.1%} {'#' * int(acc * 40)}")

        for _sid, curve in _preq_curves.items():
            assert curve[-1][1] > ACC_SWARM


# ---------------------------------------------------------------------------
# 6. Sync Mode Pipeline (epoch extraction + decoder accuracy)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestDirectModeProcessing:

    def test_sync_mode_epoch_extraction(self):
        if not SWARM_FILES:
            pytest.skip(f"No swarm files in {SWARM_DIR}")

        import logging
        import multiprocessing
        import queue as queue_mod
        from unittest.mock import MagicMock

        from dendrite.data.loaders import load_file
        from dendrite.processing.modes.mode_utils import Buffer, FanOutQueue
        from dendrite.processing.modes.synchronous_mode import SynchronousMode

        path = SWARM_FILES[0]
        loaded = load_file(str(path))
        loaded.filter_modality("eeg")
        n_channels = loaded.n_channels
        sr = loaded.sample_rate

        main_q = queue_mod.Queue()
        mode = SynchronousMode.__new__(SynchronousMode)
        mode.logger = logging.getLogger("test_direct_sync")
        mode.stop_event = multiprocessing.Event()
        mode.output_queue = FanOutQueue([main_q])
        mode.prediction_queue = queue_mod.Queue()
        mode.training_queue = MagicMock()
        mode.shared_state = MagicMock()
        mode.shared_state.get.return_value = None
        mode.mode_name = "direct_sync"
        mode.mode_type = "synchronous"
        mode.file_identifier = "test"
        mode.study_name = "test"
        mode.channel_selection = {"eeg": list(range(n_channels))}
        mode.modality_labels = {"eeg": loaded.channel_names}
        mode.modalities = ["eeg"]
        mode.sample_rate = sr
        mode.effective_sample_rate = sr
        mode.event_mapping = EVENT_MAPPING
        mode.label_mapping = {"circle": 0, "triangle": 1}
        mode.reverse_label_mapping = {0: "circle", 1: "triangle"}
        mode.epoch_tmin = EPOCH_TMIN
        mode.epoch_tmax = EPOCH_TMAX
        mode.tmin_samples = int(EPOCH_TMIN * sr)
        mode.tmax_samples = int(EPOCH_TMAX * sr)
        mode.epoch_length_samples = int((EPOCH_TMAX - EPOCH_TMIN) * sr)
        mode.training_interval = 999
        mode.decoder_source = "online"
        mode.epoch_count = 0
        mode.current_sample_index = 0
        mode.last_lsl_timestamp = 0.0
        mode.pending_epochs = []
        mode.decoder = None
        mode.decoder_config = {}
        mode._training_pending = False
        mode._pending_decoder_load = None
        mode._sample_preprocessor = None
        mode._reader = None
        mode.metrics_manager = None
        mode._rb_config = None
        mode._mode_type = "synchronous"
        mode._gpu_last_emit_time = 0.0

        buf_size = mode.tmin_samples + mode.epoch_length_samples + int(sr * 2)
        mode.buffer = Buffer(modalities=["eeg"], buffer_size=buf_size, logger=mode.logger)

        data = loaded.data
        events = loaded.events
        event_map = {idx: code for idx, code in events}

        for i in range(data.shape[1]):
            marker = event_map.get(i, 0)
            mode._process_data({
                "eeg": data[:, i:i + 1].astype(np.float32),
                "markers": np.array([[marker]], dtype=np.float32),
                "lsl_timestamp": i / sr,
                "_receive_ns": int(i / sr * 1e9),
            })

        matched_events = sum(1 for _, code in events if code in mode.event_mapping)
        print(f"\nDirect sync mode: {mode.epoch_count} epochs from {matched_events} matched events")
        assert mode.epoch_count > 0
        assert mode.epoch_count >= matched_events * 0.5

    def test_sync_mode_with_decoder(self):
        """Train decoder, feed through sync mode, measure prediction accuracy."""
        if not SWARM_FILES:
            pytest.skip(f"No swarm files in {SWARM_DIR}")

        import logging
        import multiprocessing
        import queue as queue_mod
        from unittest.mock import MagicMock

        from dendrite.data.loaders import load_file
        from dendrite.data.quality import detect_bad_channels
        from dendrite.ml.training.runner import decoder_config_from_dict, train_decoder
        from dendrite.processing.modes.mode_utils import Buffer, FanOutQueue, SamplePreprocessor
        from dendrite.processing.modes.synchronous_mode import SynchronousMode

        path = SWARM_FILES[0]
        loaded_for_training = _load_swarm_session(path)
        X, y = _epoch_loaded(loaded_for_training)
        num_classes = int(np.max(y) + 1)
        _seed_all()

        config = decoder_config_from_dict(
            {"model_type": "CSP+LDA"}, num_classes, {"eeg": list(X.shape[1:])},
        )
        decoder = train_decoder(X, y, config)
        assert decoder.is_fitted

        # Reload raw data for online processing
        loaded = load_file(str(path))
        loaded.filter_modality("eeg")
        n_channels = loaded.n_channels
        sr = loaded.sample_rate
        bad_indices = detect_bad_channels(loaded.data)["bad_channels"]

        main_q = queue_mod.Queue()
        pred_q = queue_mod.Queue()
        mode = SynchronousMode.__new__(SynchronousMode)
        mode.logger = logging.getLogger("test_direct_sync_decoder")
        mode.stop_event = multiprocessing.Event()
        mode.output_queue = FanOutQueue([main_q])
        mode.prediction_queue = pred_q
        mode.training_queue = MagicMock()
        mode.shared_state = MagicMock()
        quality_data = {
            "bad_channels": {"eeg": bad_indices},
            "effective_bad": {"eeg": bad_indices},
            "interp_version": 1,
        } if bad_indices else {}
        mode.shared_state.get.side_effect = lambda key, *a: (
            quality_data if key == "channel_quality" else None
        )
        mode.mode_name = "direct_sync"
        mode.mode_type = "synchronous"
        mode.file_identifier = "test"
        mode.study_name = "test"
        mode.channel_selection = {"eeg": list(range(n_channels))}
        mode.modality_labels = {"eeg": loaded.channel_names}
        mode.modalities = ["eeg"]
        mode.sample_rate = sr
        mode.event_mapping = EVENT_MAPPING
        mode.label_mapping = {"circle": 0, "triangle": 1}
        mode.reverse_label_mapping = {0: "circle", 1: "triangle"}
        mode.epoch_tmin = EPOCH_TMIN
        mode.epoch_tmax = EPOCH_TMAX
        mode.tmin_samples = int(EPOCH_TMIN * sr)
        mode.tmax_samples = int(EPOCH_TMAX * sr)
        mode.epoch_length_samples = int((EPOCH_TMAX - EPOCH_TMIN) * sr)
        mode.training_interval = 999
        mode.decoder_source = "database"
        mode.decoder = decoder
        mode.index_to_event_code = {0: 1, 1: 2}
        mode.epoch_count = 0
        mode.current_sample_index = 0
        mode.last_lsl_timestamp = 0.0
        mode.pending_epochs = []
        mode.decoder_config = {}
        mode._training_pending = False
        mode._pending_decoder_load = None
        mode._reader = None
        mode.metrics_manager = None
        mode._rb_config = None
        mode._mode_type = "synchronous"
        mode._gpu_last_emit_time = 0.0

        preproc_config = {"eeg": {
            "lowcut": BANDPASS[0], "highcut": BANDPASS[1],
            "apply_rereferencing": True, "filter_order": 4,
            "channel_labels": loaded.channel_names,
        }}
        mode._sample_preprocessor = SamplePreprocessor(
            preproc_config=preproc_config, sample_rate=sr,
            channel_selection=mode.channel_selection,
            modality_labels=mode.modality_labels,
            shared_state=mode.shared_state, logger=mode.logger,
        )
        mode.effective_sample_rate = mode._sample_preprocessor.effective_sample_rate

        buf_size = mode.tmin_samples + mode.epoch_length_samples + int(sr * 2)
        mode.buffer = Buffer(modalities=["eeg"], buffer_size=buf_size, logger=mode.logger)

        data = loaded.data
        events = loaded.events
        event_map = {idx: code for idx, code in events}
        code_to_class = {1: 0, 2: 1}

        # Warmup: trigger lazy preprocessor init, then freeze interpolation
        for i in range(10):
            mode._process_data({
                "eeg": data[:, i:i + 1].astype(np.float32),
                "markers": np.array([[0]], dtype=np.float32),
                "lsl_timestamp": i / sr, "_receive_ns": int(i / sr * 1e9),
            })
        if bad_indices and mode._sample_preprocessor._preprocessor:
            eeg_proc = mode._sample_preprocessor._preprocessor.processors.get("eeg")
            if eeg_proc:
                eeg_proc.freeze_interpolation(bad_indices)

        for i in range(data.shape[1]):
            marker = event_map.get(i, 0)
            mode._process_data({
                "eeg": data[:, i:i + 1].astype(np.float32),
                "markers": np.array([[marker]], dtype=np.float32),
                "lsl_timestamp": i / sr, "_receive_ns": int(i / sr * 1e9),
            })

        predictions = []
        while not pred_q.empty():
            pkt = pred_q.get_nowait()
            if pkt.get("type") == "prediction":
                d = pkt["data"]
                true_class = mode.label_mapping.get(d["true_event"])
                pred_class = code_to_class.get(d["prediction"])
                if true_class is not None and pred_class is not None:
                    predictions.append((pred_class, true_class))

        n_preds = len(predictions)
        assert n_preds > 0, f"No predictions (epochs={mode.epoch_count})"
        n_correct = sum(1 for p, t in predictions if p == t)
        accuracy = n_correct / n_preds

        print(f"\nSync mode pipeline: {accuracy:.1%} ({n_correct}/{n_preds} epochs)")
        _cv_results.append({
            "dataset": "SyncMode", "model": "CSP+LDA (online)",
            "mean": accuracy, "std": 0.0, "folds": [accuracy],
        })
        assert accuracy > ACC_SWARM


# ---------------------------------------------------------------------------
# 7. Decision Gate Unit Tests (fast, no data)
# ---------------------------------------------------------------------------


class TestDwellDetectAny:

    def test_empty_input(self):
        assert dwell_detect_any([], 3) == 0

    def test_single_element(self):
        assert dwell_detect_any([1], 1) == 0

    def test_all_same_class(self):
        assert dwell_detect_any([1] * 10, 3) == 3

    def test_alternating_no_detections(self):
        assert dwell_detect_any([0, 1] * 20, 2) == 0

    def test_exact_streak(self):
        assert dwell_detect_any([0, 0, 0, 1, 1, 1], 3) == 2

    def test_gated_predictions_break_streak(self):
        assert dwell_detect_any([1, 1, -1, 1, 1], 3) == 0

    def test_dwell_n_1(self):
        assert dwell_detect_any([1, 1, 1, 1], 1) == 3

    def test_mixed_classes(self):
        assert dwell_detect_any([0, 0, 0, 1, 1, 1, 2, 2, 2], 3) == 3

    def test_large_dwell_no_detection(self):
        assert dwell_detect_any([0, 0, 1, 1, 0, 0], 3) == 0


# ---------------------------------------------------------------------------
# 8. Report Generation
# ---------------------------------------------------------------------------


def _generate_plots():
    """Generate regression benchmark report — clean minimal style."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    PLOT_DIR.mkdir(exist_ok=True)

    has_cv = bool(_cv_results)
    has_curves = bool(_preq_curves)
    has_eval = bool(_eval_results)
    if not has_cv and not has_curves and not has_eval:
        return

    model_colors = {
        "CSP+LDA": "#2563eb", "EEGNet": "#d97706", "BDEEGNet": "#0891b2",
        "FgMDM": "#16a34a", "CSP+LDA (online)": "#c026d3",
    }
    metric_colors = {
        "epoch": "#3b82f6", "step": "#f59e0b", "dwell": "#10b981", "majority": "#a855f7",
    }

    n_rows = 3 if has_eval else 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4.2 * n_rows))

    def style_ax(ax, title):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#d1d5db")
        ax.spines["bottom"].set_color("#d1d5db")
        ax.tick_params(labelsize=8, colors="#374151")
        ax.set_title(title, fontsize=10, fontweight="600", color="#111827", pad=8)
        ax.grid(axis="y", color="#e5e7eb", lw=0.6)
        ax.set_axisbelow(True)

    def chance_line(ax):
        ax.axhline(0.5, color="#9ca3af", ls="--", lw=0.7)

    def empty_ax(ax, msg="No data"):
        style_ax(ax, "")
        ax.text(0.5, 0.5, msg, ha="center", va="center", color="#9ca3af",
                fontsize=10, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

    def _eval_bars(ax, rows, title):
        """Shared grouped-bar logic for eval cascade panels."""
        style_ax(ax, title)
        models = [r["model_type"] for r in rows]
        x = np.arange(len(models))
        bar_w = 0.18
        metrics = [
            ("epoch", "epoch_acc"), ("step", "step_acc"),
            ("dwell", "dwell_acc"), ("majority", "majority_acc"),
        ]
        for k, (label, key) in enumerate(metrics):
            vals = [r.get(key) for r in rows]
            valid_x = [xi for xi, v in zip(x, vals, strict=True) if v is not None]
            valid_v = [v for v in vals if v is not None]
            if valid_v:
                offset = (k - 1.5) * bar_w
                c = metric_colors[label]
                ax.bar([xi + offset for xi in valid_x], valid_v, bar_w * 0.85,
                       color=c, alpha=0.75, label=label.capitalize())
                for xi, v in zip(valid_x, valid_v, strict=True):
                    ax.text(xi + offset, v + 0.015, f"{v:.0%}", ha="center", va="bottom",
                            fontsize=6, fontweight="bold", color=c)
        chance_line(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=8)
        ax.set_ylim(0, 1.12)
        ax.legend(fontsize=7, ncol=4, loc="upper right", framealpha=0.8)

    # ── Panel A: CV Accuracy ──
    ax = axes[0, 0]
    style_ax(ax, "Cross-Validation Accuracy")
    if has_cv:
        cv_offline = [r for r in _cv_results if r["dataset"] != "SyncMode"]
        datasets = list(dict.fromkeys(r["dataset"] for r in cv_offline))
        models = list(dict.fromkeys(r["model"] for r in cv_offline))
        x = np.arange(len(datasets))
        width = 0.8 / max(len(models), 1)
        for i, model in enumerate(models):
            for j, ds in enumerate(datasets):
                match = [r for r in cv_offline if r["dataset"] == ds and r["model"] == model]
                if not match:
                    continue
                m, folds = match[0]["mean"], match[0]["folds"]
                offset = (i - (len(models) - 1) / 2) * width
                xpos = j + offset
                c = model_colors.get(model, f"C{i}")
                ax.bar(xpos, m, width * 0.85, color=c, alpha=0.2, edgecolor=c, lw=1)
                ax.scatter([xpos] * len(folds), folds, s=20, color=c, alpha=0.7,
                           zorder=3, edgecolors="white", linewidths=0.3)
                ax.text(xpos, m + 0.025, f"{m:.0%}", ha="center", va="bottom",
                        fontsize=7, fontweight="bold", color=c)
        chance_line(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=7, rotation=25, ha="right")
        ax.set_ylabel("5-Fold CV Accuracy", fontsize=9, color="#374151")
        ax.set_ylim(0, 1.12)
        handles = [Line2D([0], [0], marker="o", color=model_colors.get(m, "C0"), ls="",
                          markersize=5, label=m) for m in models]
        ax.legend(handles=handles, fontsize=7, loc="lower right", framealpha=0.8)
    else:
        empty_ax(ax, "No CV results")

    # ── Panel B: Session Consistency ──
    ax = axes[0, 1]
    style_ax(ax, "Session Consistency")
    if has_cv:
        cv_offline = [r for r in _cv_results if r["dataset"] != "SyncMode"]
        session_names = [ds for ds in dict.fromkeys(r["dataset"] for r in cv_offline)
                         if ds != "MOABB"]
        models = list(dict.fromkeys(r["model"] for r in cv_offline))
        sync_results = [r for r in _cv_results if r["dataset"] == "SyncMode"]

        if len(session_names) >= 2:
            for i, model in enumerate(models):
                accs = [next((r["mean"] for r in cv_offline
                              if r["dataset"] == ds and r["model"] == model), None)
                        for ds in session_names]
                valid = [(j, a) for j, a in enumerate(accs) if a is not None]
                if not valid:
                    continue
                c = model_colors.get(model, f"C{i}")
                ax.plot([v[0] for v in valid], [v[1] for v in valid], "o-",
                        color=c, label=f"{model} (CV)", markersize=5, lw=1.5)
            if sync_results:
                c = model_colors["CSP+LDA (online)"]
                ax.axhline(sync_results[0]["mean"], color=c, ls="-.", lw=1.3, alpha=0.7)
                ax.text(len(session_names) - 0.5, sync_results[0]["mean"] + 0.02,
                        f"Online {sync_results[0]['mean']:.0%}",
                        fontsize=7, color=c, ha="right")
            chance_line(ax)
            ax.set_xticks(range(len(session_names)))
            ax.set_xticklabels(session_names, fontsize=8)
            ax.set_ylabel("Accuracy", fontsize=9, color="#374151")
            ax.set_ylim(0.3, 1.05)
            ax.legend(fontsize=7, framealpha=0.8)
        else:
            empty_ax(ax, "Need >=2 sessions")
    else:
        empty_ax(ax, "No CV results")

    # ── Row 2: Eval Pipeline (only if has_eval) ──
    if has_eval:
        eval_single = _eval_results.get("Swarm", []) + _eval_results.get("MOABB", [])
        eval_cross = _eval_results.get("Cross-Session", [])

        if eval_single:
            _eval_bars(axes[1, 0], eval_single, "Eval Pipeline: Accuracy Cascade")
        else:
            empty_ax(axes[1, 0], "No single-session eval")

        if eval_cross:
            _eval_bars(axes[1, 1], eval_cross, "Cross-Session Generalization")
            ttds = [(r["model_type"], r["ttd_mean_ms"]) for r in eval_cross
                    if r.get("ttd_mean_ms") is not None]
            if ttds:
                ttd_str = "  ".join(f"{m}: {t:.0f}ms" for m, t in ttds)
                axes[1, 1].text(0.02, 0.02, f"TTD: {ttd_str}", transform=axes[1, 1].transAxes,
                                fontsize=7, color="#6b7280", va="bottom")
        else:
            empty_ax(axes[1, 1], "No cross-session eval")

    # ── Bottom row: Learning Curves + Timing ──
    bottom_row = 2 if has_eval else 1

    ax = axes[bottom_row, 0]
    style_ax(ax, "Online Learning Curves (CSP+LDA)")
    if has_curves:
        curve_colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(_preq_curves)))
        for (label, curve), c in zip(_preq_curves.items(), curve_colors, strict=True):
            trials = [n for n, _ in curve]
            accs = [a for _, a in curve]
            ax.plot(trials, accs, "o-", color=c, label=label, markersize=3, lw=1.5)
            ax.fill_between(trials, 0.5, accs, color=c, alpha=0.05)
        chance_line(ax)
        ax.set_xlabel("Training Trials", fontsize=9, color="#374151")
        ax.set_ylabel("Accuracy (Remaining)", fontsize=9, color="#374151")
        ax.set_ylim(0.3, 1.0)
        ax.legend(fontsize=7, framealpha=0.8)
    else:
        empty_ax(ax, "No learning curve data")

    ax = axes[bottom_row, 1]
    style_ax(ax, "Training & Eval Timing")
    all_eval = [r for rows in _eval_results.values() for r in rows]
    if all_eval:
        models = list(dict.fromkeys(r["model_type"] for r in all_eval))
        x = np.arange(len(models))
        bar_w = 0.35
        train_times = [np.mean([r["train_time_s"] for r in all_eval
                                if r["model_type"] == m]) for m in models]
        eval_times = [np.mean([r.get("eval_sliding_time_s", 0) for r in all_eval
                               if r["model_type"] == m]) for m in models]
        ax.bar(x - bar_w / 2, train_times, bar_w * 0.85, color="#3b82f6", alpha=0.7,
               label="Train")
        ax.bar(x + bar_w / 2, eval_times, bar_w * 0.85, color="#f59e0b", alpha=0.7,
               label="Eval (sliding)")
        for xi, t in zip(x, train_times, strict=True):
            ax.text(xi - bar_w / 2, t + 0.3, f"{t:.1f}s", ha="center", fontsize=7,
                    color="#374151")
        for xi, t in zip(x, eval_times, strict=True):
            if t > 0:
                ax.text(xi + bar_w / 2, t + 0.3, f"{t:.1f}s", ha="center", fontsize=7,
                        color="#374151")
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=8)
        ax.set_ylabel("Time (seconds)", fontsize=9, color="#374151")
        ax.legend(fontsize=7, framealpha=0.8)
    else:
        empty_ax(ax, "No timing data")

    fig.suptitle("Dendrite Regression Benchmark", fontsize=13, fontweight="600",
                 color="#111827", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(PLOT_DIR / "regression_report.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlots saved to {PLOT_DIR}/")


@pytest.mark.slow
@pytest.mark.benchmark
class TestGenerateReport:
    def test_zz_generate_plots(self):
        if not _cv_results and not _preq_curves and not _eval_results:
            pytest.skip("No results collected")
        _generate_plots()
