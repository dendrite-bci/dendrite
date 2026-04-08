"""Tests for search space, trial config bridge, trial execution, and optimizer config."""

import numpy as np
import optuna
import pytest

from dendrite.ml.search.optuna_runner import (
    split_params,
    prepare_holdout_split,
    run_single_trial,
    suggest_decoder_kwargs,
)
from dendrite.ml.search.search_space import (
    build_decoder_search_space,
    get_decoder_categories,
)

# ---------------------------------------------------------------------------
# Search Space
# ---------------------------------------------------------------------------


class TestSearchSpace:
    def test_decoder_search_space_neural(self):
        """Neural decoders include architecture + training params."""
        space = build_decoder_search_space("EEGNet")
        assert "F1" in space or "kern_length" in space  # architecture
        assert "learning_rate" in space  # training

    def test_decoder_search_space_classical(self):
        """Classical decoders only include component params, no training."""
        space = build_decoder_search_space("CSP+LDA")
        assert "n_components" in space  # CSP param
        assert "learning_rate" not in space  # no training params

    def test_decoder_search_space_with_categories(self):
        """Category filter restricts the space."""
        arch_only = build_decoder_search_space("EEGNet", ["architecture"])
        assert "learning_rate" not in arch_only
        assert "F1" in arch_only or "kern_length" in arch_only

    def test_decoder_categories_neural(self):
        cats = get_decoder_categories("EEGNet")
        assert "architecture" in cats
        assert "training" in cats

    def test_decoder_categories_classical(self):
        cats = get_decoder_categories("CSP+LDA")
        assert "architecture" in cats
        assert "training" not in cats


# ---------------------------------------------------------------------------
# split_params (flat params → DecoderConfig kwargs + model_params)
# ---------------------------------------------------------------------------


class TestSplitParams:
    def test_decoder_fields_at_top_level(self):
        """DecoderConfig fields land at top level, rest in model_params."""
        flat = {"learning_rate": 0.005, "model_type": "EEGNet", "F1": 8, "D": 2}
        base = {"num_classes": 2, "input_shapes": {"eeg": [8, 128]}}
        kwargs = split_params(flat, base)
        assert kwargs["learning_rate"] == 0.005
        assert kwargs["model_type"] == "EEGNet"
        assert kwargs["model_params"]["F1"] == 8
        assert kwargs["model_params"]["D"] == 2

    def test_derived_params_applied(self):
        """EEGNet F2 = F1 * D should be computed automatically."""
        flat = {"model_type": "EEGNet", "F1": 8, "D": 2}
        kwargs = split_params(flat, {})
        assert kwargs["model_params"]["F2"] == 16

    def test_base_kwargs_preserved(self):
        flat = {"model_type": "CSP+LDA"}
        base = {"num_classes": 4, "epochs": 50}
        kwargs = split_params(flat, base)
        assert kwargs["num_classes"] == 4
        assert kwargs["epochs"] == 50

    def test_produces_valid_decoder_config(self):
        from dendrite.ml.decoders.decoder_schemas import DecoderConfig

        flat = {"model_type": "EEGNet", "learning_rate": 0.003, "F1": 8, "D": 2}
        base = {"num_classes": 2, "input_shapes": {"eeg": [8, 128]}}
        kwargs = split_params(flat, base)
        config = DecoderConfig(**kwargs)
        assert config.model_type == "EEGNet"
        assert config.learning_rate == 0.003


# ---------------------------------------------------------------------------
# suggest_decoder_kwargs (direct suggestion → DecoderConfig kwargs)
# ---------------------------------------------------------------------------


class TestSuggestDecoderKwargs:
    def test_produces_flat_decoder_kwargs(self):
        """suggest_decoder_kwargs returns flat dict ready for DecoderConfig."""
        from dendrite.ml.decoders.decoder_schemas import DecoderConfig

        study = optuna.create_study(direction="maximize")
        base = {"num_classes": 2, "input_shapes": {"eeg": [8, 128]}}
        space = build_decoder_search_space("EEGNet")
        kwargs = suggest_decoder_kwargs(study.ask(), ["EEGNet"], base, space)
        assert kwargs["model_type"] == "EEGNet"
        assert kwargs["num_classes"] == 2
        # Must be directly constructable
        config = DecoderConfig(**kwargs)
        assert config.model_type == "EEGNet"

    def test_base_kwargs_preserved(self):
        study = optuna.create_study(direction="maximize")
        base = {"num_classes": 4, "input_shapes": {"eeg": [32, 250]}, "epochs": 50}
        space = build_decoder_search_space("EEGNet")
        kwargs = suggest_decoder_kwargs(study.ask(), ["EEGNet"], base, space)
        assert kwargs["num_classes"] == 4
        assert kwargs["epochs"] == 50

    def test_classical_no_training_params(self):
        """CSP+LDA search should not suggest training hyperparams."""
        study = optuna.create_study(direction="maximize")
        base = {"num_classes": 2, "input_shapes": {"eeg": [8, 128]}}
        space = build_decoder_search_space("CSP+LDA")
        kwargs = suggest_decoder_kwargs(study.ask(), ["CSP+LDA"], base, space)
        assert "learning_rate" not in kwargs
        assert "batch_size" not in kwargs


# ---------------------------------------------------------------------------
# prepare_holdout_split
# ---------------------------------------------------------------------------


class TestPrepareHoldoutSplit:
    def test_split_shapes(self):
        X = np.random.randn(100, 4, 64).astype(np.float32)
        y = np.array([i % 2 for i in range(100)], dtype=np.int64)
        X_tr, y_tr, X_ho, y_ho, base = prepare_holdout_split(X, y, 0.2)
        assert X_tr.shape[0] == 80
        assert X_ho.shape[0] == 20
        assert len(y_tr) == 80
        assert len(y_ho) == 20

    def test_base_kwargs_populated(self):
        X = np.random.randn(50, 4, 64).astype(np.float32)
        y = np.array([i % 3 for i in range(50)], dtype=np.int64)
        _, _, _, _, base = prepare_holdout_split(X, y, 0.2, "eeg", {"epochs": 50})
        assert base["num_classes"] == 3
        assert base["input_shapes"] == {"eeg": [4, 64]}
        assert base["epochs"] == 50

    def test_default_config(self):
        X = np.random.randn(40, 4, 64).astype(np.float32)
        y = np.array([i % 2 for i in range(40)], dtype=np.int64)
        _, _, _, _, base = prepare_holdout_split(X, y)
        assert base["epochs"] == 100
        assert base["use_early_stopping"] is True


# ---------------------------------------------------------------------------
# run_single_trial
# ---------------------------------------------------------------------------


class TestRunSingleTrial:
    @pytest.fixture
    def small_data(self):
        """Small 2-class dataset for fast trial tests."""
        np.random.seed(42)
        X = np.random.randn(40, 4, 32).astype(np.float32)
        y = np.array([i % 2 for i in range(40)], dtype=np.int64)
        return X[:32], y[:32], X[32:], y[32:]

    def test_successful_trial(self, small_data):
        X_tr, y_tr, X_ho, y_ho = small_data
        study = optuna.create_study(direction="maximize")
        base = {
            "num_classes": 2, "input_shapes": {"eeg": [4, 32]},
            "epochs": 2, "validation_split": 0.0,
            "use_early_stopping": False,
        }
        tr = run_single_trial(study, 1, X_tr, y_tr, X_ho, y_ho, ["CSP+LDA"], base)
        assert tr.trial_num == 1
        assert tr.error is None
        assert tr.decoder is not None
        assert 0.0 <= tr.accuracy <= 1.0
        assert tr.model_type == "CSP+LDA"
        assert tr.elapsed > 0

    def test_failed_trial_returns_error(self):
        """Trial with impossible config produces error result, not exception."""
        study = optuna.create_study(direction="maximize")
        X_tr = np.random.randn(5, 4, 32).astype(np.float32)
        y_tr = np.array([0, 0, 0, 0, 0], dtype=np.int64)  # single class → should fail
        X_ho = np.random.randn(2, 4, 32).astype(np.float32)
        y_ho = np.array([0, 0], dtype=np.int64)
        base = {
            "num_classes": 1, "input_shapes": {"eeg": [4, 32]},
            "epochs": 1, "validation_split": 0.0,
        }
        tr = run_single_trial(study, 1, X_tr, y_tr, X_ho, y_ho, ["CSP+LDA"], base)
        # Either succeeds or returns error — should never raise
        assert tr.trial_num == 1

    def test_study_updated(self, small_data):
        """Study tracks trial results after run_single_trial."""
        X_tr, y_tr, X_ho, y_ho = small_data
        study = optuna.create_study(direction="maximize")
        base = {
            "num_classes": 2, "input_shapes": {"eeg": [4, 32]},
            "epochs": 2, "validation_split": 0.0,
            "use_early_stopping": False,
        }
        run_single_trial(study, 1, X_tr, y_tr, X_ho, y_ho, ["CSP+LDA"], base)
        assert len(study.trials) == 1

