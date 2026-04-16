"""Tests for ML feature extractors (CSP)."""

import numpy as np
import pytest

from dendrite.ml.features.csp import CSPConfig, CSPModel

# --- CSPConfig ---


def test_csp_config_defaults():
    cfg = CSPConfig()
    assert cfg.n_components == 8
    assert cfg.reg == "ledoit_wolf"


def test_csp_config_custom():
    cfg = CSPConfig(n_components=4, reg="empirical")
    assert cfg.n_components == 4
    assert cfg.reg == "empirical"


def test_csp_config_has_hpo_metadata():
    extra = CSPConfig.model_fields["n_components"].json_schema_extra
    assert extra and "hpo" in extra


# --- CSPModel ---


@pytest.fixture
def csp_data():
    """Synthetic 2-class EEG: (40, 8, 64) — enough for CSP covariance."""
    rng = np.random.RandomState(42)
    X = rng.randn(40, 8, 64).astype(np.float32)
    y = np.array([0, 1] * 20)
    return X, y


def test_csp_model_info():
    info = CSPModel.get_model_info()
    assert info["model_type"] == "CSP"
    assert "eeg" in info["modalities"]
    assert info["component_type"] == "feature_extractor"


def test_csp_fit_transform(csp_data):
    X, y = csp_data
    csp = CSPModel(n_components=4)
    X_out = csp.fit_transform(X, y)
    assert X_out.shape == (40, 4)


def test_csp_default_components(csp_data):
    X, y = csp_data
    csp = CSPModel()  # default n_components=8
    X_out = csp.fit_transform(X, y)
    assert X_out.shape == (40, 8)


def test_csp_transform_after_fit(csp_data):
    X, y = csp_data
    csp = CSPModel(n_components=4)
    csp.fit(X, y)
    X_out = csp.transform(X[:5])
    assert X_out.shape == (5, 4)


def test_csp_components_capped_by_channels(csp_data):
    """n_components > n_channels should still work (MNE caps internally)."""
    X, y = csp_data  # 8 channels
    csp = CSPModel(n_components=8)
    X_out = csp.fit_transform(X, y)
    assert X_out.shape == (40, 8)


def test_csp_different_reg(csp_data):
    X, y = csp_data
    csp = CSPModel(n_components=4, reg="empirical")
    X_out = csp.fit_transform(X, y)
    assert X_out.shape == (40, 4)
