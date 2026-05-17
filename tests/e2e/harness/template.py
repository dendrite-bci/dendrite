"""Build a PipelineConfig dict for one session.

Both paths start from the vendored `template_config.json` — a known-good
2-mode (BenchSync + BenchAsync) config — and substitute the session-specific
streams / channels / decoder settings:

  local  -- rename the stream blocks to match the H5's replay outlets, keep
            the in-house channel layout and event mapping from the template.
  moabb  -- discover the channel layout + event map from the MOABB dataset
            itself (via MOABBLoader) and rebuild the stream blocks fresh.
"""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path

from . import config
from .datasets import SessionSpec


def _load_template() -> dict:
    return json.loads(config.TEMPLATE_CONFIG.read_text())


def _decoder_config(n_classes: int, sample_rate: float) -> dict:
    """CSP+LDA decoder block, forced for reproducibility."""
    return {
        "decoder_type": "Decoder",
        "pipeline_steps": None,
        "model_config": {
            "model_type": "CSP+LDA",
            "num_classes": n_classes,
            "device": "auto",
            "seed": 42,
            "epochs": 1,
            "batch_size": 32,
            "validation_split": 0.0,
            "use_early_stopping": False,
            "use_class_weights": True,
            "class_weight_strategy": "balanced",
            "use_lr_scheduler": False,
            "model_params": {},
            "input_shapes": None,
            "pipeline_steps": None,
            "event_mapping": None,
            "label_mapping": None,
            "sample_rate": sample_rate,
            "epoch_tmin": None,
            "epoch_tmax": None,
        },
    }


def build_config(
    spec: SessionSpec, pretrained_decoder_path: str | Path | None = None
) -> dict:
    """Construct a PipelineConfig dict for `POST /api/config/load`.

    With `pretrained_decoder_path`, a third mode instance
    (`BenchAsync_Pretrained`, `decoder_source: "database"`) is added — it loads
    that decoder instead of training online, exercising the pretrained-load /
    cross-subject path in the same run.
    """
    cfg = _build_moabb_config(spec) if spec.key == "moabb" else _build_local_config(spec)
    if pretrained_decoder_path is not None:
        _add_pretrained_mode(cfg, Path(pretrained_decoder_path))
    return cfg


def _add_pretrained_mode(cfg: dict, decoder_path: Path) -> None:
    """Add a `BenchAsync_Pretrained` mode that loads a pretrained decoder.

    Clones the `BenchAsync` block, switches it to `decoder_source: "database"`
    pointing at `decoder_path`, and sizes its window to the decoder's trained
    epoch length (read from the decoder `.json` meta) so the mode's
    channel/time-sample validation passes.
    """
    meta = json.loads(decoder_path.read_text())
    eeg_shape = (meta.get("input_shapes") or {}).get("eeg")
    sample_rate = float(meta.get("sample_rate") or 0.0)

    pre = copy.deepcopy(cfg["mode_instances"][config.ASYNC_MODE_NAME])
    pre["name"] = config.PRETRAINED_MODE_NAME
    pre["decoder_source"] = "database"
    pre["decoder_config"]["decoder_path"] = str(decoder_path)
    if eeg_shape and len(eeg_shape) > 1 and sample_rate > 0:
        pre["window_length_sec"] = eeg_shape[1] / sample_rate
    cfg["mode_instances"][config.PRETRAINED_MODE_NAME] = pre


# --- local (in-house H5) ----------------------------------------------------


def _build_local_config(spec: SessionSpec) -> dict:
    assert spec.h5_path is not None
    cfg = _load_template()
    stem = spec.h5_path.stem
    parts = dict(p.split("-", 1) for p in stem.split("_") if "-" in p)

    cfg["study_name"] = spec.study_name
    cfg["subject_id"] = parts.get("sub", "01")
    cfg["session_id"] = parts.get("ses", "01")
    cfg["recording_name"] = config.RECORDING_NAME
    cfg["experiment_description"] = (
        f"e2e replay of {spec.h5_path.name} (CSP+LDA, sync train + async eval)"
    )

    new_eeg_sid = f"replay_{spec.eeg_stream_name.lower()}"
    for sc in cfg["stream_configs"]:
        if sc.get("type") in ("EEG", "eeg"):
            sc["name"] = spec.eeg_stream_name
            sc["source_id"] = new_eeg_sid
        elif sc.get("type") == "Events":
            sc["name"] = spec.events_stream_name
            sc["source_id"] = f"events_{stem}"

    # modalities_by_stream is keyed by the EEG source_id — rekey it to match.
    old_key = next(iter(cfg["modalities_by_stream"]))
    mbs = cfg["modalities_by_stream"].pop(old_key)
    mbs["stream_name"] = spec.eeg_stream_name
    cfg["modalities_by_stream"] = {new_eeg_sid: mbs}

    for mode in cfg["mode_instances"].values():
        mode["study_name"] = spec.study_name

    return cfg


# --- moabb ------------------------------------------------------------------


@lru_cache(maxsize=4)
def _moabb_raw_layout(
    preset: str, subject: int,
) -> tuple[tuple[str, ...], tuple[str, ...], float]:
    """(channel_names, channel_types, sample_rate) for the MOABB preset/subject.

    Loaded via MOABBLoader so the names line up sample-for-sample with what
    `ReplayStreamer._replay_moabb()` broadcasts. The sample rate isn't in the
    MOABB registry metadata, so it's read off the raw here. First call triggers
    the one-time MOABB download to ~/mne_data/.
    """
    from dendrite.data.loaders import MOABBLoader, get_moabb_dataset_info

    info = get_moabb_dataset_info(preset)
    if not info:
        raise ValueError(f"Unknown MOABB preset: {preset}")
    loader = MOABBLoader(info["config"])
    raw = loader.load_as_raw(subject)
    return tuple(raw.channel_names), tuple(raw.channel_types), float(raw.sample_rate)


def _moabb_event_mapping(preset: str) -> dict[str, str]:
    """Map marker code (as str) -> event name, from MOABB's events dict."""
    from dendrite.data.loaders import get_moabb_dataset_info

    info = get_moabb_dataset_info(preset)
    if not info:
        raise ValueError(f"Unknown MOABB preset: {preset}")
    return {str(code): name for name, code in info["events"].items()}


def _build_moabb_config(spec: SessionSpec) -> dict:
    assert spec.moabb_preset is not None and spec.moabb_subject is not None
    cfg = _load_template()
    preset, subject = spec.moabb_preset, spec.moabb_subject
    ch_names, ch_types, sample_rate = _moabb_raw_layout(preset, subject)
    event_mapping = _moabb_event_mapping(preset)

    cfg["study_name"] = spec.study_name
    cfg["subject_id"] = f"{subject:02d}"
    cfg["session_id"] = spec.moabb_session or "all"
    cfg["recording_name"] = config.RECORDING_NAME
    cfg["experiment_description"] = (
        f"e2e MOABB replay: {preset} sub-{subject:02d} (CSP+LDA, sync train + async eval)"
    )

    eeg_sid = f"replay_{spec.eeg_stream_name.lower()}"
    ch_types_up = [t.upper() for t in ch_types]
    units = ["µV" if t in ("EEG", "EOG", "EMG", "ECG") else "a.u." for t in ch_types_up]
    eeg_cfg = {
        "name": spec.eeg_stream_name,
        "type": "EEG",
        "channel_count": len(ch_names),
        "sample_rate": sample_rate,
        "channel_format": "float32",
        "source_id": eeg_sid,
        "labels": list(ch_names),
        "channel_types": ch_types_up,
        "channel_units": units,
        "stream_key": "EEG",
        "acquisition_info": {},
    }
    events_cfg = {
        "name": spec.events_stream_name,
        "type": "Events",
        "channel_count": 1,
        "sample_rate": 0.0,
        "channel_format": "string",
        "source_id": f"moabb_events_{preset}",
        "labels": ["Ch_1"],
        "channel_types": ["Markers"],
        "channel_units": ["n/a"],
        "stream_key": "Events",
        "acquisition_info": {},
    }
    cfg["stream_configs"] = [events_cfg, eeg_cfg]

    eeg_ch: list[dict] = []
    eog_ch: list[dict] = []
    for idx, (label, t) in enumerate(zip(ch_names, ch_types_up, strict=True)):
        entry = {"label": label, "local_index": idx}
        (eog_ch if t == "EOG" else eeg_ch).append(entry)
    cfg["modalities_by_stream"] = {
        eeg_sid: {
            "stream_name": spec.eeg_stream_name,
            "stream_type": "EEG",
            "stream_key": "EEG",
            "sample_rate": sample_rate,
            "modalities": {
                **({"eog": eog_ch} if eog_ch else {}),
                "eeg": eeg_ch,
            },
        }
    }
    eeg_indices = [m["local_index"] for m in eeg_ch]

    sync = cfg["mode_instances"][config.SYNC_MODE_NAME]
    sync["study_name"] = spec.study_name
    sync["channel_selection"] = {"eeg": eeg_indices}
    sync["event_mapping"] = event_mapping
    sync["epoch_tmin"] = spec.epoch_tmin
    sync["epoch_tmax"] = spec.epoch_tmax
    sync["decoder_config"] = _decoder_config(spec.n_classes, sample_rate)

    async_inst = cfg["mode_instances"][config.ASYNC_MODE_NAME]
    async_inst["study_name"] = spec.study_name
    async_inst["channel_selection"] = {"eeg": eeg_indices}
    async_inst["event_mapping"] = event_mapping
    async_inst["decoder_config"] = _decoder_config(spec.n_classes, sample_rate)

    return cfg


def write_temp_config(cfg: dict, spec: SessionSpec) -> Path:
    """Write the config to a deterministic path under the harness output dir."""
    config.CONFIG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = config.CONFIG_OUTPUT_DIR / f"_e2e_{spec.label}.json"
    out.write_text(json.dumps(cfg, indent=2, default=str))
    return out
