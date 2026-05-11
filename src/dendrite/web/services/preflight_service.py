"""
Preflight Service

Pre-start validation — checks all prerequisites before pipeline start.
Returns all failures at once so the frontend can display a complete checklist.
"""

import os

from pydantic import ValidationError

from dendrite.constants import STUDIES_DIR
from dendrite.web.schemas import PreflightCheck, PreflightResult, StudyConfig
from dendrite.web.services.config_service import ConfigService
from dendrite.web.services.mode_service import ModeService
from dendrite.web.services.stream_service import StreamService


class PreflightService:
    """Validates all prerequisites before pipeline start."""

    def __init__(
        self,
        stream_service: StreamService,
        mode_service: ModeService,
        config_service: ConfigService,
    ):
        self._stream_service = stream_service
        self._mode_service = mode_service
        self._config_service = config_service

    def run_preflight(self) -> PreflightResult:
        # Single LSL resolve for both liveness + channel count checks
        liveness, channel_mismatches = self._stream_service.check_streams()

        checks = [
            # Required — pipeline will crash without these
            self._check_data_stream(),
            self._check_streams_reachable(liveness),
            self._check_stream_channels(channel_mismatches),
            self._check_output_directory(),
            # Warnings — recommended but not strictly required to start
            self._check_decoder_files(),
            self._check_bids_fields(),
        ]
        return PreflightResult(
            ready=all(c.passed for c in checks if c.required),
            checks=checks,
        )

    def _check_streams_reachable(self, liveness: dict[str, bool]) -> PreflightCheck:
        if not self._stream_service.has_streams():
            return PreflightCheck(
                id="streams_reachable",
                label="All streams reachable",
                passed=True,
                detail="No streams to check",
            )
        offline = [uid for uid, alive in liveness.items() if not alive]
        passed = len(offline) == 0
        return PreflightCheck(
            id="streams_reachable",
            label="All streams reachable",
            passed=passed,
            detail=None if passed else f"{len(offline)} stream(s) offline",
        )

    def _check_stream_channels(
        self, mismatches: dict[str, tuple[int, int]]
    ) -> PreflightCheck:
        """Verify configured channel counts match live LSL streams."""
        if not self._stream_service.has_streams():
            return PreflightCheck(
                id="stream_channels",
                label="Stream channel counts match",
                passed=True,
                detail="No streams to check",
            )
        if not mismatches:
            return PreflightCheck(
                id="stream_channels",
                label="Stream channel counts match",
                passed=True,
            )
        details = [
            f"configured {cfg} ch, live {live} ch"
            for cfg, live in mismatches.values()
        ]
        return PreflightCheck(
            id="stream_channels",
            label="Stream channel counts match",
            passed=False,
            detail=f"Mismatch — {'; '.join(details)}. Rescan streams to fix.",
        )

    def _check_data_stream(self) -> PreflightCheck:
        streams = self._stream_service.get_streams()
        _event_types = {"events", "markers", "event", "marker"}
        has_data = any(
            s.sample_rate and s.sample_rate > 0 and s.type.lower() not in _event_types
            for s in streams.values()
        )
        return PreflightCheck(
            id="data_stream",
            label="At least one data stream present",
            passed=has_data,
            detail=None if has_data else (
                f"No data stream (sample_rate > 0) among {len(streams)} configured stream(s)"
                if streams else "No streams configured"
            ),
        )

    def _check_bids_fields(self) -> PreflightCheck:
        cs = self._config_service
        fields = {
            "study_name": cs.study_name,
            "subject_id": cs.subject_id,
            "session_id": cs.session_id,
            "recording_name": cs.recording_name,
        }

        try:
            StudyConfig(**fields)
        except ValidationError as e:
            msgs = [
                f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
                for err in e.errors()
            ]
            return PreflightCheck(
                id="bids_fields",
                label="BIDS metadata complete",
                passed=False,
                required=False,
                detail=f"Validation errors: {'; '.join(msgs)}",
            )

        return PreflightCheck(
            id="bids_fields",
            label="BIDS metadata complete",
            passed=True,
            required=False,
        )

    def _check_decoder_files(self) -> PreflightCheck:
        """Check that enabled async modes with file-based decoders have valid paths."""
        instances = self._mode_service.get_all_instances()
        missing = []
        for name, cfg in instances.items():
            if not cfg.get("enabled", True):
                continue
            if cfg.get("mode") != "asynchronous":
                continue
            if cfg.get("decoder_source") not in ("database", "pretrained"):
                continue
            decoder_path = cfg.get("decoder_config", {}).get("decoder_path")
            if not decoder_path:
                missing.append(f"{name}: no decoder path set")
            elif not os.path.isfile(decoder_path):
                missing.append(f"{name}: {decoder_path}")

        passed = len(missing) == 0
        return PreflightCheck(
            id="decoder_files",
            label="Decoder files accessible",
            passed=passed,
            required=False,
            detail=None if passed else f"Missing: {'; '.join(missing)}",
        )

    def run_mode_preflight(self, instance_name: str) -> PreflightResult:
        """Run preflight checks for a single mode instance.

        Used before starting a mode mid-session. Validates the mode config
        and any decoder file dependencies.
        """
        instance = self._mode_service.get_instance(instance_name)
        if instance is None:
            return PreflightResult(
                ready=False,
                checks=[
                    PreflightCheck(
                        id="mode_exists",
                        label="Mode instance exists",
                        passed=False,
                        detail=f"Instance '{instance_name}' not found",
                    )
                ],
            )

        checks = []
        mode = instance.get("mode", "")

        # Validate channel selection (required for all modes)
        ch_sel = instance.get("channel_selection") or {}
        has_channels = any(len(indices) > 0 for indices in ch_sel.values())
        checks.append(
            PreflightCheck(
                id="channel_selection",
                label="Channels selected",
                passed=has_channels,
                detail=None if has_channels else "No channels selected",
            )
        )
        bounds_ok, bounds_detail = self._validate_selection_bounds(instance)
        checks.append(
            PreflightCheck(
                id="channel_selection_bounds",
                label="Selected channels exist in stream",
                passed=bounds_ok,
                detail=bounds_detail,
            )
        )

        # Mode-specific config validation
        if mode == "synchronous":
            evt_map = instance.get("event_mapping") or {}
            has_events = len(evt_map) >= 2
            checks.append(
                PreflightCheck(
                    id="event_mapping",
                    label="Event mapping configured",
                    passed=has_events,
                    detail=None if has_events
                    else f"Need at least 2 event classes (have {len(evt_map)})",
                )
            )
            tmin = instance.get("epoch_tmin", 0.0)
            tmax = instance.get("epoch_tmax", 2.0)
            epoch_ok = tmax > tmin
            checks.append(
                PreflightCheck(
                    id="epoch_window",
                    label="Epoch window valid",
                    passed=epoch_ok,
                    detail=None if epoch_ok
                    else f"epoch_tmax ({tmax}) must be > epoch_tmin ({tmin})",
                )
            )

        elif mode == "asynchronous":
            wlen = instance.get("window_length_sec", 1.0)
            step = instance.get("step_size_ms", 100)
            timing_ok = wlen > 0 and step > 0
            checks.append(
                PreflightCheck(
                    id="async_timing",
                    label="Window/step timing valid",
                    passed=timing_ok,
                    detail=None if timing_ok
                    else f"window_length_sec ({wlen}) and step_size_ms ({step}) must be > 0",
                )
            )
            dec_src = instance.get("decoder_source", "database")
            if dec_src != "online":
                dec_cfg = instance.get("decoder_config") or {}
                has_model = bool(dec_cfg.get("model_config"))
                checks.append(
                    PreflightCheck(
                        id="decoder_config",
                        label="Decoder configured",
                        passed=has_model,
                        detail=None if has_model else "Decoder config missing model_config",
                    )
                )

        elif mode == "neurofeedback":
            wlen = instance.get("window_length_sec", 1.0)
            checks.append(
                PreflightCheck(
                    id="nfb_window",
                    label="Window length valid",
                    passed=wlen > 0,
                    detail=None if wlen > 0 else f"window_length_sec ({wlen}) must be > 0",
                )
            )
            feat = instance.get("feature_config") or {}
            bands = feat.get("target_bands") or {}
            band_errors = []
            for name, rng in bands.items():
                if not isinstance(rng, (list, tuple)) or len(rng) != 2:
                    band_errors.append(f"{name}: must be [low, high]")
                elif rng[0] >= rng[1]:
                    band_errors.append(f"{name}: low ({rng[0]}) must be < high ({rng[1]})")
            bands_ok = len(bands) > 0 and not band_errors
            detail = None
            if not bands:
                detail = "No frequency bands configured"
            elif band_errors:
                detail = "; ".join(band_errors)
            checks.append(
                PreflightCheck(
                    id="target_bands",
                    label="Frequency bands valid",
                    passed=bands_ok,
                    detail=detail,
                )
            )

        # Check decoder file for file-based async modes
        if mode == "asynchronous" and instance.get("decoder_source") in ("database", "pretrained"):
            decoder_path = instance.get("decoder_config", {}).get("decoder_path")
            passed = bool(decoder_path) and os.path.isfile(decoder_path)
            detail = (
                None if passed
                else "No decoder path set — will use online training" if not decoder_path
                else f"File not found: {decoder_path}"
            )
            checks.append(PreflightCheck(
                id="decoder_file",
                label="Decoder file accessible",
                passed=passed,
                required=False,
                detail=detail,
            ))

        return PreflightResult(
            ready=all(c.passed for c in checks if c.required),
            checks=checks,
        )

    def _validate_selection_bounds(self, instance: dict) -> tuple[bool, str | None]:
        """Verify channel_selection indices fit each modality's channel count.

        Catches saved configs whose indices no longer match the current stream
        layout (re-discovered stream with fewer channels, or different coord system).
        """
        ch_sel = instance.get("channel_selection") or {}
        if not ch_sel:
            return True, None
        source = instance.get("source_stream")
        entry = next(
            (
                e for e in self._stream_service.get_modalities_by_stream().values()
                if (not source or e.get("stream_key") == source)
                and any(mod in e.get("modalities", {}) for mod in ch_sel)
            ),
            None,
        )
        if entry is None:
            return True, None
        problems: list[str] = []
        for mod, indices in ch_sel.items():
            if not indices:
                continue
            n_ch = len(entry["modalities"].get(mod, []))
            bad = [i for i in indices if i < 0 or i >= n_ch]
            if bad:
                problems.append(
                    f"{mod}: {sorted(set(bad))[:3]} out of range "
                    f"({n_ch} channels available)"
                )
        return (not problems), ("; ".join(problems) if problems else None)

    def _check_output_directory(self) -> PreflightCheck:
        # Check that the studies base directory is writable.
        # Actual study subdirs are created at start time by PipelineService.
        check_path = STUDIES_DIR
        if not check_path.exists():
            check_path = check_path.parent

        writable = os.access(check_path, os.W_OK)
        return PreflightCheck(
            id="output_directory",
            label="Output directory writable",
            passed=writable,
            detail=None if writable else f"Directory not writable: {check_path}",
        )
