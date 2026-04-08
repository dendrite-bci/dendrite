"""Event alignment test: compare FIF source events with H5 pipeline output.

Loads a FIF file (ground truth) and an H5 recording (pipeline output),
and verifies that events survived the pipeline with correct codes and timing.
Also includes ERP regression test (correct vs incorrect shape events).

Usage: replay ses-01 FIF through the pipeline, then run this test.
"""

import os

import h5py
import matplotlib.pyplot as plt
import mne
import numpy as np
import pytest

from tests.conftest import DENDRITE_DATA, SWARM_STUDY_ROOT

# ── Paths ────────────────────────────────────────────────────────────

SWARM_DERIV = SWARM_STUDY_ROOT / "data" / "derivatives" / "sub-01" / "training"

# (FIF original, H5 pipeline replay, label)
REPLAY_PAIRS = [
    (
        SWARM_DERIV / "sub-01_ses-01_task-training_run-01_20260304_112150_eeg_raw.fif",
        DENDRITE_DATA / "default_study" / "raw" / "task-recording_run-21_20260407_190150_raw.h5",
        "ses-01",
    ),
]

# Legacy aliases (used by event alignment tests)
FIF_PATH = REPLAY_PAIRS[0][0]
H5_PATH = REPLAY_PAIRS[0][1]

# Max allowed timing drift between matched events (seconds)
TOLERANCE_S = 0.05  # 50ms — allows for LSL transport jitter


def _load_fif_events(path: str) -> list[tuple[float, int, str]]:
    """Load events from FIF. Returns [(relative_time_s, code, name), ...]."""
    raw = mne.io.read_raw_fif(path, preload=False, verbose=False)
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    sfreq = raw.info["sfreq"]

    id_to_name = {v: k for k, v in event_id.items()}
    result = []
    t0 = events[0][0] / sfreq if len(events) > 0 else 0
    for sample_idx, _, code in events:
        t_rel = (sample_idx / sfreq) - t0
        result.append((t_rel, code, id_to_name.get(code, f"event_{code}")))
    return result


def _load_h5_events(path: str) -> list[tuple[float, int, str]]:
    """Load events from H5 DataSaver output. Returns [(relative_time_s, code, name), ...]."""
    with h5py.File(path, "r") as f:
        ev = f["Event"]
        result = []
        ts0 = float(ev[0]["timestamp"]) if len(ev) > 0 else 0
        for e in ev:
            t_rel = float(e["timestamp"]) - ts0
            code = int(e["event_id"])
            name = e["event_type"]
            if isinstance(name, bytes):
                name = name.decode()
            result.append((t_rel, code, name))
    return result


def _match_events(
    fif_events: list[tuple[float, int, str]],
    h5_events: list[tuple[float, int, str]],
    tolerance_s: float = TOLERANCE_S,
) -> tuple[list[tuple[float, float, float]], list[int], list[int], np.poly1d | None]:
    """Match FIF<->H5 events with linear clock-drift compensation.

    Two-pass approach:
    1. Coarse pass with generous tolerance to estimate linear drift
       (H5 replay clocks drift vs FIF original clock).
    2. Remove drift from H5 times, re-match with strict tolerance.

    Returns:
        matched: list of (fif_time, h5_time_raw, residual_s) tuples
        unmatched_fif: indices of FIF events with no match
        unmatched_h5: indices of H5 events with no match
        drift_poly: fitted drift polynomial (h5_corrected = h5_raw - drift_poly(fif_t))
    """
    from collections import defaultdict
    from scipy.optimize import linear_sum_assignment

    def _group_by_name(events):
        groups: dict[str, list[int]] = defaultdict(list)
        for i, (_, _, name) in enumerate(events):
            groups[name].append(i)
        return groups

    def _assign(fif_idx, h5_idx, fif_t, h5_t, tol):
        """Hungarian assignment on per-name groups. Returns matched (fi, hi) pairs."""
        fif_by_name = _group_by_name([fif_events[i] for i in fif_idx])
        h5_by_name = _group_by_name([h5_events[j] for j in h5_idx])
        # Remap: local group index -> original index
        fif_local = {name: [fif_idx[i] for i in idxs] for name, idxs in
                     defaultdict(list, {n: [fif_idx.index(fi) for fi in fis]
                                        for n, fis in
                                        _group_by_name([fif_events[i] for i in fif_idx]).items()}).items()}
        # Simpler: just re-group the original indices
        fif_groups: dict[str, list[int]] = defaultdict(list)
        h5_groups: dict[str, list[int]] = defaultdict(list)
        for i in fif_idx:
            fif_groups[fif_events[i][2]].append(i)
        for j in h5_idx:
            h5_groups[h5_events[j][2]].append(j)

        pairs = []
        for name in fif_groups:
            fi = fif_groups[name]
            hi = h5_groups.get(name, [])
            if not hi:
                continue
            cost = np.empty((len(fi), len(hi)))
            for r, f in enumerate(fi):
                for c, h in enumerate(hi):
                    cost[r, c] = abs(fif_t[f] - h5_t[h])
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] <= tol:
                    pairs.append((fi[r], hi[c]))
        return pairs

    fif_t = {i: fif_events[i][0] for i in range(len(fif_events))}
    h5_t = {j: h5_events[j][0] for j in range(len(h5_events))}

    # Pass 1: coarse match (generous tolerance) to estimate drift
    coarse_tol = max(tolerance_s * 4, 0.2)
    all_fif = list(range(len(fif_events)))
    all_h5 = list(range(len(h5_events)))
    coarse_pairs = _assign(all_fif, all_h5, fif_t, h5_t, coarse_tol)

    drift_poly = None
    h5_t_corrected = dict(h5_t)

    if len(coarse_pairs) >= 10:
        ft = np.array([fif_t[f] for f, _ in coarse_pairs])
        drift = np.array([h5_t[h] - fif_t[f] for f, h in coarse_pairs])
        drift_poly = np.poly1d(np.polyfit(ft, drift, 1))
        # Correct all H5 times
        for j in all_h5:
            h5_t_corrected[j] = h5_events[j][0] - float(drift_poly(h5_events[j][0]))

    # Pass 2: strict match on corrected times
    final_pairs = _assign(all_fif, all_h5, fif_t, h5_t_corrected, tolerance_s)

    matched_fif = {f for f, _ in final_pairs}
    matched_h5 = {h for _, h in final_pairs}

    matched = []
    for f, h in final_pairs:
        residual = abs(fif_t[f] - h5_t_corrected[h])
        matched.append((fif_t[f], h5_events[h][0], residual))

    unmatched_fif = sorted(set(all_fif) - matched_fif)
    unmatched_h5 = sorted(set(all_h5) - matched_h5)
    return matched, unmatched_fif, unmatched_h5, drift_poly


@pytest.mark.integration
class TestEventAlignment:
    """Verify events from FIF match events in H5 pipeline output."""

    def test_all_event_types_present(self):
        """Every unique event type name in FIF appears in H5."""
        fif_events = _load_fif_events(FIF_PATH)
        h5_events = _load_h5_events(H5_PATH)

        fif_names = {name for _, _, name in fif_events}
        h5_names = {name for _, _, name in h5_events}

        missing = fif_names - h5_names
        assert not missing, f"Event types in FIF but not H5: {missing}"

    def test_event_count_match(self):
        """H5 should have same number of events (+-small tolerance for edge effects)."""
        fif_events = _load_fif_events(FIF_PATH)
        h5_events = _load_h5_events(H5_PATH)

        fif_n = len(fif_events)
        h5_n = len(h5_events)
        drop_pct = abs(fif_n - h5_n) / fif_n * 100

        print(f"FIF: {fif_n} events, H5: {h5_n} events, diff: {fif_n - h5_n} ({drop_pct:.1f}%)")
        assert drop_pct < 5, f"Too many events lost: {fif_n} -> {h5_n} ({drop_pct:.1f}% drop)"

    def test_event_timing_alignment(self):
        """Matched events should align after removing linear clock drift.

        Produces a diagnostic plot for each replay pair showing raw drift,
        fitted correction line, and residual jitter after correction.
        """
        for fif_path, h5_path, ses_label in REPLAY_PAIRS:
            if not os.path.exists(fif_path) or not os.path.exists(h5_path):
                print(f"Skipping {ses_label}: files not found")
                continue

            fif_events = _load_fif_events(fif_path)
            h5_events = _load_h5_events(h5_path)

            matched, unmatched_fif, unmatched_h5, drift_poly = _match_events(
                fif_events, h5_events,
            )
            residuals = np.array([r for _, _, r in matched]) if matched else np.array([])

            match_pct = len(matched) / len(fif_events) * 100
            mean_res = float(residuals.mean()) if len(residuals) else 0
            max_res = float(residuals.max()) if len(residuals) else 0

            # Drift data for plotting
            matched_sorted = sorted(matched, key=lambda p: p[0])
            fif_t = np.array([p[0] for p in matched_sorted])
            raw_drift_ms = np.array([(p[1] - p[0]) * 1000 for p in matched_sorted])
            residual_ms = np.array([p[2] * 1000 for p in matched_sorted])

            slope_ppm = drift_poly[1] * 1e6 if drift_poly is not None else 0
            drift_label = (
                f"{drift_poly[1]*1000:+.3f} ms/s ({slope_ppm:+.0f} ppm)"
                if drift_poly is not None else "N/A"
            )

            print(f"\n[{ses_label}]")
            print(f"  Matched: {len(matched)}/{len(fif_events)} ({match_pct:.1f}%)")
            print(f"  Clock drift: {drift_label}")
            print(f"  Residual jitter: mean={mean_res*1000:.1f}ms, max={max_res*1000:.1f}ms")
            if unmatched_fif:
                for i in unmatched_fif[:3]:
                    t, _, name = fif_events[i]
                    print(f"    unmatched FIF: {name} @ {t:.3f}s")

            # --- Diagnostic plot ---
            fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08})

            # Top: raw drift + fit
            ax = axes[0]
            ax.scatter(fif_t / 60, raw_drift_ms, s=4, alpha=0.5, color="#5B8DEF",
                       label="Raw drift", zorder=2)
            if drift_poly is not None:
                fit_t = np.linspace(fif_t[0], fif_t[-1], 200)
                ax.plot(fit_t / 60, drift_poly(fit_t) * 1000, color="#E04040", lw=1.5,
                        label=f"Fit: {drift_label}", zorder=3)
            ax.axhline(0, color="gray", ls="--", lw=0.6, alpha=0.5)
            ax.set_ylabel("Drift (ms)")
            ax.set_title(
                f"Event Alignment: {ses_label}  |  "
                f"{len(matched)}/{len(fif_events)} matched  |  "
                f"residual {mean_res*1000:.1f}ms mean / {max_res*1000:.1f}ms max",
                fontsize=10, fontweight="bold",
            )
            ax.legend(fontsize=8, loc="upper left")
            ax.tick_params(labelsize=8)

            # Bottom: residual after correction
            ax = axes[1]
            ax.scatter(fif_t / 60, residual_ms, s=4, alpha=0.5, color="#3BA55D", zorder=2)
            ax.axhline(TOLERANCE_S * 1000, color="#E04040", ls=":", lw=1,
                       label=f"Tolerance ({TOLERANCE_S*1000:.0f}ms)")
            ax.set_ylabel("Residual (ms)")
            ax.set_xlabel("Time (min)")
            ax.set_ylim(0, max(TOLERANCE_S * 1000 * 1.2, residual_ms.max() * 1.3))
            ax.legend(fontsize=8, loc="upper left")
            ax.tick_params(labelsize=8)

            out_dir = os.path.dirname(h5_path).replace("raw", "")
            out_path = os.path.join(out_dir, f"{ses_label}_event_alignment.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Plot saved to {out_path}")

            assert match_pct > 95, f"[{ses_label}] Only {match_pct:.1f}% matched (need >95%)"
            assert max_res < TOLERANCE_S, (
                f"[{ses_label}] Max residual {max_res*1000:.1f}ms exceeds {TOLERANCE_S*1000}ms"
            )

    def test_relative_event_spacing(self):
        """Inter-event gaps should match FIF within tolerance (matched events only)."""
        fif_events = _load_fif_events(FIF_PATH)
        h5_events = _load_h5_events(H5_PATH)

        matched, _, _, _ = _match_events(fif_events, h5_events)
        if len(matched) < 2:
            pytest.skip("Not enough matched events for gap comparison")

        # Sort matched pairs by FIF time, then compare consecutive gaps
        matched.sort(key=lambda p: p[0])
        fif_times = np.array([p[0] for p in matched])
        h5_times = np.array([p[1] for p in matched])
        gap_errors = np.abs(np.diff(fif_times) - np.diff(h5_times))

        mean_err = float(gap_errors.mean())
        max_err = float(gap_errors.max())

        print(f"Inter-event gap error ({len(matched)-1} gaps): "
              f"mean={mean_err*1000:.1f}ms, max={max_err*1000:.1f}ms")
        assert mean_err < 0.01, f"Mean gap error {mean_err*1000:.1f}ms too high (>10ms)"

    def test_erp_correct_vs_incorrect(self):
        """Compare ERPs from original FIF vs replayed H5 pipeline output.

        Produces two plots:
        - Broadband (1-30 Hz + CAR): shows full spectral content, useful for
          verifying pipeline fidelity (FIF vs H5 correlation).
        - ErrP-optimized (0.5-10 Hz + CAR): matches swarm-study analysis,
          isolates slow ERP components (P300/Pe) without alpha/beta noise.
        """
        from dendrite.data.loaders.fif_loader import FIFLoader
        from dendrite.data.loaders.raw_h5_loader import RawH5Loader

        TARGET_TYPES = {"patterns_shape_correct", "patterns_shape_incorrect"}
        LABEL_MAP = {"patterns_shape_correct": 0, "patterns_shape_incorrect": 1}
        PLOT_CHANNELS = ["FCz", "Cz", "Pz"]

        # Each entry: (preproc_config_or_None, label, suffix, use_mne_filter)
        PREPROC_VARIANTS = [
            ({"lowcut": 1.0, "highcut": 30, "apply_rereferencing": True}, "1-30 Hz IIR + CAR", "broadband", False),
            ({"lowcut": 0.5, "highcut": 10, "apply_rereferencing": True}, "0.5-10 Hz IIR + CAR", "errp_iir", False),
            ({"l_freq": 0.5, "h_freq": 10.0}, "0.5-10 Hz FIR + CAR (MNE)", "errp_fir", True),
        ]

        def _prepare(loaded, preproc, use_mne_filter=False):
            """Drop non-EEG, align to first event, preprocess, epoch, baseline-correct."""
            eeg_mask = [t == "eeg" for t in loaded.channel_types]
            loaded.data = loaded.data[eeg_mask]
            loaded.channel_names = [n for n, m in zip(loaded.channel_names, eeg_mask) if m]
            loaded.channel_types = [t for t in loaded.channel_types if t == "eeg"]

            first_event_idx = min(idx for idx, _ in loaded.events)
            trim = max(0, first_event_idx - int(5.0 * loaded.sample_rate))
            if trim > 0:
                loaded.data = loaded.data[:, trim:]
                loaded.events = [(idx - trim, code) for idx, code in loaded.events]

            if use_mne_filter:
                # MNE FIR (zero-phase, linear-phase) — preserves ERP morphology
                raw_mne = loaded.to_mne_raw()
                raw_mne.filter(preproc.get("l_freq"), preproc.get("h_freq"),
                               picks="eeg", verbose=False)
                raw_mne.set_eeg_reference("average", verbose=False)
                loaded.data = raw_mne.get_data()
                loaded.sample_rate = raw_mne.info["sfreq"]
            else:
                # Dendrite IIR (causal, online-equivalent)
                loaded.preprocess(preproc)

            id_to_name = {code: name for name, code in loaded.event_id.items()}
            event_mapping = {c: n for c, n in id_to_name.items() if n in TARGET_TYPES}
            assert len(event_mapping) == 2, f"Expected 2 target events, got {event_mapping}"

            epoched = loaded.epoch({
                "epoch_tmin": -0.2, "epoch_tmax": 0.8,
                "event_mapping": event_mapping, "label_mapping": LABEL_MAP,
                "use_epoch_qc": True,
            })
            X, y = epoched.X, epoched.y

            bl = int(0.2 * loaded.sample_rate)
            for i in range(len(X)):
                X[i] -= X[i, :, :bl].mean(axis=1, keepdims=True)

            return X, y, loaded.channel_names, loaded.sample_rate

        def _compute_erps(X, y, ch_names):
            n_c, n_i = int((y == 0).sum()), int((y == 1).sum())
            Xc, Xi = X[y == 0], X[y == 1]
            return {
                "erp_c": Xc.mean(0), "erp_i": Xi.mean(0),
                "sem_c": Xc.std(0) / np.sqrt(n_c),
                "sem_i": Xi.std(0) / np.sqrt(n_i),
                "diff": Xc.mean(0) - Xi.mean(0),
                "n_c": n_c, "n_i": n_i, "ch": ch_names,
            }

        def _plot_erp_figure(erps, times, sr, filter_label, out_path,
                             fif_path=FIF_PATH, h5_path=H5_PATH, title=None):
            n_ch = len(PLOT_CHANNELS)
            fig = plt.figure(figsize=(5 * n_ch + 4, 11))
            gs = fig.add_gridspec(3, n_ch + 1, width_ratios=[1] * n_ch + [0.8],
                                  wspace=0.3, hspace=0.4)

            for row, (src, row_label) in enumerate([("FIF", "Original (FIF)"),
                                                     ("H5", "Pipeline (H5)")]):
                for col, ch_name in enumerate(PLOT_CHANNELS):
                    ax = fig.add_subplot(gs[row, col])
                    e = erps[src]
                    ci = e["ch"].index(ch_name)
                    ax.plot(times, e["erp_c"][ci], color="steelblue", lw=1.2, label="correct")
                    ax.fill_between(times, e["erp_c"][ci] - e["sem_c"][ci],
                                    e["erp_c"][ci] + e["sem_c"][ci], color="steelblue", alpha=0.15)
                    ax.plot(times, e["erp_i"][ci], color="indianred", lw=1.2, label="incorrect")
                    ax.fill_between(times, e["erp_i"][ci] - e["sem_i"][ci],
                                    e["erp_i"][ci] + e["sem_i"][ci], color="indianred", alpha=0.15)
                    ax.axvline(0, color="gray", ls="--", alpha=0.5, lw=0.8)
                    ax.axhline(0, color="gray", ls="-", alpha=0.2, lw=0.8)
                    ax.set_title(f"{ch_name} -- {row_label}", fontsize=10, fontweight="bold")
                    ax.set_xlabel("Time (ms)", fontsize=8)
                    ax.set_ylabel("uV", fontsize=8)
                    ax.tick_params(labelsize=7)
                    if col == 0:
                        ax.legend(fontsize=7, loc="upper right")

            for col, ch_name in enumerate(PLOT_CHANNELS):
                ax = fig.add_subplot(gs[2, col])
                ci_f = erps["FIF"]["ch"].index(ch_name)
                ci_h = erps["H5"]["ch"].index(ch_name)
                ax.plot(times, erps["FIF"]["diff"][ci_f], color="#1E5AA8", lw=1.5,
                        label="FIF (corr-incorr)")
                ax.plot(times, erps["H5"]["diff"][ci_h], color="#228B22", lw=1.5, ls="--",
                        label="H5 (corr-incorr)")
                ax.axvline(0, color="gray", ls="--", alpha=0.5, lw=0.8)
                ax.axhline(0, color="gray", ls="-", alpha=0.2, lw=0.8)
                ax.set_title(f"{ch_name} -- Difference wave", fontsize=10, fontweight="bold")
                ax.set_xlabel("Time (ms)", fontsize=8)
                ax.set_ylabel("d uV", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.legend(fontsize=7, loc="upper right")

            # Stats panel
            ax_s = fig.add_subplot(gs[:, n_ch])
            ax_s.axis("off")
            pk_s = int((200 + 200) / 1000 * sr)
            pk_e = int((600 + 200) / 1000 * sr)
            p3s = int((250 + 200) / 1000 * sr)
            p3e = int((500 + 200) / 1000 * sr)

            lines = [
                "PREPROCESSING",
                f"  Filter: {filter_label}",
                f"  Epoch:  -200 to 800 ms",
                f"  Baseline: -200 to 0 ms",
                "",
                "EPOCHS",
                f"  {'':6} {'Corr':>5} {'Incorr':>6} {'Total':>5}",
                f"  {'FIF':6} {erps['FIF']['n_c']:>5} {erps['FIF']['n_i']:>6}"
                f" {erps['FIF']['n_c']+erps['FIF']['n_i']:>5}",
                f"  {'H5':6} {erps['H5']['n_c']:>5} {erps['H5']['n_i']:>6}"
                f" {erps['H5']['n_c']+erps['H5']['n_i']:>5}",
                "",
                "DIFF PEAK (200-600 ms)",
                f"  {'Ch':<4} {'FIF':>7} {'H5':>7} {'d':>7}",
            ]
            for ch_name in PLOT_CHANNELS:
                ci_f = erps["FIF"]["ch"].index(ch_name)
                ci_h = erps["H5"]["ch"].index(ch_name)
                f_w = erps["FIF"]["diff"][ci_f, pk_s:pk_e]
                h_w = erps["H5"]["diff"][ci_h, pk_s:pk_e]
                f_pk = f_w[np.argmax(np.abs(f_w))]
                h_pk = h_w[np.argmax(np.abs(h_w))]
                lines.append(f"  {ch_name:<4} {f_pk:>+6.2f} {h_pk:>+6.2f} {f_pk-h_pk:>+6.2f}")

            lines += ["", "DIFF WAVE SIMILARITY"]
            for ch_name in PLOT_CHANNELS:
                ci_f = erps["FIF"]["ch"].index(ch_name)
                ci_h = erps["H5"]["ch"].index(ch_name)
                d_fif = erps["FIF"]["diff"][ci_f]
                d_h5 = erps["H5"]["diff"][ci_h]
                rmse = np.sqrt(np.mean((d_fif - d_h5) ** 2))
                corr = np.corrcoef(d_fif, d_h5)[0, 1]
                lines.append(f"  {ch_name:<4} r={corr:.3f}  RMSE={rmse:.3f}")

            lines += ["", "MEAN AMP (250-500 ms)"]
            lines.append(f"  {'Ch':<4} {'FIFd':>7} {'H5d':>7}")
            for ch_name in PLOT_CHANNELS:
                ci_f = erps["FIF"]["ch"].index(ch_name)
                ci_h = erps["H5"]["ch"].index(ch_name)
                f_m = erps["FIF"]["diff"][ci_f, p3s:p3e].mean()
                h_m = erps["H5"]["diff"][ci_h, p3s:p3e].mean()
                lines.append(f"  {ch_name:<4} {f_m:>+6.2f} {h_m:>+6.2f}")

            lines += [
                "",
                "FILES",
                f"  FIF: ...{os.path.basename(fif_path)[-30:]}",
                f"  H5:  ...{os.path.basename(h5_path)[-30:]}",
            ]
            ax_s.text(0.02, 0.97, "\n".join(lines), transform=ax_s.transAxes,
                      fontsize=8, fontfamily="monospace", verticalalignment="top")

            fig.suptitle(title or f"ERP: FIF vs H5 ({filter_label})",
                         fontsize=13, fontweight="bold", y=0.99)
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"ERP plot saved to {out_path}")

        # ── Run all filter variants x replay pairs ──
        for fif_path, h5_path, ses_label in REPLAY_PAIRS:
            if not os.path.exists(fif_path) or not os.path.exists(h5_path):
                print(f"Skipping {ses_label}: files not found")
                continue

            out_dir = os.path.dirname(h5_path).replace("raw", "")

            for preproc, filter_label, suffix, use_mne in PREPROC_VARIANTS:
                fif_loaded = FIFLoader(fif_path).load()
                h5_loaded = RawH5Loader(h5_path).load()
                assert fif_loaded.event_id is not None
                assert h5_loaded.event_id is not None

                X_fif, y_fif, ch_fif, sr_fif = _prepare(fif_loaded, preproc, use_mne)
                X_h5, y_h5, ch_h5, sr_h5 = _prepare(h5_loaded, preproc, use_mne)

                print(f"\n[{ses_label} / {filter_label}]")
                print(f"FIF epochs: {(y_fif==0).sum()} correct, {(y_fif==1).sum()} incorrect")
                print(f"H5  epochs: {(y_h5==0).sum()} correct, {(y_h5==1).sum()} incorrect")
                assert (y_fif == 0).sum() >= 5 and (y_fif == 1).sum() >= 5
                assert (y_h5 == 0).sum() >= 5 and (y_h5 == 1).sum() >= 5

                times = np.linspace(-200, 800, X_fif.shape[2])
                erps = {
                    "FIF": _compute_erps(X_fif, y_fif, ch_fif),
                    "H5": _compute_erps(X_h5, y_h5, ch_h5),
                }

                title = f"ERP: FIF vs H5 -- {ses_label} ({filter_label})"
                out_path = os.path.join(out_dir, f"{ses_label}_erp_{suffix}.png")
                _plot_erp_figure(erps, times, sr_h5, filter_label, out_path,
                                 fif_path=fif_path, h5_path=h5_path, title=title)
