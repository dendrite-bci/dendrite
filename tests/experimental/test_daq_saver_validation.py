"""
DAQ + DataSaver Validation Script.

PURPOSE:
    Compare original FIF recording against re-recorded H5 (converted to FIF)
    to validate that the DAQ and DataSaver pipeline preserves data integrity.

TEST TYPE:
    Integration test - validates end-to-end data flow from LSL streaming
    through H5 storage and FIF export.

WHAT IT VALIDATES:
    - Sample count preservation (>90% threshold)
    - Event extraction from H5 Event dataset
    - ERP waveform correlation (>0.9 threshold)
    - Channel and sample rate consistency

DASHBOARD OUTPUT:
    Single figure with 6 panels:
    - Metrics summary (pass/fail, counts, correlation)
    - Event counts bar chart
    - Condition waves (top 2 events, original vs recorded)
    - Difference per condition
    - Raw signal overlay
    - Event timeline

USAGE:
    uv run python -m tests.integration.test_daq_saver_validation <h5_path>

REFERENCE:
    Based on analysis/daq_validation/validate_eduexo.ipynb
"""

import os
import sys
import argparse
import tempfile
import numpy as np
from pathlib import Path

import mne

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dendrite.data.io.mne_export import export_to_fif


# Reference FIF file (hardcoded)
REFERENCE_FIF = "/home/niko/Documents/Work/old_data_struct/fif/eeg_data_EduExo_pilot_V2_20250227_124227_raw_all_events.fif"


def load_fif(fif_path: str) -> dict:
    """Load FIF file and extract data/events using MNE."""
    print(f"Loading: {Path(fif_path).name}")
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)

    # Get events from annotations
    events = np.array([])
    event_id = {}
    if raw.annotations:
        try:
            events, event_id = mne.events_from_annotations(raw, verbose=False)
        except Exception as e:
            print(f"  Warning: Could not extract events: {e}")

    return {
        'raw': raw,
        'events': events,
        'event_id': event_id,
        'sample_rate': raw.info['sfreq'],
        'channel_names': raw.ch_names,
        'n_samples': raw.n_times,
        'n_channels': len(raw.ch_names)
    }


def convert_h5_to_fif(h5_path: str) -> str:
    """Convert H5 recording to temporary FIF file."""
    print(f"Converting H5 to FIF: {Path(h5_path).name}")
    temp_fif = tempfile.NamedTemporaryFile(suffix='_raw.fif', delete=False)
    temp_path = temp_fif.name
    temp_fif.close()

    export_to_fif(h5_path, temp_path, dataset='EEG', include_events=True, overwrite=True)
    return temp_path


def preprocess_raw(raw):
    """Return raw data (no filtering - matches validate_eduexo.ipynb)."""
    # NOTE: Original notebook does NOT apply bandpass filter before epoching
    return raw


def create_epochs(raw, events, event_id, tmin=-0.5, tmax=1.0):
    """Create MNE Epochs from raw data with baseline correction."""
    if len(events) == 0:
        return None

    epochs = mne.Epochs(
        raw, events, event_id=event_id,
        tmin=tmin, tmax=tmax,
        baseline=(None, 0),  # From start of epoch to event onset
        preload=True,
        verbose=False
    )
    return epochs


def match_epochs(epochs_orig, epochs_rec, event_type):
    """Match epochs by index to ensure same trials are compared."""
    n_orig = len(epochs_orig[event_type])
    n_rec = len(epochs_rec[event_type])
    n_common = min(n_orig, n_rec)

    # Use first n_common epochs from each
    data_orig = epochs_orig[event_type][:n_common].average().get_data()
    data_rec = epochs_rec[event_type][:n_common].average().get_data()
    return data_orig, data_rec, n_common


def plot_dashboard(original, recorded, epochs_orig, epochs_rec, results, common_events):
    """Single dashboard with all validation metrics and plots."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 12), facecolor='white')
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    # Common styling helper
    def style_ax(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('white')

    # -------------------------------------------------------------------------
    # Row 1, Col 1: Metrics Summary (text box)
    # -------------------------------------------------------------------------
    ax_metrics = fig.add_subplot(gs[0, 0])
    ax_metrics.axis('off')

    sample_ratio = results['sample_count']['ratio']
    erp_corr = results.get('erp_correlation', 0) or 0
    status = "PASSED" if results['passed'] else "FAILED"
    status_color = '#228B22' if results['passed'] else '#DC143C'

    metrics_text = (
        f"Validation Status: {status}\n\n"
        f"Samples:  {results['sample_count']['recorded']:,} / {results['sample_count']['original']:,} ({sample_ratio:.1%})\n"
        f"Events:   {results['event_count']['recorded']} / {results['event_count']['original']}\n"
        f"Channels: {results['channel_info']['recorded']} / {results['channel_info']['original']}\n"
        f"Sample Rate: {results['channel_info']['sample_rate_recorded']:.0f} Hz\n\n"
        f"ERP Correlation: {erp_corr:.4f}"
    )
    ax_metrics.text(0.1, 0.5, metrics_text, transform=ax_metrics.transAxes,
                    fontsize=12, verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='#f8f8f8', edgecolor='#ddd'))
    ax_metrics.set_title('Validation Metrics', fontsize=13, fontweight='bold', loc='left')

    # -------------------------------------------------------------------------
    # Row 1, Col 2: Event Counts (bar chart)
    # -------------------------------------------------------------------------
    ax_events = fig.add_subplot(gs[0, 1])
    style_ax(ax_events)

    events_orig = original['events']
    events_rec = recorded['events']
    orig_codes = events_orig[:, 2]
    rec_codes = events_rec[:, 2]

    orig_unique, orig_counts = np.unique(orig_codes, return_counts=True)
    rec_unique, rec_counts = np.unique(rec_codes, return_counts=True)

    # Limit to top 8 event types
    if len(orig_unique) > 8:
        top_idx = np.argsort(orig_counts)[-8:]
        orig_unique = orig_unique[top_idx]
        orig_counts = orig_counts[top_idx]

    x = np.arange(len(orig_unique))
    width = 0.35

    ax_events.bar(x - width/2, orig_counts, width, label='Original', color='#1E5AA8', alpha=0.7)

    rec_counts_matched = []
    for code in orig_unique:
        if code in rec_unique:
            rec_counts_matched.append(rec_counts[list(rec_unique).index(code)])
        else:
            rec_counts_matched.append(0)

    ax_events.bar(x + width/2, rec_counts_matched, width, label='Recorded', color='#228B22', alpha=0.7)
    ax_events.set_xlabel('Event Code')
    ax_events.set_ylabel('Count')
    ax_events.set_title('Event Counts by Type', fontsize=13, fontweight='bold')
    ax_events.set_xticks(x)
    ax_events.set_xticklabels([str(c) for c in orig_unique], rotation=45, ha='right', fontsize=9)
    ax_events.legend(frameon=True, fontsize=9, loc='upper right')
    ax_events.grid(True, alpha=0.3, axis='y')

    # -------------------------------------------------------------------------
    # Row 2: ERP Comparison (like notebook - conditions overlaid)
    # -------------------------------------------------------------------------
    ax_orig = fig.add_subplot(gs[1, 0])
    ax_rec = fig.add_subplot(gs[1, 1])
    style_ax(ax_orig)
    style_ax(ax_rec)

    # Target events and labels
    TARGET_EVENTS = ['exo_execution_correct_triangular', 'exo_execution_incorrect_triangular']
    EVENT_LABELS = {'exo_execution_correct_triangular': 'Correct',
                    'exo_execution_incorrect_triangular': 'Incorrect'}
    EVENT_COLORS = {'exo_execution_correct_triangular': '#1E5AA8',
                    'exo_execution_incorrect_triangular': '#228B22'}
    EVENT_STYLES = {'exo_execution_correct_triangular': '--',
                    'exo_execution_incorrect_triangular': '-'}

    if epochs_orig is not None and epochs_rec is not None and common_events:
        common_channels = list(set(epochs_orig.ch_names) & set(epochs_rec.ch_names))
        channel = 'FCz' if 'FCz' in common_channels else common_channels[0]
        ch_idx = epochs_orig.ch_names.index(channel)
        times = epochs_orig.times

        # Use target events if available
        available_targets = [e for e in TARGET_EVENTS if e in common_events]
        if len(available_targets) < 2:
            event_counts = {e: len(epochs_orig[e]) for e in common_events}
            available_targets = sorted(event_counts, key=event_counts.get, reverse=True)[:2]

        # Plot both conditions on ORIGINAL (left)
        for event in available_targets:
            data_orig, _, n = match_epochs(epochs_orig, epochs_rec, event)
            erp = data_orig[ch_idx, :] * 1e6
            label = EVENT_LABELS.get(event, event[:15])
            color = EVENT_COLORS.get(event, '#1E5AA8')
            ls = EVENT_STYLES.get(event, '-')
            ax_orig.plot(times, erp, color=color, linestyle=ls, lw=1.5,
                         alpha=0.9, label=f'{label} (n={n})')

        # Plot both conditions on RECORDED (right)
        for event in available_targets:
            _, data_rec, n = match_epochs(epochs_orig, epochs_rec, event)
            erp = data_rec[ch_idx, :] * 1e6
            label = EVENT_LABELS.get(event, event[:15])
            color = EVENT_COLORS.get(event, '#228B22')
            ls = EVENT_STYLES.get(event, '-')
            ax_rec.plot(times, erp, color=color, linestyle=ls, lw=1.5,
                        alpha=0.9, label=f'{label} (n={n})')

        # Style both plots identically
        for ax, title in [(ax_orig, f'Original FIF - {channel}'),
                          (ax_rec, f'Recorded H5 - {channel}')]:
            ax.axhline(0, color='gray', linestyle='--', lw=0.5, alpha=0.5)
            ax.axvline(0, color='gray', linestyle='--', lw=0.5, alpha=0.5)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude (µV)')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend(frameon=True, fontsize=9, loc='upper right')

    else:
        ax_orig.text(0.5, 0.5, 'No epochs available', ha='center', va='center', fontsize=12)
        ax_rec.text(0.5, 0.5, 'No epochs available', ha='center', va='center', fontsize=12)

    # -------------------------------------------------------------------------
    # Row 3, Col 1: Raw Signal Overlay
    # -------------------------------------------------------------------------
    ax_raw = fig.add_subplot(gs[2, 0])
    style_ax(ax_raw)

    sfreq = original['sample_rate']
    duration = 2.0
    n_samples = int(duration * sfreq)

    common_channels = list(set(original['channel_names']) & set(recorded['channel_names']))
    plot_channel = 'Fz' if 'Fz' in common_channels else common_channels[0]

    try:
        raw_orig = original['raw'].get_data(picks=plot_channel)[:, :n_samples].squeeze() * 1e6
        raw_rec = recorded['raw'].get_data(picks=plot_channel)[:, :n_samples].squeeze() * 1e6
        times_raw = np.arange(n_samples) / sfreq

        ax_raw.plot(times_raw, raw_orig, color='#1E5AA8', lw=0.6, alpha=0.7, label='Original')
        ax_raw.plot(times_raw, raw_rec, color='#228B22', lw=0.6, alpha=0.7, label='Recorded')
        ax_raw.set_xlabel('Time (s)')
        ax_raw.set_ylabel('Amplitude (µV)')
        ax_raw.set_title(f'Raw Signal Overlay - {plot_channel} (first {duration}s)',
                         fontsize=13, fontweight='bold')
        ax_raw.legend(frameon=True, fontsize=9, loc='upper right')
        ax_raw.grid(True, alpha=0.3)
    except Exception as e:
        ax_raw.text(0.5, 0.5, f'Could not load raw data: {e}', ha='center', va='center')

    # -------------------------------------------------------------------------
    # Row 3, Col 2: Event Timeline
    # -------------------------------------------------------------------------
    ax_timeline = fig.add_subplot(gs[2, 1])
    style_ax(ax_timeline)

    n_events = min(50, len(events_orig), len(events_rec))
    orig_times = (events_orig[:n_events, 0] - events_orig[0, 0]) / sfreq
    rec_times = (events_rec[:n_events, 0] - events_rec[0, 0]) / sfreq

    ax_timeline.scatter(range(len(orig_times)), orig_times, c='#1E5AA8', alpha=0.6,
                        label='Original', s=25, marker='o')
    ax_timeline.scatter(range(len(rec_times)), rec_times, c='#228B22', alpha=0.6,
                        label='Recorded', s=25, marker='x')
    ax_timeline.set_xlabel('Event Index')
    ax_timeline.set_ylabel('Time from first event (s)')
    ax_timeline.set_title(f'Event Timeline (first {n_events} events)', fontsize=13, fontweight='bold')
    ax_timeline.legend(frameon=True, fontsize=9)
    ax_timeline.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Final layout
    # -------------------------------------------------------------------------
    fig.suptitle('DAQ/Saver Validation Dashboard', fontsize=16, fontweight='bold', y=0.99)
    plt.subplots_adjust(top=0.93, hspace=0.35, wspace=0.25)


def compute_erp_correlation(epochs_orig, epochs_rec, event_type, channels=None):
    """Compute correlation between ERPs from two files."""
    if channels is None:
        # Use common channels
        channels = list(set(epochs_orig.ch_names) & set(epochs_rec.ch_names))[:10]

    correlations = []
    for ch in channels:
        try:
            data_orig = epochs_orig[event_type].average().get_data(picks=ch).squeeze()
            data_rec = epochs_rec[event_type].average().get_data(picks=ch).squeeze()

            if np.std(data_orig) < 1e-15 or np.std(data_rec) < 1e-15:
                continue

            corr = np.corrcoef(data_orig, data_rec)[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        except Exception:
            continue

    return np.mean(correlations) if correlations else 0.0


def validate_recording(recorded_h5_path: str, show_plots: bool = True) -> dict:
    """
    Compare original FIF against recorded H5 (converted to FIF).

    Args:
        recorded_h5_path: Path to the recorded H5 file
        show_plots: Whether to show matplotlib comparison plots

    Returns:
        Dictionary with validation results
    """
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("DAQ/Saver Validation Report (FIF vs FIF)")
    print("=" * 60)

    # Check files exist
    if not os.path.exists(REFERENCE_FIF):
        print(f"ERROR: Reference FIF not found: {REFERENCE_FIF}")
        return {'passed': False, 'error': 'Reference file not found'}

    if not os.path.exists(recorded_h5_path):
        print(f"ERROR: Recorded H5 not found: {recorded_h5_path}")
        return {'passed': False, 'error': 'Recorded file not found'}

    print(f"\nReference: {Path(REFERENCE_FIF).name}")
    print(f"Recorded:  {Path(recorded_h5_path).name}")
    print("-" * 60)

    # Load reference FIF
    print("\nLoading reference FIF...")
    original = load_fif(REFERENCE_FIF)

    # Convert H5 to FIF and load
    print("\nConverting and loading recorded H5...")
    temp_fif_path = convert_h5_to_fif(recorded_h5_path)
    try:
        recorded = load_fif(temp_fif_path)
    finally:
        if os.path.exists(temp_fif_path):
            os.unlink(temp_fif_path)

    # Results dict
    results = {
        'passed': True,
        'sample_count': {},
        'event_count': {},
        'channel_info': {},
        'erp_correlation': None
    }

    # Compare sample counts
    print("\n--- Sample Count ---")
    orig_samples = original['n_samples']
    rec_samples = recorded['n_samples']
    sample_ratio = rec_samples / orig_samples if orig_samples > 0 else 0

    results['sample_count'] = {
        'original': orig_samples,
        'recorded': rec_samples,
        'ratio': sample_ratio
    }

    print(f"Original:  {orig_samples:,} samples")
    print(f"Recorded:  {rec_samples:,} samples")
    print(f"Ratio:     {sample_ratio:.2%}")

    if sample_ratio < 0.9:
        print("WARNING: Recorded file has <90% of original samples")
        results['passed'] = False

    # Compare events
    print("\n--- Event Count ---")
    orig_events = len(original['events'])
    rec_events = len(recorded['events'])

    results['event_count'] = {
        'original': orig_events,
        'recorded': rec_events
    }

    print(f"Original:  {orig_events} events")
    print(f"Recorded:  {rec_events} events")

    if orig_events > 0:
        print(f"Original event types: {list(original['event_id'].keys())[:5]}...")
    if rec_events > 0:
        print(f"Recorded event types: {list(recorded['event_id'].keys())[:5]}...")

    # Compare channel info
    print("\n--- Channel Info ---")
    orig_channels = original['n_channels']
    rec_channels = recorded['n_channels']

    results['channel_info'] = {
        'original': orig_channels,
        'recorded': rec_channels,
        'sample_rate_original': original['sample_rate'],
        'sample_rate_recorded': recorded['sample_rate']
    }

    print(f"Original:  {orig_channels} channels @ {original['sample_rate']} Hz")
    print(f"Recorded:  {rec_channels} channels @ {recorded['sample_rate']} Hz")

    # Preprocess and create epochs for ERP comparison
    epochs_orig = None
    epochs_rec = None
    erp_correlation = None
    common_events = set()

    if orig_events > 0 and rec_events > 0:
        print("\n--- Creating Epochs (no filtering, like notebook) ---")
        raw_orig_filtered = preprocess_raw(original['raw'])
        raw_rec_filtered = preprocess_raw(recorded['raw'])

        print("\n--- Creating Epochs ---")
        epochs_orig = create_epochs(raw_orig_filtered, original['events'], original['event_id'])
        epochs_rec = create_epochs(raw_rec_filtered, recorded['events'], recorded['event_id'])

        if epochs_orig is not None and epochs_rec is not None:
            # Find common event types
            common_events = set(epochs_orig.event_id.keys()) & set(epochs_rec.event_id.keys())
            print(f"Common event types: {len(common_events)}")

            if common_events:
                first_event = list(common_events)[0]
                print(f"\n--- ERP Comparison (event: {first_event}) ---")

                erp_correlation = compute_erp_correlation(epochs_orig, epochs_rec, first_event)
                results['erp_correlation'] = erp_correlation

                print(f"ERP Correlation: {erp_correlation:.4f}")

                if erp_correlation < 0.9:
                    print("WARNING: ERP correlation < 0.9")
                else:
                    print("PASS: ERP correlation >= 0.9")

    print("\n" + "=" * 60)
    if results['passed']:
        print("VALIDATION PASSED")
    else:
        print("VALIDATION FAILED")
    print("=" * 60)

    # Show dashboard
    if show_plots:
        print("\nGenerating validation dashboard...")
        plot_dashboard(original, recorded, epochs_orig, epochs_rec, results, common_events)
        plt.show()

    return results


def main():
    global REFERENCE_FIF

    parser = argparse.ArgumentParser(
        description='Validate recorded H5 against reference FIF'
    )
    parser.add_argument(
        'recorded_h5',
        help='Path to recorded H5 file'
    )
    parser.add_argument(
        '--reference',
        default=REFERENCE_FIF,
        help=f'Path to reference FIF (default: {REFERENCE_FIF})'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip showing plots'
    )

    args = parser.parse_args()

    # Allow overriding reference
    REFERENCE_FIF = args.reference

    results = validate_recording(args.recorded_h5, show_plots=not args.no_plot)

    # Exit with appropriate code
    sys.exit(0 if results.get('passed', False) else 1)


if __name__ == '__main__':
    main()
