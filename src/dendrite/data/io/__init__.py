"""
Data I/O - H5 I/O, MNE Export, and BIDS Conversion

Clean, focused modules for working with H5 recordings.

Examples:
    # Load data from H5
    from dendrite.data.io import load_dataset, get_h5_info
    df = load_dataset('recording.h5', 'EEG')
    info = get_h5_info('recording.h5')

    # Export to MNE/FIF
    from dendrite.data.io import to_mne_raw, export_to_fif
    raw = to_mne_raw('recording.h5')
    export_to_fif('recording.h5', 'output.fif')

    # Export to BIDS
    from dendrite.data.io import export_recording_to_bids, export_study_to_bids
    export_recording_to_bids('recording.h5', 'bids_output/')
    export_study_to_bids('my_study', 'bids_output/')
"""

from .bids_export import (
    export_recording_to_bids,
    export_study_to_bids,
)
from .h5_io import (
    get_channel_info,
    get_h5_info,
    get_h5_metadata,
    load_dataset,
    load_events,
)
from .mne_export import (
    export_to_fif,
    guess_channel_type,
    to_mne_raw,
)

__all__ = [
    "load_dataset",
    "get_h5_info",
    "get_h5_metadata",
    "get_channel_info",
    "load_events",
    "to_mne_raw",
    "export_to_fif",
    "guess_channel_type",
    "export_recording_to_bids",
    "export_study_to_bids",
]
