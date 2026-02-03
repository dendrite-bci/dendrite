"""Modality name normalization utilities.

Centralizes modality string handling (EEG, EMG, etc.) to ensure consistency
across config schemas, data processing, and GUI components.

Conventions:
- Internal/config: lowercase ('eeg', 'emg')
- Display/user-facing: uppercase ('EEG', 'EMG')
"""


def normalize_modality(name: str) -> str:
    """Normalize modality name to lowercase for internal use.

    Args:
        name: Modality name (e.g., 'EEG', 'eeg', 'Eeg')

    Returns:
        Lowercase modality name (e.g., 'eeg'), or empty string if None/empty
    """
    return name.lower() if name else ""


def normalize_modality_dict(d: dict) -> dict:
    """Normalize all modality keys in a dict to lowercase.

    Args:
        d: Dict with modality keys (e.g., {'EEG': [...], 'EMG': [...]})

    Returns:
        Dict with lowercase keys (e.g., {'eeg': [...], 'emg': [...]})
    """
    if not d:
        return {}
    return {normalize_modality(k): v for k, v in d.items()}


def display_modality(name: str) -> str:
    """Format modality name for display (uppercase).

    Args:
        name: Modality name (e.g., 'eeg', 'EEG')

    Returns:
        Uppercase modality name (e.g., 'EEG'), or empty string if None/empty
    """
    return name.upper() if name else ""
