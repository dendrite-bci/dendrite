"""
BIDS-compliant export from H5 recordings.

Export recordings to Brain Imaging Data Structure (BIDS) format with:
- FIF files for MNE compatibility
- Sidecar JSON metadata
- Channel and event TSV files
- Dataset description and participants files
"""

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

from dendrite.constants import BIDS_VERSION

from .h5_io import get_channel_info, get_h5_metadata, load_dataset, load_events
from .mne_export import export_to_fif, guess_channel_type

logger = logging.getLogger(__name__)

# Common IO errors for H5/metadata operations
_IO_ERRORS = (OSError, KeyError, ValueError, IndexError)


def export_recording_to_bids(
    h5_path: str | Path,
    output_dir: str | Path,
    study_name: str | None = None,
    include_sourcedata: bool = True,
) -> Path:
    """
    Export a single H5 recording to BIDS format.

    Creates:
        - study_name/dataset_description.json
        - study_name/participants.tsv
        - study_name/sourcedata/sub-XX/ses-XX/*.h5 (original file copy)
        - study_name/sub-XX/ses-XX/eeg/*_eeg.fif
        - study_name/sub-XX/ses-XX/eeg/*_eeg.json
        - study_name/sub-XX/ses-XX/eeg/*_channels.tsv
        - study_name/sub-XX/ses-XX/eeg/*_events.tsv (if events exist)

    Args:
        h5_path: Path to source H5 file
        output_dir: BIDS output directory (study folder created inside)
        study_name: Study name for BIDS root folder (auto-detected from H5 if None)
        include_sourcedata: Whether to copy original H5 to sourcedata/

    Returns:
        Path to exported FIF file.
    """
    h5_path = Path(h5_path)
    output_dir = Path(output_dir)

    # Extract BIDS fields from H5 metadata or filename
    bids_info = _extract_bids_info(h5_path)
    sub = bids_info["subject"]
    ses = bids_info["session"]
    task = bids_info["task"]
    run = bids_info["run"]
    # Use provided study_name or fall back to extracted/default
    if study_name is None:
        study_name = bids_info["study_name"]

    # Build BIDS filename base
    bids_base = f"sub-{sub}_ses-{ses}_task-{task}_run-{run}"

    # Study root directory - check if output_dir is already a BIDS dataset
    if (output_dir / "dataset_description.json").exists():
        # output_dir is already a BIDS dataset, use it directly
        study_dir = output_dir
        logger.info(f"Using existing BIDS dataset: {study_dir}")
    else:
        # Create study subfolder
        study_dir = output_dir / study_name

    # Create directory structure
    bids_eeg_dir = study_dir / f"sub-{sub}" / f"ses-{ses}" / "eeg"
    bids_eeg_dir.mkdir(parents=True, exist_ok=True)

    # Copy to sourcedata if requested
    if include_sourcedata:
        sourcedata_dir = study_dir / "sourcedata" / f"sub-{sub}" / f"ses-{ses}"
        _copy_sourcedata_files(h5_path, sourcedata_dir, study_name)

    # Export FIF
    fif_path = bids_eeg_dir / f"{bids_base}_eeg.fif"
    sfreq = bids_info.get("sample_rate")
    if sfreq is None:
        raise ValueError(f"sample_rate not found in H5 metadata: {h5_path}")
    export_to_fif(str(h5_path), sfreq, str(fif_path), overwrite=True)

    # Generate sidecar JSON
    sidecar = generate_sidecar_json(h5_path, bids_info)
    json_path = bids_eeg_dir / f"{bids_base}_eeg.json"
    _write_json(json_path, sidecar)

    # Generate channels.tsv
    channels_tsv = generate_channels_tsv(h5_path)
    tsv_path = bids_eeg_dir / f"{bids_base}_channels.tsv"
    _write_tsv(tsv_path, channels_tsv)

    # Generate events.tsv if events exist
    try:
        events_tsv = generate_events_tsv(h5_path)
        if events_tsv:
            events_path = bids_eeg_dir / f"{bids_base}_events.tsv"
            _write_tsv(events_path, events_tsv)
    except _IO_ERRORS as e:
        logger.debug(f"No events to export: {e}")

    # Generate dataset_description.json (create or update)
    desc_path = study_dir / "dataset_description.json"
    if not desc_path.exists():
        description = generate_dataset_description(study_name)
        _write_json(desc_path, description)

    # Generate participants.tsv (create or update)
    participants_path = study_dir / "participants.tsv"
    _update_participants_tsv(participants_path, sub)

    # Add dataset entry to database
    try:
        _add_dataset_entry(
            fif_path=fif_path,
            study_name=study_name,
            bids_base=bids_base,
            sfreq=sfreq,
            description=f"BIDS export from {h5_path.name}",
        )
    except (OSError, ValueError, RuntimeError) as e:
        logger.warning(f"Could not add dataset entry: {e}")

    logger.info(f"Exported BIDS recording: {fif_path}")
    return fif_path


def export_study_to_bids(
    study_name: str, output_dir: str | Path, db_path: str | Path | None = None
) -> Path:
    """
    Export all recordings in a study to BIDS format.

    Creates complete BIDS dataset with:
        - All recordings exported
        - dataset_description.json
        - participants.tsv

    Args:
        study_name: Study name to export
        output_dir: BIDS output directory
        db_path: Path to database (uses default if None)

    Returns:
        Path to output directory.
    """
    from dendrite.data.storage.database import Database, RecordingRepository

    output_dir = Path(output_dir)
    study_dir = output_dir / study_name
    study_dir.mkdir(parents=True, exist_ok=True)

    # Query recordings (use Database default path if not specified)
    db = Database(db_path) if db_path else Database()
    db.init_db()
    repo = RecordingRepository(db)
    recordings = repo.get_recordings_by_study(study_name)

    if not recordings:
        raise ValueError(f"No recordings found for study: {study_name}")

    # Export each recording (pass output_dir, export_recording_to_bids creates study subdir)
    participants = set()
    for rec in recordings:
        h5_path = rec.get("hdf5_file_path")
        if h5_path and Path(h5_path).exists():
            export_recording_to_bids(h5_path, output_dir)
            participants.add(rec.get("subject_id", "unknown"))
        else:
            logger.warning(f"H5 file not found: {h5_path}")

    # Generate dataset_description.json in study directory
    description = generate_dataset_description(study_name)
    _write_json(study_dir / "dataset_description.json", description)

    # Generate participants.tsv in study directory
    participants_tsv = generate_participants_tsv(list(participants))
    _write_tsv(study_dir / "participants.tsv", participants_tsv)

    logger.info(f"Exported {len(recordings)} recordings to BIDS: {study_dir}")
    return study_dir


def generate_dataset_description(study_name: str) -> dict[str, Any]:
    """Generate BIDS dataset_description.json content."""
    return {
        "Name": study_name,
        "BIDSVersion": BIDS_VERSION,
        "DatasetType": "raw",
        "Authors": ["BMI System"],
        "GeneratedBy": [{"Name": "Dendrite BMI", "Version": "1.0"}],
    }


def generate_participants_tsv(subjects: list[str]) -> list[dict[str, str]]:
    """Generate BIDS participants.tsv content."""
    return [{"participant_id": f"sub-{sub}"} for sub in sorted(subjects)]


def generate_sidecar_json(h5_path: Path, bids_info: dict[str, Any]) -> dict[str, Any]:
    """Generate BIDS _eeg.json sidecar content."""
    h5_path = Path(h5_path)

    # Get channel info
    try:
        ch_info = get_channel_info(str(h5_path))
        n_channels = ch_info["count"]
        sfreq = ch_info.get("sample_rate") or bids_info.get("sample_rate")
    except _IO_ERRORS:
        n_channels = 0
        sfreq = bids_info.get("sample_rate")

    metadata = get_h5_metadata(str(h5_path))

    return {
        "TaskName": bids_info.get("task", "unknown"),
        "SamplingFrequency": sfreq,
        "EEGChannelCount": n_channels,
        "EEGReference": "unknown",
        "PowerLineFrequency": 50,
        "SoftwareFilters": "n/a",
        "RecordingType": "continuous",
        "RecordingDuration": metadata.get("duration", "n/a"),
        "Manufacturer": metadata.get("amplifier", "unknown"),
        "SourceFile": h5_path.name,
    }


def generate_channels_tsv(h5_path: str | Path, dataset: str = "EEG") -> list[dict[str, str]]:
    """Generate BIDS _channels.tsv content."""
    try:
        ch_info = get_channel_info(str(h5_path), dataset)
        labels = ch_info["labels"]
    except _IO_ERRORS:
        return []

    rows = []
    for label in labels:
        ch_type = guess_channel_type(label)
        rows.append({"name": label, "type": ch_type.upper(), "units": "µV", "status": "good"})
    return rows


def generate_events_tsv(h5_path: str | Path, event_dataset: str = "Event") -> list[dict[str, Any]]:
    """Generate BIDS _events.tsv content."""
    h5_path = str(h5_path)
    try:
        df = load_events(h5_path, event_dataset)
    except KeyError:
        return []

    if df.empty:
        return []

    if "timestamp" not in df.columns:
        return []

    type_col = "event_type" if "event_type" in df.columns else None

    # Get EEG start time for onset calculation
    try:
        eeg_df = load_dataset(h5_path, "EEG")
        eeg_df.columns = eeg_df.columns.str.lower()  # Normalize for legacy compat
        eeg_start = eeg_df["timestamp"].iloc[0] if "timestamp" in eeg_df.columns else 0
    except _IO_ERRORS:
        eeg_start = 0

    rows = []
    for _, row in df.iterrows():
        onset = float(row["timestamp"]) - eeg_start
        event = {
            "onset": round(onset, 6),
            "duration": 0,
            "trial_type": row[type_col] if type_col else "event",
        }
        rows.append(event)

    return rows


def _extract_bids_info(h5_path: Path) -> dict[str, Any]:
    """Extract BIDS fields from H5 file metadata or filename."""
    info: dict[str, Any] = {
        "subject": "001",
        "session": "01",
        "task": "task",
        "run": "01",
        "study_name": "study",
        "sample_rate": None,
    }

    # Try H5 metadata first
    try:
        metadata = get_h5_metadata(str(h5_path))
        if "subject_id" in metadata:
            info["subject"] = str(metadata["subject_id"])
        if "session_id" in metadata:
            info["session"] = str(metadata["session_id"])
        if "recording_name" in metadata:
            info["task"] = str(metadata["recording_name"])
        if "study_name" in metadata:
            info["study_name"] = str(metadata["study_name"])
        if "sample_rate" in metadata:
            info["sample_rate"] = float(metadata["sample_rate"])
    except _IO_ERRORS:
        pass

    # Try to parse from filename: sub-XX_ses-XX_task-XX_run-XX_eeg.h5
    filename = h5_path.stem
    patterns = {
        "subject": r"sub-([^_]+)",
        "session": r"ses-([^_]+)",
        "task": r"task-([^_]+)",
        "run": r"run-(\d+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, filename)
        if match:
            info[key] = match.group(1)

    return info


def _copy_sourcedata_files(h5_path: Path, sourcedata_dir: Path, study_name: str) -> None:
    """Copy source H5 and supplementary files to BIDS sourcedata directory."""
    sourcedata_dir.mkdir(parents=True, exist_ok=True)

    # Copy original H5 file
    shutil.copy2(h5_path, sourcedata_dir / h5_path.name)
    logger.info(f"Copied source H5 to: {sourcedata_dir}")

    # Copy supplementary files if found
    supplementary_files = [
        ("config", _find_supplementary_file(h5_path, study_name, "config", "config", ".json")),
        ("metrics", _find_supplementary_file(h5_path, study_name, "metrics", "metrics", ".h5")),
    ]
    for file_type, file_path in supplementary_files:
        if file_path and file_path.exists():
            shutil.copy2(file_path, sourcedata_dir / file_path.name)
            logger.info(f"Copied {file_type} to: {sourcedata_dir}")


def _write_json(path: Path, data: dict) -> None:
    """Write JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _write_tsv(path: Path, rows: list[dict]) -> None:
    """Write TSV file from list of dicts."""
    if not rows:
        return

    headers = list(rows[0].keys())

    with open(path, "w") as f:
        f.write("\t".join(headers) + "\n")
        for row in rows:
            values = [str(row.get(h, "")) for h in headers]
            f.write("\t".join(values) + "\n")


def _update_participants_tsv(path: Path, subject_id: str) -> None:
    """Create or update participants.tsv with a new subject."""
    participant_id = f"sub-{subject_id}"

    # Read existing participants
    existing = set()
    if path.exists():
        with open(path) as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip header
                if line.strip():
                    existing.add(line.strip().split("\t")[0])

    # Add if new
    if participant_id not in existing:
        if not path.exists():
            with open(path, "w") as f:
                f.write("participant_id\n")
                f.write(f"{participant_id}\n")
        else:
            with open(path, "a") as f:
                f.write(f"{participant_id}\n")


def _extract_file_identifier(h5_path: Path) -> str:
    """Extract the BIDS file identifier from H5 filename.

    H5 files follow BIDS pattern: sub-XX_ses-XX_task-XX_run-XX_eeg.h5
    Returns the identifier (everything except _eeg suffix).
    """
    filename = h5_path.stem  # Remove .h5 extension

    # Remove known suffixes to get base identifier
    for suffix in ("_eeg", "_metrics", "_emg"):
        if filename.endswith(suffix):
            return filename[: -len(suffix)]

    return filename


def _get_project_root() -> Path:
    """Get project root directory."""
    # This file is in src/dendrite/data/io/
    return Path(__file__).parent.parent.parent.parent.parent


def _find_supplementary_file(
    h5_path: Path, study_name: str, file_type: str, subdir: str, extension: str
) -> Path | None:
    """Find supplementary file (config/metrics) for a recording.

    Args:
        h5_path: Path to the H5 recording file.
        study_name: Study name for directory lookup.
        file_type: Prefix for filename (e.g., 'config', 'metrics').
        subdir: Subdirectory under data/ (e.g., 'config', 'metrics').
        extension: File extension including dot (e.g., '.json', '.h5').

    Returns:
        Path to the file if found, None otherwise.
    """
    identifier = _extract_file_identifier(h5_path)
    filename = f"{file_type}_{identifier}{extension}"
    project_root = _get_project_root()

    search_paths = [
        project_root / "data" / subdir / study_name / filename,
        project_root / "data" / subdir / filename,
        h5_path.parent / filename,
        h5_path.parent.parent / subdir / study_name / filename,
    ]

    for path in search_paths:
        if path.exists():
            logger.debug(f"Found {file_type}: {path}")
            return path

    logger.debug(f"{file_type.capitalize()} not found for {identifier}")
    return None


def _add_dataset_entry(
    fif_path: Path, study_name: str, bids_base: str, sfreq: float, description: str | None = None
) -> None:
    """Add a dataset entry to the database for the exported FIF file."""
    from dendrite.data.storage.database import Database, DatasetRepository, StudyRepository

    db = Database()
    db.init_db()

    # Get study_id if study exists
    study_repo = StudyRepository(db)
    study = study_repo.get_by_name(study_name)
    study_id = study["study_id"] if study else None

    # Add dataset entry
    dataset_repo = DatasetRepository(db)
    dataset_repo.add_dataset(
        name=bids_base,
        file_path=str(fif_path),
        study_id=study_id,
        sampling_rate=sfreq,
        description=description,
    )
