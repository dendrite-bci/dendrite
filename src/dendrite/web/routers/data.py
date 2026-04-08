"""
Data explorer REST endpoints — studies, recordings, decoders.
"""

import asyncio
from collections.abc import Awaitable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from dendrite.ml.decoders import get_decoder_metadata
from dendrite.web.deps import get_data_service, get_pipeline_service, require_local
from dendrite.web.schemas import StudyCreateRequest, StudyUpdateRequest

router = APIRouter(prefix="/api/data", tags=["data"])


async def _recording_response(coro: Awaitable[Any]) -> Any:
    """Shared error handling for recording-detail endpoints."""
    try:
        result = await coro
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from None
    except OSError as e:
        raw = str(e)
        msg = raw.lower()
        if "unable to lock file" in msg or "already open for write" in msg or "GetLastError() = 33" in raw:
            raise HTTPException(
                409, "Recording file is locked — still being written. Try again after stopping."
            ) from None
        raise
    if result is None:
        raise HTTPException(404, "Recording not found")
    return result


# --- Studies ---


@router.get("/studies")
def list_studies():
    return get_data_service().studies.get_all_studies()


@router.post("/studies")
def create_study(req: StudyCreateRequest):
    return get_data_service().studies.get_or_create(req.study_name, req.description)


@router.get("/studies/{study_id}")
def get_study(study_id: int):
    svc = get_data_service()
    study = svc.studies.get_by_id(study_id)
    if not study:
        raise HTTPException(404, "Study not found")
    study["recording_count"] = len(svc.recordings.get_recordings_by_study(study_id))
    study["decoder_count"] = len(svc.decoders.get_decoders_by_study(study_id))
    return study


@router.put("/studies/{study_id}")
def update_study(study_id: int, req: StudyUpdateRequest):
    svc = get_data_service()
    if not svc.studies.update_study(study_id, description=req.description):
        raise HTTPException(404, "Study not found or no changes")
    return svc.studies.get_by_id(study_id)


@router.delete("/studies/{study_id}")
def delete_study(study_id: int):
    svc = get_data_service()
    study = svc.studies.get_by_id(study_id)
    if study:
        ps = get_pipeline_service()
        if ps.is_recording and ps.study_name == study["study_name"]:
            raise HTTPException(409, "Cannot delete study while recording to it")
    if not svc.studies.delete_study(study_id):
        raise HTTPException(404, "Study not found")
    return {"ok": True}


@router.post("/studies/import-folder")
async def import_study_folder(request: dict[str, Any]):
    """Scan a folder for .h5 files and register them as recordings under a study."""
    folder_path = request.get("folder_path", "")
    study_name = request.get("study_name", "")
    if not folder_path or not study_name:
        raise HTTPException(422, "folder_path and study_name are required")
    try:
        result = await asyncio.to_thread(
            get_data_service().import_study_folder,
            folder_path,
            study_name,
            request.get("description", ""),
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from None


# --- Recordings ---


@router.get("/recordings")
def list_recordings(study_id: int | None = None, search: str | None = None):
    svc = get_data_service()
    if search:
        return svc.recordings.search_recordings(search)
    if study_id is not None:
        return svc.recordings.get_recordings_by_study(study_id)
    return svc.recordings.get_all_recordings()


@router.get("/recordings/{recording_id}")
def get_recording(recording_id: int):
    rec = get_data_service().recordings.get_by_id(recording_id)
    if not rec:
        raise HTTPException(404, "Recording not found")
    return rec


@router.delete("/recordings/{recording_id}")
def delete_recording(recording_id: int):
    try:
        if not get_data_service().recordings.delete_recording(recording_id):
            raise HTTPException(404, "Recording not found")
    except OSError:
        raise HTTPException(409, "Recording file is locked — still being written") from None
    return {"ok": True}


@router.get("/recordings/{recording_id}/file-info")
async def get_recording_file_info(recording_id: int):
    svc = get_data_service()
    return await _recording_response(asyncio.to_thread(svc.get_recording_file_info, recording_id))


@router.get("/recordings/{recording_id}/channels")
async def get_recording_channels(recording_id: int):
    svc = get_data_service()
    return await _recording_response(asyncio.to_thread(svc.get_recording_channels, recording_id))


@router.get("/recordings/{recording_id}/signal-preview")
async def get_signal_preview(
    recording_id: int,
    max_points: int = Query(default=15000, ge=100, le=50000),
    max_channels: int = Query(default=8, ge=1, le=128),
):
    svc = get_data_service()
    return await _recording_response(
        asyncio.to_thread(svc.get_signal_preview, recording_id, max_points, max_channels)
    )


@router.get("/recordings/{recording_id}/erp")
async def get_erp_preview(
    recording_id: int,
    epoch_tmin: float = Query(default=-0.2, ge=-2.0, le=0.0),
    epoch_tmax: float = Query(default=0.8, ge=0.1, le=5.0),
    lowcut: float = Query(default=0.5, ge=0.01, le=100.0),
    highcut: float = Query(default=30.0, ge=1.0, le=500.0),
    apply_rereferencing: bool = Query(default=False),
    modality: str = Query(default="eeg"),
):
    svc = get_data_service()
    return await _recording_response(
        asyncio.to_thread(
            svc.get_erp_preview, recording_id,
            epoch_tmin, epoch_tmax, lowcut, highcut, apply_rereferencing, modality,
        )
    )


@router.get("/recordings/{recording_id}/qc-preview")
async def get_qc_preview(
    recording_id: int,
    lowcut: float = Query(default=0.5, ge=0.1, le=100.0),
    highcut: float = Query(default=50.0, ge=1.0, le=500.0),
    apply_rereferencing: bool = Query(default=True),
    bad_channel_mode: str = Query(default="exclude"),
    max_points: int = Query(default=50000, ge=100, le=200000),
    channels: str = Query(default="", description="Comma-separated channel indices"),
):
    channel_indices = [int(c) for c in channels.split(",") if c.strip()] if channels else None
    svc = get_data_service()
    return await _recording_response(
        asyncio.to_thread(
            svc.get_qc_preview, recording_id, lowcut, highcut, apply_rereferencing,
            bad_channel_mode, max_points, channel_indices,
        )
    )


@router.get("/recordings/{recording_id}/event-summary")
async def get_event_summary(recording_id: int):
    svc = get_data_service()
    return await _recording_response(asyncio.to_thread(svc.get_event_summary, recording_id))


@router.get("/recordings/{recording_id}/summary")
async def get_session_summary(recording_id: int):
    svc = get_data_service()
    return await _recording_response(asyncio.to_thread(svc.get_session_summary, recording_id))


@router.get("/recordings/{recording_id}/telemetry")
async def get_telemetry(recording_id: int):
    svc = get_data_service()
    return await _recording_response(asyncio.to_thread(svc.get_telemetry, recording_id))


@router.get("/recordings/{recording_id}/mode-performance")
async def get_mode_performance(recording_id: int):
    svc = get_data_service()
    return await _recording_response(asyncio.to_thread(svc.get_mode_performance, recording_id))


# --- Decoders ---


@router.get("/decoders")
def list_decoders(study_id: int | None = None, search: str | None = None):
    svc = get_data_service()
    if search:
        return svc.decoders.search_decoders(search)
    if study_id is not None:
        return svc.decoders.get_decoders_by_study(study_id)
    return svc.decoders.get_all_decoders()


def _get_decoder_or_404(decoder_id: int) -> dict:
    dec = get_data_service().decoders.get_decoder_by_id(decoder_id)
    if not dec:
        raise HTTPException(404, "Decoder not found")
    return dec


@router.get("/decoders/{decoder_id}")
def get_decoder(decoder_id: int):
    return _get_decoder_or_404(decoder_id)


@router.get("/decoders/{decoder_id}/metadata")
def get_decoder_metadata_endpoint(decoder_id: int):
    """Read full decoder metadata (event_mapping, label_mapping, etc.) from saved JSON."""
    dec = _get_decoder_or_404(decoder_id)
    try:
        return get_decoder_metadata(dec["decoder_path"])
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(404, str(e)) from None


@router.delete("/decoders/{decoder_id}")
def delete_decoder(decoder_id: int):
    if not get_data_service().decoders.delete_decoder(decoder_id):
        raise HTTPException(404, "Decoder not found")
    return {"ok": True}


# --- File pickers ---


@router.post("/studies/pick-folder", dependencies=[Depends(require_local)])
async def pick_study_folder():
    try:
        path = await asyncio.wait_for(
            asyncio.to_thread(open_native_picker, "folder", "Select folder with recordings"),
            timeout=120,
        )
    except (TimeoutError, asyncio.CancelledError):
        return {"path": None}
    return {"path": path or None}


EEG_FILE_FILTER = (
    "EEG files (*.fif;*.h5;*.hdf5;*.xdf;*.edf;*.set)"
    "|*.fif;*.h5;*.hdf5;*.xdf;*.edf;*.set|All files (*.*)|*.*"
)


def open_native_picker(mode: str, title: str, file_filter: str | None = None) -> str:
    """Open a native OS file/folder picker via subprocess (thread-safe).

    Args:
        mode: "file" or "folder"
        title: Dialog title
        file_filter: Windows-style filter string (file mode only)
    """
    import subprocess
    import sys

    # NOTE: title and file_filter are server-controlled literals — never pass
    # unsanitised user input (PowerShell interprets $, `, and other chars).
    try:
        if sys.platform == "darwin":
            verb = "choose folder" if mode == "folder" else "choose file"
            result = subprocess.run(
                ["osascript", "-e", f"POSIX path of ({verb})"],
                capture_output=True, text=True, timeout=120,
            )
        elif sys.platform == "win32":
            # Escape double-quotes to prevent PowerShell injection
            safe_title = title.replace('"', '`"')
            if mode == "folder":
                ps_script = f"""
Add-Type -AssemblyName System.Windows.Forms
$f = New-Object System.Windows.Forms.FolderBrowserDialog
$f.Description = "{safe_title}"
if ($f.ShowDialog() -eq 'OK') {{ $f.SelectedPath }}
"""
            else:
                filt = (file_filter or "All files (*.*)|*.*").replace('"', '`"')
                ps_script = f"""
Add-Type -AssemblyName System.Windows.Forms
$f = New-Object System.Windows.Forms.OpenFileDialog
$f.Filter = "{filt}"
if ($f.ShowDialog() -eq 'OK') {{ $f.FileName }}
"""
            result = subprocess.run(
                ["powershell", "-Command", ps_script],
                capture_output=True, text=True, timeout=120,
            )
        else:
            result = _linux_picker(mode, title, file_filter)
        return result.stdout.strip() if result.returncode == 0 else ""
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def _linux_picker(mode: str, title: str, file_filter: str | None):
    """Try zenity, fall back to kdialog."""
    import subprocess

    zenity_cmd = ["zenity", "--file-selection", f"--title={title}"]
    kdialog_cmd: list[str]
    if mode == "folder":
        zenity_cmd.append("--directory")
        kdialog_cmd = ["kdialog", "--getexistingdirectory", "."]
    else:
        kdialog_cmd = ["kdialog", "--getopenfilename", "."]
        if file_filter:
            kdialog_cmd.append(file_filter.split("|")[0])
    try:
        return subprocess.run(zenity_cmd, capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        return subprocess.run(kdialog_cmd, capture_output=True, text=True, timeout=120)
