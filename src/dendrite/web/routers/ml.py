"""
ML Workbench REST endpoints — data loading, model listing, training, decoder saving.
"""

from fastapi import APIRouter, HTTPException

from dendrite.web.deps import get_ml_service
from dendrite.web.schemas import (
    MoabbLoadRequest,
    RecordingLoadRequest,
    SaveDecoderRequest,
    TrainingStartRequest,
)

router = APIRouter(prefix="/api/ml", tags=["ml"])


# --- Data Loading ---


@router.get("/moabb/datasets")
async def list_moabb_datasets():
    """Discover available MOABB datasets."""
    return await get_ml_service().discover_moabb_datasets()


@router.post("/moabb/load")
async def load_moabb_dataset(req: MoabbLoadRequest):
    """Load a MOABB dataset into memory for ML workflows."""
    try:
        return await get_ml_service().load_moabb_dataset(req.model_dump())
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except Exception as e:
        raise HTTPException(500, f"Failed to load MOABB dataset: {e}") from e



@router.post("/load-recording")
async def load_recording(req: RecordingLoadRequest):
    """Load a recording HDF5 file into memory for ML workflows."""
    try:
        return await get_ml_service().load_recording(req.model_dump())
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    except Exception as e:
        raise HTTPException(500, f"Failed to load recording: {e}") from e


# --- Models ---


@router.get("/models")
def list_models():
    """List all available model/decoder types from the registry."""
    return get_ml_service().list_models()


@router.get("/models/{model_type}/schema")
def get_model_schema(model_type: str):
    """Get the JSON Schema for a model's config (for dynamic form generation)."""
    schema = get_ml_service().get_model_config_schema(model_type)
    if schema is None:
        raise HTTPException(404, f"Model type '{model_type}' not found")
    return schema


@router.get("/search-categories/{decoder_type}")
def get_search_categories_for_decoder(decoder_type: str):
    """Return available search categories for a specific decoder type."""
    from dendrite.ml.search.search_space import build_decoder_search_space, get_decoder_categories

    categories = get_decoder_categories(decoder_type)
    space = build_decoder_search_space(decoder_type)
    return {
        "categories": categories,
        "total_params": len(space),
    }


# --- Training ---


@router.post("/train")
async def start_training(req: TrainingStartRequest):
    """Start a training job in the background."""
    try:
        return await get_ml_service().start_training(req.model_dump())
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: int):
    """Cancel a running job (training, evaluation, or benchmark)."""
    if not await get_ml_service().cancel_job(job_id):
        raise HTTPException(404, "Job not found or not running")
    return {"ok": True}


# --- Jobs ---


@router.get("/jobs")
def list_jobs(study_id: int | None = None, job_type: str | None = None):
    """List jobs, optionally filtered by study and/or type."""
    return get_ml_service().list_jobs(study_id=study_id, job_type=job_type)


@router.get("/jobs/{job_id}")
def get_job(job_id: int):
    """Get a job with live progress if running."""
    job = get_ml_service().get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@router.post("/jobs/{job_id}/save-decoder")
async def save_decoder(job_id: int, req: SaveDecoderRequest):
    """Save a trained model from a completed job as a named decoder."""
    result = await get_ml_service().save_decoder(
        job_id, req.decoder_name, req.description
    )
    if result is None:
        raise HTTPException(
            400,
            "Cannot save decoder: job not completed or decoder not available",
        )
    return result


# --- Evaluation & Benchmark ---


@router.post("/evaluate")
async def start_evaluation(request: dict):
    """Run epoch-by-epoch evaluation of a trained decoder on loaded data."""
    try:
        return await get_ml_service().start_evaluation(request)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None


@router.post("/jobs/{job_id}/reaggregate")
async def reaggregate_eval(job_id: int, request: dict):
    """Re-aggregate stored eval results with a new decision gate."""
    try:
        return get_ml_service().reaggregate_eval(job_id, request)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None


@router.post("/benchmark")
async def start_benchmark(request: dict):
    """Run k-fold CV benchmark across multiple model types."""
    try:
        return await get_ml_service().start_benchmark(request)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None


@router.delete("/jobs/{job_id}")
def delete_job(job_id: int):
    """Delete a job record."""
    if not get_ml_service().delete_job(job_id):
        raise HTTPException(404, "Job not found")
    return {"ok": True}
