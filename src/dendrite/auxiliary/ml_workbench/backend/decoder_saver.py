"""Save trained decoders to disk and register in database."""

import json
from typing import Any

from dendrite.auxiliary.ml_workbench.backend.types import TrainResult
from dendrite.data.storage.database import Database, DecoderRepository, StudyRepository
from dendrite.ml.decoders.decoder_schemas import DecoderConfig
from dendrite.utils.logger_central import get_logger

logger = get_logger(__name__)


def _extract_decoder_metadata(
    result: TrainResult,
    training_dataset_name: str | None = None,
) -> dict[str, Any]:
    """Extract metadata from training result for database storage."""
    metadata: dict[str, Any] = {
        "cv_mean": None,
        "cv_std": None,
        "cv_folds": None,
        "channel_names_json": None,
        "class_labels_json": None,
        "modality": None,
        "training_dataset_name": training_dataset_name,
    }

    if result.cv_results:
        metadata["cv_mean"] = result.cv_results.get("mean_accuracy")
        metadata["cv_std"] = result.cv_results.get("std_accuracy")
        metadata["cv_folds"] = result.cv_results.get("n_folds")

    decoder_config = result.decoder.config
    if decoder_config.channel_labels:
        metadata["channel_names_json"] = json.dumps(decoder_config.channel_labels)
    if decoder_config.label_mapping:
        metadata["class_labels_json"] = json.dumps(decoder_config.label_mapping)
    if decoder_config.modality:
        metadata["modality"] = decoder_config.modality

    return metadata


def _prepare_search_result_json(optuna_results: dict[str, Any] | None) -> str | None:
    """Prepare search result JSON if Optuna search was done."""
    if not optuna_results:
        return None
    return json.dumps(
        {
            "n_trials": optuna_results.get("n_completed", 0),
            "best_value": optuna_results.get("best_value"),
            "best_params": optuna_results.get("best_params", {}),
            "n_pruned": optuna_results.get("n_pruned", 0),
            "n_failed": optuna_results.get("n_failed", 0),
        }
    )


def save_decoder(
    result: TrainResult,
    decoder_name: str,
    study_name: str,
    description: str = "",
    optuna_results: dict[str, Any] | None = None,
    final_config: DecoderConfig | None = None,
    training_dataset_name: str | None = None,
) -> str | None:
    """Save trained decoder to disk and register in database.

    Args:
        result: Training result containing the decoder
        decoder_name: User-provided name for the decoder
        study_name: Study name for organizing the decoder
        description: Optional description
        optuna_results: Optuna search results if search was done
        final_config: Best config from Optuna search, or None for base config
        training_dataset_name: Name of the training dataset

    Returns:
        Path to saved decoder, or None if save failed
    """
    try:
        saved_path = result.decoder.save(decoder_name, study_name=study_name)

        config = final_config or result.decoder.config
        metadata = _extract_decoder_metadata(result, training_dataset_name)

        db = Database()
        study_repo = StudyRepository(db)
        decoder_repo = DecoderRepository(db)

        study = study_repo.get_or_create(study_name)
        study_id = study["study_id"]

        decoder_id = decoder_repo.add_decoder(
            study_id=study_id,
            decoder_name=decoder_name,
            decoder_path=saved_path,
            model_type=config.model_type,
            training_accuracy=result.accuracy,
            validation_accuracy=result.val_accuracy,
            cv_mean_accuracy=metadata["cv_mean"],
            cv_std_accuracy=metadata["cv_std"],
            cv_folds=metadata["cv_folds"],
            source="offline_trainer",
            description=description,
            channel_names=metadata["channel_names_json"],
            class_labels=metadata["class_labels_json"],
            training_dataset_name=metadata["training_dataset_name"],
            modality=metadata["modality"],
            training_config=json.dumps(config.model_dump(exclude_none=True)),
            search_result=_prepare_search_result_json(optuna_results),
        )

        if decoder_id:
            logger.info(f"Registered decoder in database with ID: {decoder_id}")
        else:
            logger.warning("Failed to register decoder in database (may already exist)")

        return saved_path

    except Exception as e:
        logger.error(f"Failed to save decoder: {e}")
        return None
