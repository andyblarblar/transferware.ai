from pathlib import Path
import logging

from torchvision.datasets import ImageFolder

from transferwareai.models.adt import Model
from transferwareai.models.construct import get_abstract_factory
from transferwareai.data.dataset import CacheDataset, ApiCache
from transferwareai.config import settings


class TrainingJob:
    """Simple wrapper class over the training pipeline."""

    def exec(self, update_cache: bool = True):
        """Runs the training pipeline end to end. Parameters are taken from the global config object."""
        logging.info("Starting training job")

        # Retrieve implementation being used
        factory = get_abstract_factory(settings.model_implimentation, "training")

        logging.info(f"Loaded factory {factory.__class__.__name__}")

        logging.info(f"Updating cache")

        # Update cache
        train_ds = self._get_train_ds(update_cache)

        logging.info("Cache updated")

        # Train model
        trainer = factory.get_trainer()

        logging.info("Starting training")
        trained_model = trainer.train(train_ds)

        # Validate model
        logging.info("Starting validation")
        valid_ds = self._get_valid_ds()

        validator = factory.get_validator()
        class_val, val_percent = validator.validate(trained_model, valid_ds)

        logging.info(f"Class val: {class_val}")
        logging.info(f"Total val: {val_percent}")

        # Deploy model if over threshold
        if val_percent > settings.training.validation_threshold:
            self._deploy_model(trained_model)
        else:
            logging.error("Validation under threshold! Not deploying.")

    def _get_train_ds(self, update_cache: bool = True) -> CacheDataset:
        res_path = Path(settings.training.resource_dir)

        # Update TCC cache
        api = ApiCache.from_cache(
            res_path.joinpath("cache"), no_update=not update_cache
        )
        return CacheDataset(api, skip_ids=settings.training.skip_ids)

    def _get_valid_ds(self) -> ImageFolder:
        res_path = Path(settings.training.validation_dir)

        return ImageFolder(str(res_path.absolute()))

    def _deploy_model(self, model: Model):
        pass  # TODO
