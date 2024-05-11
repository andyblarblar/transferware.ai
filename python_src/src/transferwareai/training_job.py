from pathlib import Path
import logging
import requests
import tarfile

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
            if not self._deploy_model(trained_model):
                logging.error("Failed to deploy model.")
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

    def _deploy_model(self) -> bool:
        res_path = Path(settings.training.resource_dir)
        host = settings.query.host
        port = settings.query.port

        # Tarball resources
        tar_path = res_path.parent / "model.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            logging.info("Creating tarball")
            for f in res_path.glob("*"):
                logging.debug(f"Adding {f}")
                tar.add(f)
            logging.info("Tarball created")

        # Upload to query API
        logging.info("Uploading model to query API")
        try:
            with tarfile.open(tar_path, "r") as f:
                r = requests.post(
                    f"http://{host}:{port}/update",
                    files={"file": f},
                    headers={"Authorization": settings.access_token},
                )
                r.raise_for_status()
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to upload model: {e}")
            return False
        except tarfile.ReadError as e:
            logging.error(f"Failed to read tarball: {e}")
            return False
        logging.info("Model uploaded")

        # Reload model
        logging.info("Reloading model")
        try:
            r = requests.post(
                f"http://{host}:{port}/reload",
                headers={"Authorization": settings.access_token},
            )
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to reload model: {e}")
            return False
        logging.info("Model reloaded")

        # Cleanup
        tar_path.unlink()
        # Model fully deployed
        return True
