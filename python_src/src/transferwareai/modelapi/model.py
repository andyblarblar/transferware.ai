from pathlib import Path
from typing import Optional

from ..config import settings
from ..models.adt import Model, Validator, AbstractModelFactory
from ..models.construct import get_abstract_factory
from ..data.dataset import CacheDataset
from ..tccapi.api_cache import ApiCache

# Model singleton
_model: Optional[Model] = None
_ds: Optional[CacheDataset] = None
_api: Optional[ApiCache] = None


def initialize_model():
    """Initializes the model as given by the config."""
    global _model, _ds, _api
    factory = get_abstract_factory(settings.model_implimentation, "query")
    _model = factory.get_model()

    res_path = Path(settings.query.resource_dir)
    _api = ApiCache.from_cache(res_path.joinpath("cache"), no_update=False)
    _ds = CacheDataset(_api, skip_ids=settings.training.skip_ids)


def get_model() -> Model:
    """Returns the query model being used. Must have been initialized first."""
    global _model
    if _model is None:
        raise ValueError("Model is not initialized")

    return _model


def get_api() -> ApiCache:
    """Returns a handle to the api cache."""
    global _api
    if _api is None:
        raise ValueError("Model is not initialized")

    return _api


def get_dataset() -> CacheDataset:
    """Returns a handle to the cache dataset."""
    global _ds
    if _ds is None:
        raise ValueError("Model is not initialized")

    return _ds
