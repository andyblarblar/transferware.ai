from pathlib import Path
from typing import Optional
from aiorwlock import RWLock

from ..config import settings
from ..models.adt import Model, Validator, AbstractModelFactory
from ..models.construct import get_abstract_factory
from ..data.dataset import CacheDataset
from ..tccapi.api_cache import ApiCache

# Model singleton
_model: Optional[Model] = None
_model_lock = RWLock()

_ds: Optional[CacheDataset] = None
_ds_lock = RWLock()

_api: Optional[ApiCache] = None
_api_lock = RWLock()


def initialize_model():
    """Initializes the model as given by the config."""
    global _model, _ds, _api
    factory = get_abstract_factory(settings.model_implimentation, "query")
    _model = factory.get_model()

    res_path = Path(settings.query.resource_dir)
    _api = ApiCache.from_cache(
        res_path.joinpath("cache"), no_update=not settings.update_cache
    )
    _ds = CacheDataset(_api, skip_ids=settings.training.skip_ids)


async def reload_model():
    """Reloads the model from disk."""
    global _model, _model_lock
    async with _model_lock.writer_lock:
        factory = get_abstract_factory(settings.model_implimentation, "query")
        _model = factory.get_model()


async def reload_api_cache(update: bool = True):
    """Reloads the api cache from disk, optionally updating the cache in the process."""
    global _api, _api_lock, _ds, _ds_lock

    async with _api_lock.writer_lock:
        async with _ds_lock.writer_lock:
            res_path = Path(settings.query.resource_dir)
            _api = ApiCache.from_cache(
                res_path.joinpath("cache"),
                no_update=(not settings.update_cache) and (not update),
            )
            _ds = CacheDataset(_api, skip_ids=settings.training.skip_ids)


async def get_model() -> Model:
    """Returns the query model being used. Must have been initialized first."""
    global _model, _model_lock
    if _model is None:
        raise ValueError("Model is not initialized")

    # Lock model whenever it is in use to allow for updates
    try:
        await _model_lock.reader_lock.acquire()
        yield _model
    finally:
        _model_lock.reader_lock.release()


async def get_api() -> ApiCache:
    """Returns a handle to the api cache."""
    global _api, _api_lock
    if _api is None:
        raise ValueError("Model is not initialized")

    try:
        await _api_lock.reader_lock.acquire()
        yield _api
    finally:
        _api_lock.reader_lock.release()


async def get_dataset() -> CacheDataset:
    """Returns a handle to the cache dataset."""
    global _ds, _ds_lock
    if _ds is None:
        raise ValueError("Model is not initialized")

    try:
        await _ds_lock.reader_lock.acquire()
        yield _ds
    finally:
        _ds_lock.reader_lock.release()
