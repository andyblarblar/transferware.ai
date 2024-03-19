from torch.utils.data import Dataset

from transferwareai.tccapi.api_cache import ApiCache


class CacheDataset(Dataset):
    """Dataset wrapping the TCC api cache."""

    def __init__(self, cache: ApiCache):
        self._cache = cache

    def class_labels(self) -> list[str]:
        pass  # TODO impl
