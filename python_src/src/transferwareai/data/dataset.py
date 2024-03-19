from torch.utils.data import Dataset


class CacheDataset(Dataset):
    """Dataset wrapping the TCC api cache."""

    def class_labels(self) -> list[str]:
        pass  # TODO impl
