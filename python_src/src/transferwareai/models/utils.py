from torch.utils.data import Subset, Dataset
from sklearn.model_selection import train_test_split

from transferwareai.data.dataset import CacheDataset


def create_test_train_split(
    data: CacheDataset, test_size: float
) -> tuple[Dataset, Dataset]:
    """Creates a stratified test and train split of the dataset."""

    # generate indices: instead of the actual data we pass in integers instead
    train_indices, test_indices, _, _ = train_test_split(
        range(len(data)),
        data.targets,
        stratify=data.targets,
        test_size=test_size,
    )

    # generate subset based on indices
    train_split = Subset(data, train_indices)
    test_split = Subset(data, test_indices)

    return test_split, train_split
