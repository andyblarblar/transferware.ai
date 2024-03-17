from abc import ABC, abstractmethod
from torch import Tensor
from torch.utils.data import Dataset
from dataclasses import dataclass


@dataclass
class ImageMatch:
    """An image matching to the query image/"""
    id: int
    """Id of matching image"""
    confidence: float
    """Confidence metric of matching image. Could be 0-1, or a distance, depending on model."""


class Model(ABC):
    """Interface for query models."""

    @abstractmethod
    def query(self, image: Tensor) -> list[ImageMatch]:
        """Takes the input image, and finds the 10 closest images from the dataset used for training."""
        ...

    @abstractmethod
    def reload(self):
        """Reloads the model from disk."""
        ...


class Trainer(ABC):
    """Interface for methods of training models."""

    @abstractmethod
    def train(self, dataset: Dataset) -> Model:
        """Creates a model, training it and creating related files like caches."""
        ...


class Validator(ABC):
    """Interface for validation techniques."""

    @abstractmethod
    def validate(self, model: Model, validation_set: Dataset) -> float:
        """
        Validates the model against real world, untrained data. Returns validation percent. This is a validation
        of the model in the final system, rather than validation in a deep learning context.
        """
        ...


class AbstractModelFactory(ABC):
    """Interface for creating families of query models."""
    @abstractmethod
    def get_model(self) -> Model:
        ...

    @abstractmethod
    def get_trainer(self) -> Trainer:
        ...

    @abstractmethod
    def get_validator(self) -> Validator:
        ...
