from typing import Optional

from ..config import settings
from ..models.adt import Model, Validator, AbstractModelFactory
from ..models.construct import get_abstract_factory

# Model singleton
_model: Optional[Model] = None


def initialize_model():
    """Initializes the model as given by the config."""
    global _model
    factory = get_abstract_factory(settings.model_implimentation, "query")
    _model = factory.get_model()


def get_model() -> Model:
    """Returns the query model being used. Must have been initialized first."""
    global _model
    if _model is None:
        raise ValueError("Model is not initialized")

    return _model
