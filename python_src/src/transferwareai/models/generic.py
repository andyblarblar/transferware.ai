from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import logging
from .adt import Validator, Model


class GenericValidator(Validator):
    """Applies the validation strategy entirely on the model interface."""
    def __init__(self, device: str):
        super().__init__()
        self.device = device

    def validate(self, model: Model, validation_set: ImageFolder) -> float:
        logging.debug("Starting generic validation")
        num_correct = 0

        dl = DataLoader(validation_set, shuffle=False, batch_size=1)
        idx_to_class = {j: i for (i, j) in validation_set.class_to_idx.items()}

        for img, id in dl:
            img = img.to(self.device)

            matches = model.query(img[0])
            id = int(idx_to_class[int(id[0])])

            # Check if correct id is in top 10 matches
            for match in matches:
                if match.id == id:
                    num_correct += 1
                    break

        return num_correct / len(validation_set)
