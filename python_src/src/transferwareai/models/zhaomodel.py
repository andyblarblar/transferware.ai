# This file adapts the method by Zhao et. al. in https://doi.org/10.1016/j.daach.2023.e00269
import logging
from pathlib import Path

import torch
import torchvision
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .adt import Model, Validator, Trainer, AbstractModelFactory, ImageMatch
from .utils import create_test_train_split
from ..data.dataset import CacheDataset


class ZhaoModel(Model):
    def __init__(self, resource_dir: Path) -> None:
        super().__init__()
        # TODO load annoy and model

    def query(self, image: Tensor) -> list[ImageMatch]:
        pass

    def reload(self):
        pass

    def get_resource_files(self) -> list[Path]:
        pass


class ZhaoTrainer(Trainer):
    def __init__(self, outer_dataset: Path):
        super().__init__()
        self._outer_dataset = outer_dataset

    def train(self, dataset: CacheDataset) -> Model:
        logging.debug("Entering Zhao trainer")

        # data pre-processing
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        dataset.set_transforms(transform)

        global_step = 0

        # Loading pre-trained parameters of vgg16
        model_vgg16_pth = torch.load(self._outer_dataset.joinpath("vgg16.pth"))

        # loading vgg16 model
        model_vgg16 = torchvision.models.vgg16()

        model_vgg16.load_state_dict(model_vgg16_pth)

        model_vgg16.classifier[2] = torch.nn.Dropout(p=0.5)
        model_vgg16.classifier[5] = torch.nn.Dropout(p=0.5)

        # Changing the model structure to fit the categories
        model_vgg16.classifier[6] = torch.nn.Linear(4096, dataset.class_num())

        # Setting learning rate
        lr = 1e-5
        # Training the model for certain number of epochs (in this case we will use 30 epochs)
        epochs = 30  # TODO update

        # Using cuda for gpu training
        device = torch.device("cuda")  # TODO make global param
        model = model_vgg16.to(device)

        # Using Adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Defining the loss function
        criteon = torch.nn.CrossEntropyLoss()

        # Split into test and train
        test_set, train_set = create_test_train_split(dataset, test_size=0.3)

        test_dataloader = DataLoader(
            test_set, batch_size=15, shuffle=False, num_workers=10
        )
        train_dataloader = DataLoader(
            train_set, batch_size=15, shuffle=True, num_workers=10
        )

        logging.debug("Data prepared, starting training")

        writer = SummaryWriter(
            str(self._outer_dataset.joinpath("tensorboard_logs").absolute())
        )

        best_val_loss = 1e20

        # Training process begins
        for epoch in range(epochs):
            logging.debug(f"Starting epoch {epoch}")
            loss_sum = 0
            for step, (x, y) in tqdm(
                enumerate(train_dataloader), total=len(train_dataloader)
            ):
                # Migrating data to gpu
                x, y = x.to(device), y.to(device)

                # Turn on model training mode
                model.train()
                # Generating predictions
                logits = model(x)
                # Calculating losses
                loss = criteon.forward(logits, y)
                # Recording total losses
                loss_sum = loss_sum + loss.detach()
                # print(loss)
                # Optimizer gradient zeroed
                optimizer.zero_grad()
                # Back propagation of loss to obtain loss gradient
                loss.backward()
                # Optimizing model
                optimizer.step()
                # Recording global step count
                global_step += 1

                writer.add_scalar("train/loss", loss, global_step)
            loss_sum /= len(train_dataloader)

            # Opening the model evaluation mode
            model.eval()

            logging.debug("Train complete, starting test")

            loss_sum_eval = 0
            # Evaluating in test sets
            for step, (x, y) in tqdm(
                enumerate(test_dataloader), total=len(test_dataloader)
            ):
                x, y = x.to(device), y.to(device)

                # Calculating model prediction results
                logits = model(x)
                # Calculating losses
                loss = criteon.forward(logits, y)
                loss_sum_eval = loss_sum_eval + loss.detach()

                writer.add_scalar("val/loss", loss, (epoch + 1) * step)
            loss_sum_eval /= len(test_dataloader)

            writer.add_scalars(
                "Train vs Test", {"val": loss_sum_eval, "train": loss_sum}, global_step
            )

            # Save model if improved
            if loss_sum_eval < best_val_loss:
                logging.debug("Saving best model")
                torch.save(
                    model_vgg16.state_dict(),
                    self._outer_dataset.joinpath("vgg16_train.pth"),
                )
                best_val_loss = loss_sum_eval

        # TODO generate annoy cache here
        return ZhaoModel(resource_dir=self._outer_dataset)


class ZhaoValidator(Validator):
    def validate(self, model: Model, validation_set: Dataset) -> float:
        pass


class ZhaoModelFactory(AbstractModelFactory):

    def get_model(self) -> Model:
        pass

    def get_trainer(self) -> Trainer:
        return ZhaoTrainer(self._resource_path)

    def get_validator(self) -> Validator:
        return ZhaoValidator()
