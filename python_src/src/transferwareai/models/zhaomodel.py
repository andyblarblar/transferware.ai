# This file adapts the method by Zhao et. al. in https://doi.org/10.1016/j.daach.2023.e00269
from pathlib import Path

import torch
import torchvision
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from .adt import Model, Validator, Trainer, AbstractModelFactory, ImageMatch
from .utils import create_test_train_split
from ..data.dataset import CacheDataset


class ZhaoModel(Model):

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

    def train(self, dataset: CacheDataset) -> Model: # TODO test, should work but need to download weights
        # data pre-processing
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        global_step = 0

        # Loading pre-trained parameters of vgg16
        model_vgg16_pth = torch.load(
            self._outer_dataset.joinpath("vgg16.pth")
        )  # TODO test

        # Recording loss sum, prediction matrix, and tp, tn, fp, fn in each epoch
        loss_sum_list = []

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
        device = torch.device("cpu")  # TODO make global param
        model = model_vgg16.to(device)

        # Using Adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Defining the loss function
        criteon = torch.nn.CrossEntropyLoss()

        # Split into test and train
        test_set, train_set = create_test_train_split(dataset, test_size=0.2)

        test_dataloader = DataLoader(test_set, batch_size=10, shuffle=False)
        train_dataloader = DataLoader(train_set, batch_size=10, shuffle=True)

        # Training process begins
        for epoch in range(epochs):
            loss_sum = 0
            for step, (x, y) in tqdm(enumerate(train_dataloader)):
                # Migrating data to gpu
                x, y = x.to(device), y.to(device)

                # Preprocess
                x = transform(x)

                # Turn on model training mode
                model.train()
                # Generating predictions
                logits = model(x)
                # Calculating losses
                loss = criteon(logits, y)
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

            # Recording loss sum
            loss_sum_list.append(loss_sum)

            # Opening the model evaluation mode
            model.eval()

            # Evaluating in test sets
            for step, (x, y) in enumerate(test_dataloader):
                x, y = x.to(device), y.to(device)

                # Preprocess
                x = transform(x)

                # Calculating model prediction results
                logits = model(x)
                # TODO how to test????

            # TODO only save if best
            torch.save(
                model_vgg16.state_dict(),
                self._outer_dataset.joinpath("vgg16_train.pth"),
            )


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
