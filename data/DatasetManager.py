import os
import random
import torchvision
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import torch
from utils.file_handlers import load_pickle, save_pickle
from torch.utils.data import DataLoader, Subset
from typing import Optional, Tuple
from dataclasses import dataclass
from parameters.SimpleConvolutionParameters import SimpleConvolutionParameters

from models.SimpleConvolution import SimpleConvolution
import logging
from torch.utils.data._utils.collate import default_collate

logging.basicConfig(level=logging.INFO)


@dataclass
class DatasetsManager:
    def __init__(self):
        self.logger = logging.getLogger("DatasetsManager")

    def get_dataset(
        self,
        dataset_name: str,
        random_state: int,
        flatten: bool = False,
        conv_parameters: SimpleConvolutionParameters = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """returns the x_train, y_train, x_test, y_test from a given dataset"""
        torch.random.manual_seed(random_state)
        np.random.seed(random_state)

        if dataset_name in [
            "cifar10",
        ]:
            return self.__load_from_torch(
                dataset_name=dataset_name,
                conv_parameters=conv_parameters,
                flatten=flatten,
                random_state=random_state,
            )
        else:
            raise Exception(f"Dataset {dataset_name} not found.")

    def __load_from_torch(
        self,
        dataset_name: str,
        conv_parameters: SimpleConvolutionParameters,
        flatten: bool,
        random_state: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """returns the mnist dataset"""

        trainset, testset = self.__load_torch_cifar10()

        batch_size_train = len(trainset)
        batch_size_test = len(testset)

        train_loader = DataLoader(
            trainset,
            batch_size=batch_size_train,
        )
        test_loader = DataLoader(
            testset,
            batch_size=batch_size_test,
        )

        for data in train_loader:
            x_train, y_train = data

        for data in test_loader:
            x_test, y_test = data
        if flatten:
            x_train = torch.stack([x.flatten() for x in x_train])
            x_test = torch.stack([x.flatten() for x in x_test])
            x_train = x_train.numpy()
            x_test = x_test.numpy()
        y_test = y_test.numpy().reshape(-1, 1)
        y_train = y_train.numpy().reshape(-1, 1)
        if conv_parameters is not None and not flatten:
            folder = f"data/preprocessed/{dataset_name}"
            os.makedirs(folder, exist_ok=True)
            unique_name = f"seed_{random_state}_channels_{conv_parameters.channels}_gpa_{conv_parameters.global_average_pooling}"

            X_test_name = os.path.join(folder, f"{unique_name}_X_test.pickle")
            y_test_name = os.path.join(folder, f"{unique_name}_y_test.pickle")
            X_train_name = os.path.join(folder, f"{unique_name}_X_train.pickle")
            y_train_name = os.path.join(folder, f"{unique_name}_y_train.pickle")

            if os.path.exists(X_test_name) and os.path.exists(X_train_name):
                x_train = load_pickle(X_train_name)
                x_test = load_pickle(X_test_name)
                y_train = load_pickle(y_train_name)
                y_test = load_pickle(y_test_name)

            else:
                simple_convolution = SimpleConvolution(
                    in_channels=x_train[0].shape[0],
                    channels=conv_parameters.channels,
                    global_average_pooling=conv_parameters.global_average_pooling,
                    batch_norm=conv_parameters.batch_norm,
                    seed=random_state,
                )
                with torch.no_grad():
                    x_train = simple_convolution(x_train).detach().numpy()

                    x_test = simple_convolution(x_test).detach().numpy()
                save_pickle(x_train, X_train_name)
                save_pickle(x_test, X_test_name)
                save_pickle(y_train, y_train_name)
                save_pickle(y_test, y_test_name)
        return (
            x_train,
            x_test,
            y_train,
            y_test,
        )

    def __load_torch_cifar10(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )

        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
        return trainset, testset

    def torch_load_cifar_10_as_tensors(
        self,
        batch_size: Optional[int] = None,
        normalize: Optional[bool] = False,
        subsample: Optional[int] = 0,
        seed: Optional[int] = 0,
        device: Optional[torch.device] = torch.device("cpu"),
    ):
        """Used for Resnet-34"""
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        if normalize:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        else:
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.RandomRotation([90, 90])]
            )

        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )

        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )

        if not batch_size:
            batch_size_train = len(trainset.data)
            batch_size_test = len(testset.data)
        else:
            batch_size_train = batch_size
            batch_size_test = batch_size

        # Get a balanced torch Subset
        if subsample > 0:
            train_idxs = []
            for label in range(10):
                train_idx = np.where((np.array(trainset.targets) == label))[0]
                train_idx = np.random.choice(train_idx, size=subsample, replace=False)
                train_idxs.append(train_idx)

            train_idxs = np.concatenate(train_idxs)
            random.shuffle(train_idxs)
            trainset = Subset(trainset, train_idxs)

        train_loader = DataLoader(
            trainset,
            batch_size=batch_size_train,
            collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)),
        )
        test_loader = DataLoader(
            testset,
            batch_size=batch_size_test,
            collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)),
        )
        return train_loader, test_loader

    # ***********************
    # Simulated Data
    # ***********************
    def load_synthetic_data(
        self,
        n_observations: int,
        n_features: int,
        split_size: int,
        noise: Optional[bool] = True,
    ) -> Tuple[np.ndarray, str]:
        """creates and returns crafted synthetic data"""
        np.random.seed(0)
        raw_signals = np.random.randn(n_observations, n_features)
        beta = np.random.randn(1, raw_signals.shape[1])
        norm_beta = beta / np.linalg.norm(beta)

        if noise:
            epsilon = np.random.normal(0, np.sqrt(0.5), raw_signals.shape[0])
        else:
            epsilon = np.zeros(raw_signals.shape[0])

        # y = X * beta + eps
        labels = raw_signals @ norm_beta.T + epsilon.reshape(-1, 1)

        x_train = raw_signals[:-split_size, :]
        x_test = raw_signals[-split_size:, :]
        y_train = labels[:-split_size, :]
        y_test = labels[-split_size:, :]

        return x_train, y_train, x_test, y_test

    def get_synthetic_dataset_normal_dist(
        self,
        n_observations: int,
        n_features: int,
        split_number: int,
        number_classes: int = 10,
        seed: int = 0,
    ):
        np.random.seed(seed)

        X_ = np.random.standard_normal([n_observations, n_features])
        betas = np.random.standard_normal([n_features, 1])
        labels = np.round(X_ @ betas + np.random.standard_normal([n_observations, 1]))
        qq = np.quantile(
            labels, q=np.arange(0, 1 + 1 / number_classes, 1 / number_classes)
        )

        lab = 0 * labels
        i = 0
        while i < number_classes:
            lab += (labels >= qq[i]) * (labels < qq[i + 1]) * i
            i += 1

        x_train = X_[:-split_number, :]
        x_test = X_[-split_number:, :]
        y_train = lab[:-split_number]
        y_test = lab[-split_number:]

        return x_train, y_train, x_test, y_test
