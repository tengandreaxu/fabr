import time
import torch
import torchvision
import numpy as np
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)


class ResNet34:
    def __init__(self, n_output: int, learning_rate: float, weight_decay: float):

        self.model = torchvision.models.resnet34()
        self.model.fc = torch.nn.Linear(
            in_features=self.model.fc.in_features, out_features=n_output
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=learning_rate, weight_decay=5e-4
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.test_accuracies = []
        self.train_accuracies = []
        self.logger = logging.getLogger("ResNet34")

    def get_test_predictions(
        self,
        validation_loader: torch.utils.data.dataloader.DataLoader,
    ) -> Tuple[torch.Tensor, float, float]:
        with torch.no_grad():
            accuracies = []
            losses = []
            outputs = []
            for batch in validation_loader:
                images, labels = batch

                output = self.model(images)
                outputs.append(output)
                loss = torch.nn.CrossEntropyLoss()(output, labels)
                accuracies.append(self.accuracy(output, labels))
                losses.append(loss.item())
        return torch.cat(outputs, 0), np.mean(accuracies), np.mean(losses)

    def train(self, train_loader, test_loader, epochs: int, seed: int):
        torch.manual_seed(seed)
        accuracies = []
        average_epoch_time = []
        for epoch in range(epochs):

            start_epoch_time = time.monotonic()
            total_labels = 0
            correct = 0
            batch = 0

            if epoch == 80 or epoch == 120:
                self.optimizer.__dict__["param_groups"][0]["weight_decay"] = (
                    self.optimizer.__dict__["param_groups"][0]["weight_decay"] / 10
                )
            for train, labels in train_loader:
                batch += 1
                self.model.zero_grad()
                output = self.model(train)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()

                _, train_correct = torch.max(output.data, 1)
                correct += (train_correct == labels).sum().item()
                total_labels += len(labels)
                train_accuracy = correct * 100 / total_labels
            _, test_acc, _ = self.get_test_predictions(test_loader)
            test_acc = test_acc * 100
            accuracies.append(
                {
                    "train_accuracy": train_accuracy,
                    "test_accuracy": test_acc,
                    "epoch": epoch,
                }
            )

            end_time = time.monotonic()
            epoch_time = end_time - start_epoch_time
            average_epoch_time.append(epoch_time)
            self.test_accuracies.append(test_acc)
            self.train_accuracies.append(train_accuracy)

            mean_time = np.mean(average_epoch_time)
            logging.info(
                f"Epoch: {epoch}\tTrain Loss: {loss:.3f}\tTrain Acc.: {train_accuracy:.3f}%\tTest Acc.: {test_acc:.3f}\tTime: {mean_time:.3f}s"
            )
        self.accuracies = accuracies

    def save_model(self, file_name: str):
        torch.save(self.state_dict(), file_name)

    def accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))
