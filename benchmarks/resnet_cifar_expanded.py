import os
import numpy as np
import pandas as pd
import torch
from utils.file_handlers import save_pickle
from models.ResNet34 import ResNet34
from data.DatasetManager import DatasetsManager

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    """Following Arora et al 2020, we use ResNet34 as benchmark with:

    - 160 training epochs
    - 0.1 learning rate
    - 5e-4 weight decay <- decay at epoch 80 and 120 by 10
    - 0.9 momentum
    """
    weight_decay = 0  # 5e-4 Arora uses 5e-4
    learning_rate = 0.001  # 0.1 arora uses 0.001

    dm = DatasetsManager()
    epochs = 160
    runs = 20
    batch_size = 32  #  160 Arora uses 160
    ns = [256, 512, 1024, 2048, 5000]

    for n in ns:
        output_dir = (
            f"results/resnet_on_cifar/{n}_batch={batch_size}_lr={learning_rate}"
        )
        os.makedirs(output_dir, exist_ok=True)
        for seed in range(runs):
            # Load all and sample with seed as run
            resnet34 = ResNet34(
                n_output=10, learning_rate=learning_rate, weight_decay=weight_decay
            )
            train_loader, test_loader = dm.torch_load_cifar_10_as_tensors(
                normalize=True,
                subsample=n,
                batch_size=batch_size,
                seed=seed,
                device=device,
            )

            resnet34.train(
                train_loader=train_loader,
                test_loader=test_loader,
                epochs=epochs,
                seed=seed,
            )
            y_test = []
            y_hat = []
            for data, labels in test_loader:
                y_test += labels.cpu().numpy().tolist()
                with torch.no_grad():
                    predictions = resnet34.model(data)
                    _, y_hat_sub = torch.max(predictions, dim=1)
                    y_hat += y_hat_sub
            save_pickle(y_test, os.path.join(output_dir, f"seed_{seed}_y_test.pickle"))
            save_pickle(y_hat, os.path.join(output_dir, f"seed_{seed}_y_hat.pickle"))
            accuracies = pd.DataFrame(resnet34.accuracies)

            accuracies.to_csv(
                os.path.join(
                    output_dir,
                    f"seed_{seed}_test_accuracies.csv",
                )
            )
