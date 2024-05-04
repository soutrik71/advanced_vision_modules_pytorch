import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.tuner import Tuner

from src.lightning_base.dataloader import Cifar10DataModule
from src.lightning_base.trainer import (
    Cifar10ModelLightningBase,
    Cifar10ModelLightningCosineLR,
    Cifar10ModelLightningExpLR,
)


def loss_metric_plot(trainer):
    # Plotting the metrics
    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")

    aggreg_metrics = []
    agg_col = "epoch"
    for i, dfg in metrics.groupby(agg_col):
        agg = dict(dfg.mean())
        agg[agg_col] = i
        aggreg_metrics.append(agg)
    df_metrics = pd.DataFrame(aggreg_metrics)

    df_metrics[["train_loss", "val_loss"]].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="Loss"
    )
    df_metrics[["train_acc", "val_acc"]].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="ACC"
    )
    plt.show()


def prediction_loop(model, device, dataloader):
    model.eval()
    dataloader.setup("test")
    features, targets, predictions = [], [], []

    with torch.no_grad():
        for data_batch, labels in dataloader.test_dataloader():
            data_batch, labels = data_batch.to(device), labels.to(device)

            for data, target in zip(data_batch, labels):
                data = data.unsqueeze(0)
                pred_logit = model(data)
                pred_label = torch.argmax(pred_logit, dim=1).squeeze()

                features.append(
                    data.detach().cpu().numpy().squeeze(0).transpose((1, 2, 0))
                )
                targets.append(target.detach().cpu().numpy().item())
                predictions.append(pred_label.detach().cpu().numpy().item())

    return features, targets, predictions


def show_misclassified(mis_features: list, samples: int):
    if samples % 5 == 0:
        plt.figure(figsize=(10, 10))  # specifying the overall grid size
        for i in range(samples):
            plt.subplot(
                int(samples // 5),
                int(samples / (samples // 5)),
                i + 1,
            )  # the number of images in the grid is 5*5 (25)
            plt.imshow(mis_features[i], interpolation="nearest")

        plt.show()


def find_mismatches(array1, array2):
    # Compare the arrays element-wise
    array1 = np.array(array1) if not isinstance(array1, np.ndarray) else array1
    array2 = np.array(array2) if not isinstance(array2, np.ndarray) else array2

    mismatches = array1 != array2

    # Get the indices where there are mismatches
    mismatch_indices = np.where(mismatches)

    return mismatch_indices[0]


def lightning_model_maker(
    torch_model,
    learning_rate,
    seed,
    mean,
    std,
    num_workers,
    batch_size,
    epochs,
    callbacks=None,
):
    # lightning wrapped model
    lightning_model = Cifar10ModelLightningBase(
        model=torch_model,
        learning_rate=learning_rate,
    )

    # lightning dataloder
    dm = Cifar10DataModule(
        seed=seed,
        mean=mean,
        std=std,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    # setting up the trainer module
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=CSVLogger(save_dir="./logs", name="Mnist_exp"),
        callbacks=callbacks,
    )

    return lightning_model, dm, trainer


def find_best_lr(trainer, lightning_model, dataloader, learning_rate, n_rounds=150):
    tuner = Tuner(trainer=trainer)
    lr_finder = tuner.lr_find(
        lightning_model,
        datamodule=dataloader,
        num_training=n_rounds,
        max_lr=learning_rate,
    )
    fig = lr_finder.plot(suggest=True)
    suggested_lr = lr_finder.suggestion()
    return suggested_lr
