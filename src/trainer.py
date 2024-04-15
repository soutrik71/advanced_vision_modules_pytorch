# updated training module---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, OneCycleLR, StepLR
from torch_lr_finder import LRFinder
from torcheval.metrics import BinaryAccuracy, MulticlassAccuracy
from tqdm import tqdm

from src.helper import EarlyStopping


def train_module(
    model: torch.nn.Module,
    device: torch.device,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    metric,
    train_losses: list,
    train_metrics: list,
):

    # setting model to train mode
    model.train()
    pbar = tqdm(iterable=train_dataloader, desc="training", colour="green")

    # batch metrics
    train_loss = 0
    train_metric = 0
    processed_batch = 0

    for idx, (data, label) in enumerate(pbar):
        # setting up device
        data = data.to(device)
        label = label.to(device)

        # forward pass output
        preds = model(data)

        # calc loss
        loss = criterion(preds, label)
        train_loss += loss.item()
        # print(f"training loss for batch {idx} is {loss}")

        # backpropagation
        optimizer.zero_grad()  # flush out  existing grads
        loss.backward()  # back prop of weights wrt loss
        optimizer.step()  # optimizer step -> minima

        # metric calc
        preds = torch.argmax(preds, dim=1)
        # print(f"preds:: {preds}")
        metric.update(preds, label)
        train_metric += metric.compute().detach().item()

        # updating batch count
        processed_batch += 1

        pbar.set_description(
            f"Avg Train Loss: {train_loss/processed_batch} Avg Train Metric: {train_metric/processed_batch}"
        )

    # It's typically called after the epoch completes
    metric.reset()
    # updating epoch metrics
    train_losses.append(train_loss / processed_batch)
    train_metrics.append(train_metric / processed_batch)

    return train_losses, train_metrics


# updated testing modules---
def test_module(
    model: torch.nn.Module,
    device: torch.device,
    test_dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    metric,
    test_losses,
    test_metrics,
):
    # setting model to eval mode
    model.eval()
    pbar = tqdm(test_dataloader, desc="testing", colour="red")

    # batch metrics
    test_loss = 0
    test_metric = 0
    processed_batch = 0

    with torch.inference_mode():
        for idx, (data, label) in enumerate(pbar):
            data, label = data.to(device), label.to(device)
            # predictions
            preds = model(data)
            # print(preds.shape)
            # print(label.shape)

            # loss calc
            loss = criterion(preds, label)
            test_loss += loss.item()

            # metric calc
            preds = torch.argmax(preds, dim=1)
            metric.update(preds, label)
            test_metric += metric.compute().detach().item()

            # updating batch count
            processed_batch += 1

            pbar.set_description(
                f"Avg Test Loss: {test_loss/processed_batch} Avg Test Metric: {test_metric/processed_batch}"
            )

        # It's typically called after the epoch completes
        metric.reset()
        # updating epoch metrics
        test_losses.append(test_loss / processed_batch)
        test_metrics.append(test_metric / processed_batch)

    return test_losses, test_metrics


def model_drivers(
    model: nn.Module,
    learning_rate: float,
    num_classes: int,
    model_name: str,
    device: torch.device,
    optimizer_type: str = "adam",
):
    """Initialize drivers for training"""
    # optmizer
    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4
        )
    # loss
    criterion = nn.CrossEntropyLoss()
    # metric
    metric = MulticlassAccuracy(device=device, num_classes=num_classes)
    # Early stopping
    early_stopping = EarlyStopping(patience=5, verbose=True, model_name=model_name)

    return optimizer, criterion, metric, early_stopping


def suggested_lr(model, optimizer, criterion, device, end_lr, num_iter, train_loader):
    """Suggested learning rate"""
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(
        train_loader, end_lr=end_lr, num_iter=num_iter, step_mode="exp"
    )
    lr_finder.plot()
    lr_finder.reset()


def get_lr_scheduler(type, *args, **kwargs):
    if type == "StepLR":
        return StepLR(*args, **kwargs)
    elif type == "ExponentialLR":
        return ExponentialLR(*args, **kwargs)

    elif type == "CyclicLR":
        return OneCycleLR(*args, **kwargs)
