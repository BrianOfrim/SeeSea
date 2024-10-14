import logging
import math
from typing import Callable

import torch
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


def collate(samples):
    """Collate the samples into a batch"""
    images = [s["image"] for s in samples]
    labels = [s["label"] for s in samples]

    return torch.stack(images), torch.tensor(labels)


def preprocess(transform: Callable, label_key: str, sample):
    """Pereprocess the webdataset sample"""
    return {"image": transform(sample["jpg"]), "label": sample["json"][label_key]}


def train_one_epoch(model, criterion, optimizer, loader, device, scheduler=None):
    """Train the model for one epoch"""
    model.train()
    running_loss = 0.0
    inputs_processed = 0

    for inputs, label in tqdm(loader, leave=False, desc="Training", disable=LOGGER.level > logging.INFO):
        # pring the percentage of the dataset that has been processed

        inputs = inputs.to(device)
        label = label.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        outputs = outputs.view(-1)  # Flatten outputs to match wind_speeds shape
        loss = criterion(outputs, label.view(-1))

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * inputs.size(0)
        inputs_processed += inputs.size(0)

    return running_loss / inputs_processed


@torch.no_grad()
def evaluate_model(model, criterion, loader, device):
    """Evaluate the model"""
    model.eval()
    running_loss = 0.0
    inputs_processed = 0

    for inputs, label in tqdm(loader, leave=False, desc="Eval", disable=LOGGER.level > logging.INFO):
        inputs = inputs.to(device)
        label = label.to(device)

        outputs = model(inputs)
        outputs = outputs.view(-1)
        loss = criterion(outputs, label.view(-1))

        running_loss += loss.item() * inputs.size(0)
        inputs_processed += inputs.size(0)

    if inputs_processed == 0:
        return math.inf

    return running_loss / inputs_processed
