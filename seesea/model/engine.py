import os
import logging
import math
from typing import Callable, Tuple

import torch
from torch import nn
from tqdm import tqdm

import timm

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


def continuous_single_output_model_factory(model_name: str, weights_path: str = None) -> Tuple[nn.Module, Callable]:
    """
    Create a model with one continuous output from the model name and weights path

    Args:
        model_name: The name of the model to create
        weights_path: The path to the weights file to load into the model. If None, the default weights are used

    Returns:
        The model and an image transform function
    """

    model = timm.create_model(model_name, pretrained=True, num_classes=1)

    data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
    transform = timm.data.create_transform(**data_cfg)

    if weights_path is not None:
        # verify that the weights file exists
        if not os.path.exists(weights_path):
            raise ValueError(f"Model weights file {weights_path} does not exist")
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        LOGGER.debug("Loaded weights from %s", weights_path)

    return model, transform
