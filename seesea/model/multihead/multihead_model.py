import torch
from torch import nn
from torch.nn import functional as F

from enum import Enum, auto
from dataclasses import dataclass
from typing import List

SIN_INDEX = 0
COS_INDEX = 1


class OutputType(Enum):
    """Types of outputs supported by the model"""

    LINEAR = auto()  # Standard linear regression output
    ANGLE = auto()  # Angular output (e.g. for heading prediction)


def get_head_type(name: str) -> OutputType:
    # if the name ends with "_deg" then it's an angle head
    if name.endswith("_deg"):
        return OutputType.ANGLE
    else:
        return OutputType.LINEAR


def get_output_size(type: OutputType):
    if type == OutputType.LINEAR:
        return 1
    elif type == OutputType.ANGLE:
        return 2
    else:
        raise ValueError(f"Unsupported output type: {type}")


class OutputHead(nn.Module):
    def __init__(self, name: str, input_size: int):
        super().__init__()
        self.name = name
        self.type = get_head_type(name)
        self.output_layer = nn.Linear(input_size, get_output_size(self.type))

    def get_output_size(self):
        if self.type == OutputType.LINEAR:
            return 1
        elif self.type == OutputType.ANGLE:
            return 2
        else:
            raise ValueError(f"Unsupported output type: {self.type}")

    def forward(self, x):
        return self.output_layer(x)

    def loss_fn(self, logits, labels):
        if self.type == OutputType.LINEAR:
            return F.mse_loss(logits.squeeze(-1), labels)
        elif self.type == OutputType.ANGLE:
            # For angle outputs, logits has shape [batch_size, 2] (sin and cos components)
            # labels has shape [batch_size]
            # Need to convert labels to sin/cos components
            angle_rad = torch.deg2rad(labels)
            target_sin = torch.sin(angle_rad)
            target_cos = torch.cos(angle_rad)

            sin_loss = F.mse_loss(logits[:, SIN_INDEX], target_sin)
            cos_loss = F.mse_loss(logits[:, COS_INDEX], target_cos)
            return sin_loss + cos_loss
        else:
            raise ValueError(f"Unsupported output type: {self.type}")


class MultiHeadModel(nn.Module):
    def __init__(self, base_model, output_head_names: List[str]):
        super().__init__()
        if len(output_head_names) < 1:
            raise ValueError("output_head_names must be at least 1")

        self.base_model = base_model

        # Add global average pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        if hasattr(base_model, "config"):
            # Added this for Swin Transformer
            hidden_size = base_model.config.hidden_size
            self.base_model.classifier = nn.Identity()
            self.pool = None  # Not needed for Swin Transformer

        # Handle both ResNet (fc) and other models (classifier)
        elif hasattr(base_model, "fc"):
            hidden_size = base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        elif hasattr(base_model, "classifier"):
            if hasattr(base_model.classifier, "in_features"):
                hidden_size = base_model.classifier.in_features
            else:
                # For sequential classifiers, get the input features of the last linear layer
                for layer in reversed(base_model.classifier):
                    if isinstance(layer, nn.Linear):
                        hidden_size = layer.in_features
                        break
            self.base_model.classifier = nn.Identity()
        else:
            raise ValueError("Model must have either 'fc' or 'classifier' as final layer")

        self.output_heads = [OutputHead(name=name, input_size=hidden_size) for name in output_head_names]
        self.heads = nn.ModuleList([head for head in self.output_heads])

    def forward(self, pixel_values, labels):
        if labels is None:
            raise ValueError("labels cannot be None")

        # Assume labels shape [batch_size, len(self.heads) + 1]
        # Columns 0 to len(self.heads)-1 hold the absolute labels for each head,
        # and the last column (labels[:, -1]) holds the absolute image direction.
        full_label_mask = ~torch.isnan(labels)
        # Exclude the image_direction_deg (last column) from the loss computation mask:
        head_label_mask = full_label_mask[:, : len(self.heads)]

        # Extract base image direction from the last column.
        base_direction = labels[:, -1]

        # Extract features from the base model.
        features = self.base_model.base_model(pixel_values)[0]  # Shape: [batch_size, channels, height, width]
        if self.pool is not None:
            features = self.pool(features)  # Shape: [batch_size, channels, 1, 1]
            features = features.flatten(1)  # Shape: [batch_size, channels]
        else:
            features = features.mean(dim=1)

        all_outputs = []
        total_loss = 0.0
        valid_labels = 0  # Count number of heads with valid labels

        # Iterate over heads. Their corresponding absolute labels for the i-th head are in columns 0 to len(self.heads)-1.
        for i, head in enumerate(self.heads):
            head_output = head(features)
            all_outputs.append(head_output)

            # Get the absolute label for this head.
            head_labels = labels[:, i]
            head_mask = head_label_mask[:, i]

            if head.type == OutputType.ANGLE:
                # Subtract base direction and normalize to [0, 360)
                effective_labels = (head_labels - base_direction) % 360
            else:
                effective_labels = head_labels

            if head_mask.any():
                head_loss = head.loss_fn(head_output[head_mask], effective_labels[head_mask])
                total_loss += head_loss
                valid_labels += head_mask.sum()

        # Concatenate the outputs from all heads.
        logits = torch.cat(all_outputs, dim=1)

        if valid_labels > 0:
            # average the loss based on how many outputs contributed to it
            total_loss = total_loss / valid_labels

        return total_loss, logits
