import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel, AutoModel, PretrainedConfig
import os
from enum import Enum, auto

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


class MultiHeadConfig(PretrainedConfig):
    """Configuration for the multi-head regression model"""

    model_type = "multihead_regression"

    def __init__(self, base_model_name=None, output_head_names=None, **kwargs):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.output_head_names = output_head_names if output_head_names is not None else []


class MultiHeadModel(PreTrainedModel):
    """Multi-head regression model"""

    config_class = MultiHeadConfig

    def __init__(self, config):
        super().__init__(config)

        # Store config
        self.config = config

        if len(self.config.output_head_names) < 1:
            raise ValueError("output_head_names must be at least 1")

        # Initialize base model
        self.backbone = AutoModel.from_pretrained(config.base_model_name)
        hidden_size = self.backbone.num_features  # Swin models use num_features instead of in_features

        # Create output heads
        self.output_heads = [OutputHead(name=name, input_size=hidden_size) for name in self.config.output_head_names]
        self.heads = nn.ModuleList([head for head in self.output_heads])

    def forward(self, pixel_values, labels=None):
        # Extract features from the base model
        features = self.backbone(pixel_values)[0]
        features = features.mean(dim=1)  # Global average pooling

        all_outputs = []
        for head in self.heads:
            head_output = head(features)
            all_outputs.append(head_output)

        # Concatenate the outputs from all heads
        logits = torch.cat(all_outputs, dim=1)

        loss = None
        if labels is not None:
            # Assume labels shape [batch_size, len(self.heads) + 1]
            full_label_mask = ~torch.isnan(labels)
            head_label_mask = full_label_mask[:, : len(self.heads)]
            base_direction = labels[:, -1]

            total_loss = 0.0
            valid_labels = 0

            for i, head in enumerate(self.heads):
                head_labels = labels[:, i]
                head_mask = head_label_mask[:, i]

                if head.type == OutputType.ANGLE:
                    effective_labels = (head_labels - base_direction) % 360
                else:
                    effective_labels = head_labels

                if head_mask.any():
                    head_output = all_outputs[i]
                    head_loss = head.loss_fn(head_output[head_mask], effective_labels[head_mask])
                    total_loss += head_loss
                    valid_labels += head_mask.sum()

            if valid_labels > 0:
                loss = total_loss / valid_labels

        # Return in HuggingFace format
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

    def get_output_head_names(self):
        """Returns list of output head names"""
        return self.config.output_head_names

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load from pretrained"""
        config = kwargs.pop("config", None)

        if config is None:
            raise ValueError("config must be provided")

        model = cls(config)

        # Load the model state dict
        model_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        state_dict = torch.load(model_path, weights_only=True)

        # Load the state dict
        model.load_state_dict(state_dict)

        return model

    def save_pretrained(self, save_directory, state_dict=None, safe_serialization=True, **kwargs):
        """Save the model to a directory"""
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Use provided state dict or get current one
        if state_dict is None:
            state_dict = self.state_dict()

        # Save the model state dict
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(state_dict, model_path)

        # Save the config
        self.config.save_pretrained(save_directory)
