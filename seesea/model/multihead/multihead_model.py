import torch
from torch import nn


class MultiHeadModel(nn.Module):
    def __init__(self, base_model, num_outputs):
        super().__init__()
        if num_outputs < 1:
            raise ValueError("num_outputs must be at least 1")

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

        self.heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(num_outputs)])

    def forward(self, pixel_values, labels):
        if labels is None:
            raise ValueError("labels cannot be None")

        features = self.base_model.base_model(pixel_values)[0]  # Shape: [batch_size, channels, height, width]

        if self.pool is not None:
            # Apply pooling and reshape
            features = self.pool(features)  # Shape: [batch_size, channels, 1, 1]
            features = features.flatten(1)  # Shape: [batch_size, channels]
            logits = torch.cat([head(features) for head in self.heads], dim=1)

        else:
            # Swin transformer
            # Feature Shape: [batch_size, num_patches, hidden_size]
            # Global Average Pooling: [batch_size, hidden_size]
            features = features.mean(dim=1)

        logits = torch.cat([head(features) for head in self.heads], dim=1)

        loss_fct = nn.MSELoss()
        loss = loss_fct(logits, labels.float())

        return loss, logits
