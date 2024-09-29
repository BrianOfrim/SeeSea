"Training script for the SeeSea model."

import os
import logging
import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import seesea.utils as utils
from seesea.observation import Observation, ImageObservation

LOGGER = logging.getLogger(__name__)


# Custom Dataset Class
class WindspeedDataset(Dataset):
    def __init__(self, image_observation_file: str, transform=None, max_length=None):
        """
        Args:
            image_observation_file (string): Path to the JSON file with image filenames and observation data.
            transform (callable, optional): Optional transforms to be applied to the images.
        """
        image_observations_json = utils.load_json(image_observation_file)
        if image_observations_json is None:
            # throw an exception if the json file is not loaded
            raise Exception(f"Failed to load image observation data from {image_observation_file}")
        if len(image_observations_json) == 0:
            # throw an exception if the json file is empty
            raise Exception(f"No image observation data found in {image_observation_file}")

        self.image_observations = [utils.from_dict(ImageObservation, obs) for obs in image_observations_json]
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        size = len(self.image_observations)
        if self.max_length is not None:
            size = min(size, self.max_length)
        return size

    def __getitem__(self, idx):
        image_path = self.image_observations[idx].image_path
        image = Image.open(image_path).convert("RGB")
        observation = self.image_observations[idx].observation

        wind_speed_mps = observation.wind_speed_mps
        wind_speed_mps = np.array([wind_speed_mps]).astype("float32")
        wind_speed_mps = torch.from_numpy(wind_speed_mps)

        if self.transform:
            image = self.transform(image)

        return image, wind_speed_mps


data_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # Normalize with ImageNet mean and std
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def load_model(model_name):
    """Dynamically get the model class from torchvision.models using getattr"""
    model_class = getattr(models, model_name)

    # Load the model with the default pre-trained weights
    model = model_class(weights="DEFAULT")

    return model


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(description="Train the SeeSea model")
    arg_parser.add_argument("--input", help="The directory containing the training data", default="data")
    arg_parser.add_argument("--output", help="The directory to write the output files to", default="data/train")
    arg_parser.add_argument("--log", type=str, help="Log level", default="INFO")
    arg_parser.add_argument("--log-file", type=str, help="Log file", default=None)
    arg_parser.add_argument("--epochs", type=int, help="The number of epochs to train for", default=30)
    arg_parser.add_argument("--batch-size", type=int, help="The batch size to use for training", default=32)
    arg_parser.add_argument("--learning-rate", type=float, help="The learning rate to use for training", default=0.001)
    arg_parser.add_argument("--model", type=str, help="The model to use for training", default="resnet18")
    arg_parser.add_argument("--model-path", type=str, help="The path to save the trained model", default="model.pth")
    input_args = arg_parser.parse_args()

    # setup the loggers
    LOGGER.setLevel(input_args.log)

    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_logging_handler = logging.StreamHandler()
    console_logging_handler.setFormatter(log_formatter)
    LOGGER.addHandler(console_logging_handler)

    if input_args.log_file is not None:
        file_logging_handler = logging.FileHandler(input_args.log_file)
        file_logging_handler.setFormatter(log_formatter)
        LOGGER.addHandler(file_logging_handler)

    # train the model
    LOGGER.info("Training the model")

    if not os.path.exists(input_args.output):
        os.makedirs(input_args.output)

    train_file = os.path.join(input_args.input, "train.json")
    val_file = os.path.join(input_args.input, "val.json")

    train_dataset = WindspeedDataset(train_file, transform=data_transforms)
    val_dataset = WindspeedDataset(val_file, transform=data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=input_args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=input_args.batch_size, shuffle=False, num_workers=4)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = load_model(input_args.model)

    # Modify the fully connected layer to output a single value
    # The original fully connected layer has model.fc = nn.Linear(2048, 1000)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=1)

    model = model.to(device)

    # Loss Function and Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=input_args.learning_rate)

    training_start_time = datetime.datetime.now(tz=datetime.timezone.utc)
    # Training Loop

    train_losses = []
    val_losses = []

    for epoch in range(input_args.epochs):
        model.train()
        running_loss = 0.0

        for inputs, wind_speeds in train_loader:
            inputs = inputs.to(device)
            wind_speeds = wind_speeds.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            outputs = outputs.view(-1)  # Flatten outputs to match wind_speeds shape
            loss = criterion(outputs, wind_speeds.view(-1))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_loss)
        LOGGER.info("Epoch %d/%d, Training Loss: %.4f", epoch + 1, input_args.epochs, epoch_loss)

        # Validation Phase
        model.eval()
        val_running_loss = 0.0

        with torch.no_grad():
            for inputs, wind_speeds in val_loader:
                inputs = inputs.to(device)
                wind_speeds = wind_speeds.to(device)

                outputs = model(inputs)
                outputs = outputs.view(-1)
                loss = criterion(outputs, wind_speeds.view(-1))

                val_running_loss += loss.item() * inputs.size(0)

        val_epoch_loss = val_running_loss / len(val_dataset)
        val_losses.append(val_epoch_loss)
        LOGGER.info("Epoch %d/%d, Validation Loss: %.4f", epoch + 1, input_args.epochs, val_epoch_loss)

    training_end_time = datetime.datetime.now(tz=datetime.timezone.utc)
    LOGGER.info("Training complete. Training time: %s", training_end_time - training_start_time)

    # Save the model
    timestamp_str = training_start_time.strftime("%Y_%m_%d_%H%M")
    model_dir = os.path.join(input_args.output, timestamp_str)

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    model_filepath = os.path.join(model_dir, "model.pth")
    torch.save(model.state_dict(), model_filepath)
    LOGGER.info("Model saved to %s", model_filepath)

    # Plot the training and validation loss
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    loss_plot_filepath = os.path.join(model_dir, "loss_plot.png")
    plt.savefig(loss_plot_filepath)

    LOGGER.info("Loss plot saved to %s", loss_plot_filepath)
