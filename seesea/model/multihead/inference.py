"""Inference script for the multihead model"""

import argparse
import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
from seesea.model.multihead.multihead_model import MultiHeadModel, MultiHeadConfig
import matplotlib.pyplot as plt


def main(args):
    """Main function for the inference script"""

    config = MultiHeadConfig.from_pretrained(args.model_dir)
    image_processor = AutoImageProcessor.from_pretrained(args.model_dir)
    model = MultiHeadModel.from_pretrained(args.model_dir, config=config)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    # Load image
    dataset = load_dataset("webdataset", data_dir=args.dataset, split=args.split, streaming=True).shuffle()

    for sample in dataset:
        image = sample["jpg"]
        labels = [sample["json"].get(name) for name in config.output_head_names]
        image_name = sample["__key__"]

        transformed_image = image_processor(image, return_tensors="pt")
        transformed_image = transformed_image["pixel_values"]
        transformed_image = transformed_image.to(device)

        # Create labels tensor with NaN for missing values
        label_tensor = torch.tensor([[float(l) if l is not None else float("nan") for l in labels]]).to(device)

        with torch.no_grad():
            outputs = model(transformed_image, label_tensor)
            outputs = outputs["logits"]  # Get logits from output dict

        outputs = outputs.squeeze().cpu().numpy()
        labels = label_tensor.squeeze().cpu().numpy()

        print(f"Image: {image_name}")
        for name, pred, target in zip(config.output_head_names, outputs, labels):
            print(f"  {name}: Predicted: {pred}, Expected: {target}, Diff: {pred - target}")

        if args.show_images:
            # show the image with the predicted and expected values
            plt.imshow(image)
            plt.title(f"Image: {image_name}")
            plt.axis("off")

            # Create subtitle with all predictions
            subtitle = ""
            for name, pred, target in zip(config.output_head_names, outputs, labels):
                subtitle += f"{name}: target={target:.3f}, pred={pred:.3f}, diff={pred - target:.3f}\n"

            plt.suptitle(subtitle)

            # Set the window title
            plt.gcf().canvas.manager.set_window_title(image_name)

            plt.show()


def get_args_parser():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")

    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--show-images", action="store_true")

    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
