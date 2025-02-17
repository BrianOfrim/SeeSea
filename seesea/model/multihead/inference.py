"""Inference script for the multihead model"""

import argparse
import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
from seesea.model.multihead.multihead_model import MultiHeadModel, MultiHeadConfig


def main(args):
    """Main function for the inference script"""
    base_model_name = "microsoft/swin-tiny-patch4-window7-224"

    config = MultiHeadConfig(base_model_name=base_model_name, output_head_names=args.output_head_names)

    image_processor = AutoImageProcessor.from_pretrained(base_model_name)

    # Load model
    # model = MultiHeadModel(config)

    model = MultiHeadModel.from_pretrained(args.save_dir, config=config)

    # model.save_pretrained(args.save_dir)

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
        labels = [sample["json"].get(name) for name in args.output_head_names]
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
        for name, pred, target in zip(args.output_head_names, outputs, labels):
            print(f"  {name}: Predicted: {pred}, Expected: {target}, Diff: {pred - target}")


def get_args_parser():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output-head-names", type=str, nargs="+", required=True)
    parser.add_argument("--save-dir", type=str, required=True)

    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
