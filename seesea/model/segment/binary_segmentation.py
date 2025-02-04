"""
Visualize the output of the fine-tuned model
"""

import os
import torch
import numpy as np
import torch.nn as nn
from transformers import AutoImageProcessor, AutoConfig, AutoModelForSemanticSegmentation
from datasets import load_dataset
import matplotlib.pyplot as plt
from PIL import Image

consider_as_sea = ["sea", "river", "lake"]
prevent_relabel_if_top = ["sky"]  # Categories that prevent relabeling if they're the top prediction
MIN_SEA_PERCENTAGE = 0.10  # 10% minimum sea requirement


def load_image_dataset(path, split, image_key, streaming=True):
    """Load dataset based on path format and return with consistent image access"""
    if os.path.isdir(path):
        # Check if it's a directory with images
        image_files = [f for f in os.listdir(path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if image_files:
            # Create a simple dataset from directory of images
            def image_dataset():
                for img_file in image_files:
                    img_path = os.path.join(path, img_file)
                    yield {"image": Image.open(img_path), "__key__": img_file}

            return image_dataset()

        # Try loading as webdataset
        try:
            return load_dataset("webdataset", data_dir=path, split=split, streaming=streaming).shuffle()
        except:
            pass

        # Try loading as a Hugging Face dataset directory
        try:
            return load_dataset(path, split=split, streaming=streaming).shuffle()
        except:
            pass

    # Try loading as a Hugging Face dataset name
    try:
        dataset = load_dataset(path, split=split, streaming=streaming).shuffle()
        return dataset
    except:
        raise ValueError(f"Could not load dataset from {path}")


def get_image_from_sample(sample, image_key):
    """Extract image from dataset sample using configured key"""
    if isinstance(sample, dict):
        # Try the provided key first
        if image_key in sample:
            return sample[image_key]

        # Try common image keys
        for key in ["image", "img", "jpg", "jpeg", "png", "image.jpg"]:
            if key in sample:
                return sample[key]

    # If sample is already an image
    if hasattr(sample, "convert") and callable(sample.convert):
        return sample

    raise ValueError(f"Could not find image in sample using key {image_key}")


def main(args):
    model = AutoModelForSemanticSegmentation.from_pretrained(args.model)
    image_processor = AutoImageProcessor.from_pretrained(args.model)
    dataset = load_image_dataset(args.dataset, args.split, args.image_key, streaming=args.streaming)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    config = AutoConfig.from_pretrained(args.model)
    id2label = config.id2label
    label2id = config.label2id

    # seed the random number generator
    np.random.seed(42)

    # create a map between the label and the color - convert to RGB by dropping alpha channel
    id_to_color = [plt.cm.rainbow(np.random.random())[:3] for _ in range(len(id2label))]
    palette = np.array(id_to_color)

    model.eval()
    for sample in dataset:
        try:
            image = get_image_from_sample(sample, args.image_key)
            name = sample.get("__key__", "unknown")
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue

        inputs = image_processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits.cpu()

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],  # PIL size is (width, height)
            mode="bilinear",
            align_corners=False,
        )

        # Compute original argmax segmentation mask and keep a copy for display
        original_pred_seg = upsampled_logits.argmax(dim=1)[0].clone()

        # Specify the number of top predictions to display
        n_top = 10
        # Compute top n predictions for each pixel. Shape: (n_top, height, width)
        top_preds = upsampled_logits.topk(n_top, dim=1)[1][0]

        # Apply softmax to logits to get probabilities
        probabilities = torch.nn.functional.softmax(upsampled_logits[0], dim=0)

        # Create an updated segmentation from the top n predictions
        pred_seg = original_pred_seg.clone()

        # --- Updated logic with confidence sum threshold ---
        CONFIDENCE_THRESHOLD = 0.01  # 1% threshold

        # Get top n probabilities and their indices
        top_probs, top_indices = probabilities.topk(n_top, dim=0)

        # Create mask for pixels where top prediction is in prevent_relabel_if_top list
        prevent_mask = torch.zeros_like(pred_seg, dtype=torch.bool)
        for label in prevent_relabel_if_top:
            if label in label2id:
                prevent_mask |= original_pred_seg == label2id[label]

        # Calculate sum of probabilities for sea-related classes within top n
        sea_confidence_sum = torch.zeros_like(pred_seg, dtype=torch.float)
        for label in consider_as_sea:
            if label in label2id:
                label_id = label2id[label]
                # Create a mask for where this label appears in top n
                label_in_top_n = top_indices == label_id
                # Add the corresponding probabilities where the label appears
                sea_confidence_sum += (top_probs * label_in_top_n).sum(dim=0)

        # Create mask where sum of sea-related confidences exceeds threshold
        # AND the pixel is not in prevent_relabel_if_top categories
        mask = (sea_confidence_sum > CONFIDENCE_THRESHOLD) & ~prevent_mask

        # Relabel any pixel where the criteria are met
        pred_seg[mask] = label2id["sea"]

        print(f"Number of pixels that will be relabeled: {mask.sum()}")

        # Determine image shape
        height, width = original_pred_seg.shape

        # --- Create overlay for the original argmax segmentation ---
        color_seg_original = np.zeros((height, width, 3), dtype=np.uint8)
        legend_elements_original = []
        for label_id, color in enumerate(palette):
            if (original_pred_seg == label_id).sum() > 0:
                color_seg_original[original_pred_seg == label_id] = (np.array(color[:3]) * 255).astype(np.uint8)
                legend_elements_original.append(plt.Rectangle((0, 0), 1, 1, fc=color, label=id2label[label_id]))

        # --- Create overlay for the top n relabeled segmentation ---
        color_seg_updated = np.zeros((height, width, 3), dtype=np.uint8)
        legend_elements_updated = []
        for label_id, color in enumerate(palette):
            if (pred_seg == label_id).sum() > 0:
                color_seg_updated[pred_seg == label_id] = (np.array(color[:3]) * 255).astype(np.uint8)
                legend_elements_updated.append(plt.Rectangle((0, 0), 1, 1, fc=color, label=id2label[label_id]))

        image_np = np.array(image)
        masked_image_original = (image_np * 0.5 + color_seg_original * 0.5).astype(np.uint8)
        masked_image_updated = (image_np * 0.5 + color_seg_updated * 0.5).astype(np.uint8)

        # Plot the overlays side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(30, 15))  # Changed to 2 subplots
        axes[0].imshow(masked_image_original)
        axes[0].set_title("Original Argmax Segmentation Overlay")
        axes[0].axis("off")
        axes[0].legend(
            handles=legend_elements_original,
            bbox_to_anchor=(0.5, -0.1),
            loc="upper center",
            ncol=len(legend_elements_original),
        )

        # Create binary mask
        binary_mask = torch.zeros_like(pred_seg, dtype=torch.bool)

        # Add areas that were originally sea-related
        for label in consider_as_sea:
            if label in label2id:
                binary_mask |= original_pred_seg == label2id[label]

        # Add areas that were relabeled to sea
        binary_mask |= mask  # mask from earlier contains the relabeled pixels

        # Create binary overlay
        color_seg_binary = np.zeros((height, width, 3), dtype=np.uint8)
        # Sea areas in blue
        color_seg_binary[binary_mask] = [0, 0, 255]  # Blue for sea
        # Non-sea areas in green
        color_seg_binary[~binary_mask] = [0, 255, 0]  # Green for non-sea

        # Create the binary overlay image
        masked_image_binary = (image_np * 0.7 + color_seg_binary * 0.3).astype(np.uint8)

        # Calculate sea percentage
        total_pixels = binary_mask.numel()
        sea_pixels = binary_mask.sum().item()
        sea_percentage = sea_pixels / total_pixels

        # Plot binary overlay with border based on sea percentage
        axes[1].imshow(masked_image_binary)
        axes[1].set_title(f"Binary Sea/Non-Sea Overlay\n{sea_percentage:.1%} sea")
        axes[1].axis("off")

        # Add legend for binary mask
        legend_elements_binary = [
            plt.Rectangle((0, 0), 1, 1, fc=(0, 0, 1), label="Sea"),
            plt.Rectangle((0, 0), 1, 1, fc=(0, 1, 0), label="Non-Sea"),
        ]
        axes[1].legend(
            handles=legend_elements_binary,
            bbox_to_anchor=(0.5, -0.1),
            loc="upper center",
            ncol=len(legend_elements_binary),
        )

        # ----- Begin interactive hover tooltip ----- #
        # Create an annotation object that will serve as the tooltip
        annotation = axes[0].annotate(
            "",
            xy=(0, 0),
            xytext=(20, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
        )
        annotation.set_visible(False)

        def hover(event):
            # Ensure the event is within the axes for the original overlay
            if event.inaxes == axes[0] and event.xdata is not None and event.ydata is not None:
                x, y = int(event.xdata), int(event.ydata)
                # Verify that the (x, y) coordinates fall inside the image bounds
                if 0 <= x < width and 0 <= y < height:
                    # Get top n predictions and their probabilities
                    probs, top_indices = probabilities[:, y, x].topk(n_top)
                    # Convert prediction IDs to labels and format with probabilities
                    top_labels_with_probs = [
                        f"{id2label.get(idx.item(), 'Unknown')}: {prob.item():.1%}"
                        for idx, prob in zip(top_indices, probs)
                    ]
                    annotation_text = f"Top {n_top} Predictions:\n" + "\n".join(top_labels_with_probs)
                    annotation.set_text(annotation_text)
                    annotation.xy = (x, y)
                    annotation.set_visible(True)
                    fig.canvas.draw_idle()
            else:
                # Hide the annotation if the cursor is not in the relevant axes
                if annotation.get_visible():
                    annotation.set_visible(False)
                    fig.canvas.draw_idle()

        # Connect the hover callback to the figure
        fig.canvas.mpl_connect("motion_notify_event", hover)
        # ----- End interactive hover tooltip ----- #

        plt.show()


def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize the output of the fine-tuned model")
    parser.add_argument("--model", type=str, help="Model to use", required=True)
    parser.add_argument("--dataset", type=str, help="Path or name of the dataset", required=True)
    parser.add_argument("--split", type=str, help="Split to use", default="test")
    parser.add_argument("--image-key", type=str, help="Key for accessing image in dataset", default="image")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode for dataset loading")
    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
