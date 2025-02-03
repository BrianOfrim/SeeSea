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

label_as_sea = ["earth", "mountain", "sand", "water"]


def main(args):
    # model = torch.load(os.path.join(args.model_dir, "model.pt"))
    model = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    dataset = load_dataset("webdataset", data_dir=args.dataset, split=args.split, streaming=True).shuffle()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    config = AutoConfig.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    id2label = config.id2label
    label2id = config.label2id

    # seed the random number generator
    np.random.seed(42)

    # create a map between the label and the color - convert to RGB by dropping alpha channel
    id_to_color = [plt.cm.rainbow(np.random.random())[:3] for _ in range(len(id2label))]
    palette = np.array(id_to_color)

    model.eval()
    for sample in dataset:
        image = sample["image.jpg"]
        label_mask = sample["mask.png"]
        name = sample["__key__"]

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

        # Create an updated segmentation from the top n predictions
        pred_seg = original_pred_seg.clone()

        # Create a boolean mask for pixels with a label in 'label_as_sea'
        mask_from_labels = torch.zeros_like(pred_seg, dtype=torch.bool)
        for label in label_as_sea:
            mask_from_labels |= pred_seg == label2id[label]

        # Check if 'sea' is among the top n predictions
        sea_in_top_n = (top_preds == label2id["sea"]).any(dim=0)

        print(f"Number of pixels that will be relabeled: {mask_from_labels.sum()}")

        # Relabel: where label belongs to label_as_sea and has 'sea' in its top n predictions.
        mask = mask_from_labels & sea_in_top_n
        pred_seg[mask] = label2id["sea"]

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

        # Plot the two overlays side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(30, 15))
        axes[0].imshow(masked_image_original)
        axes[0].set_title("Original Argmax Segmentation Overlay")
        axes[0].axis("off")
        axes[0].legend(
            handles=legend_elements_original,
            bbox_to_anchor=(0.5, -0.1),
            loc="upper center",
            ncol=len(legend_elements_original),
        )

        axes[1].imshow(masked_image_updated)
        axes[1].set_title("Top n-Relabeled Segmentation Overlay")
        axes[1].axis("off")
        axes[1].legend(
            handles=legend_elements_updated,
            bbox_to_anchor=(0.5, -0.1),
            loc="upper center",
            ncol=len(legend_elements_updated),
        )

        # ----- Begin interactive hover tooltip below ----- #
        # Create an annotation object that will serve as the tooltip.
        annotation = axes[0].annotate(
            "",
            xy=(0, 0),
            xytext=(20, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
        )
        annotation.set_visible(False)

        # This callback updates the annotation with the top n labels at the cursor location.
        def hover(event):
            # Ensure the event is within the axes for the original overlay.
            if event.inaxes == axes[0] and event.xdata is not None and event.ydata is not None:
                x, y = int(event.xdata), int(event.ydata)
                # Verify that the (x, y) coordinates fall inside the image bounds.
                if 0 <= x < width and 0 <= y < height:
                    # Access the top n predictions for the pixel (note indexing: [n_top, row, col]).
                    top_ids = top_preds[:, y, x].tolist()
                    # Convert prediction IDs to labels using integer keys.
                    top_labels = [id2label.get(t, "Unknown") for t in top_ids]
                    annotation_text = f"Top {n_top} Predictions:\n" + "\n".join(top_labels)
                    annotation.set_text(annotation_text)
                    annotation.xy = (x, y)
                    annotation.set_visible(True)
                    fig.canvas.draw_idle()
            else:
                # Hide the annotation if the cursor is not in the relevant axes.
                if annotation.get_visible():
                    annotation.set_visible(False)
                    fig.canvas.draw_idle()

        # Connect the hover callback to the figure.
        fig.canvas.mpl_connect("motion_notify_event", hover)
        # ----- End interactive hover tooltip ----- #

        plt.show()


def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize the output of the fine-tuned model")
    parser.add_argument("--model-dir", type=str, help="Path to the model directory", required=True)
    parser.add_argument("--dataset", type=str, help="Path to the dataset", required=True)
    parser.add_argument("--split", type=str, help="Split to use", default="test")
    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
