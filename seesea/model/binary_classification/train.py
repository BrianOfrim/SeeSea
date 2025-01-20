"""
Train a binary classification model to determin if a given image contains a body of water or not.
"""

import os
import shutil
import datasets
import argparse
from PIL import Image

import matplotlib.pyplot as plt
from transformers import AutoModelForImageClassification
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    dataset = datasets.load_dataset("Andron00e/Places365-custom", cache_dir=args.cache_dir)

    # Shuffle the dataset
    dataset = dataset["train"].shuffle(seed=42)

    label_names = dataset.features["labels"].names

    # make id to label and label to id
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {label: i for i, label in enumerate(label_names)}

    image_counters = {label: 0 for label in label_names}

    # go through the dataset and save the images to a folder according to their label

    os.makedirs(args.output_dir, exist_ok=True)
    for label in label_names:
        os.makedirs(os.path.join(args.output_dir, label), exist_ok=True)

    for example in tqdm(dataset):
        image = example["image"]
        label = example["labels"]

        assert label < len(label_names)
        label_name = id2label[label]

        # save the image to the folder according to its label, with the name being the image counter with 6 digits
        image_counter = image_counters[label_name]
        image_counter_str = f"{image_counter:06d}"
        image_path = os.path.join(args.output_dir, label_name, f"{image_counter_str}.jpg")
        image_counters[label_name] += 1
        # skip if the image already exists
        if os.path.exists(image_path):
            continue
        image.save(image_path)

    # # Split the dataset into train and validation based on percentage
    # train_dataset = dataset.select(range(int(len(dataset) * 0.8)))
    # val_dataset = dataset.select(range(int(len(dataset) * 0.2)))

    # print("Label names:", label_names)

    # model = AutoModelForImageClassification.from_pretrained("mobilenet_v2_1.0_224", id2label=id2label, label2id=label2id)

    # for example in dataset:
    #     print(example)
    #     # draw the image image
    #     plt.imshow(example["image"])
    #     plt.title(label_names[example["labels"]])
    #     plt.show()
