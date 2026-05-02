"""
dataset.py
----------
Dataset management and preprocessing for the Flickr8k Image Captioning project.
"""

import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from vocab_builder import clean_caption


class FlickrDataset(Dataset):
    """Custom PyTorch Dataset for the Flickr8k image captioning task."""

    def __init__(self, root_dir, captions_file, vocab, transform=None):
        self.root_dir = root_dir

        ext = os.path.splitext(captions_file)[1].lower()
        if ext == ".csv":
            self.df = pd.read_csv(captions_file)
            expected_cols = {"image_name", "caption"}
            if not expected_cols.issubset(self.df.columns):
                raise ValueError(f"CSV file must contain columns {expected_cols}.")
        else:
            self.df = pd.read_csv(
                captions_file,
                sep="\t",
                header=None,
                names=["image_caption", "caption"],
                engine="python"
            )
            self.df["image_name"] = self.df["image_caption"].astype(str).str.split("#").str[0]
            self.df = self.df[["image_name", "caption"]]

        self.df["caption"] = self.df["caption"].fillna("").astype(str).map(clean_caption)
        self.df = self.df[self.df["caption"].str.len() > 0].reset_index(drop=True)

        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = row["image_name"]
        caption = row["caption"]

        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        tokens = [self.vocab["<bos>"]]
        tokens += [self.vocab.get(word, self.vocab["<unk>"]) for word in caption.split()]
        tokens.append(self.vocab["<eos>"])

        caption_tensor = torch.tensor(tokens, dtype=torch.long)
        return image, caption_tensor


def get_transforms(img_size=224):
    """Return standard image transformations for CNN-based feature extractors."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    images, captions = zip(*batch)

    images = torch.stack(images, dim=0)

    lengths = [len(c) for c in captions]
    max_len = max(lengths)

    padded_captions = torch.full((len(captions), max_len), fill_value=0, dtype=torch.long)
    for i, cap in enumerate(captions):
        padded_captions[i, :lengths[i]] = cap

    return images, padded_captions, lengths


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Test FlickrDataset loading.")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to image directory.")
    parser.add_argument("--captions_file", type=str, required=True, help="Path to captions file.")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to vocabulary JSON file.")
    args = parser.parse_args()

    with open(args.vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    transform = get_transforms()
    dataset = FlickrDataset(args.root_dir, args.captions_file, vocab, transform)

    img, caption = dataset[0]
    print("Image tensor shape:", img.shape)
    print("Caption indices:", caption.tolist())
