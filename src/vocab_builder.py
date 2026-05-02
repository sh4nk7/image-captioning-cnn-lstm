"""
vocab_builder.py
----------------
Utilities to clean Flickr8k captions, build a vocabulary, and export
processed train/validation files used by the project.
"""

import os
import re
import csv
import json
import argparse
from collections import Counter


def clean_caption(caption: str) -> str:
    """
    Normalize and clean a single caption string.

    - Lowercase all text
    - Remove punctuation and digits
    - Remove single-character words
    - Strip extra whitespace
    """
    caption = caption.lower()
    caption = re.sub(r"[^a-z\s]", "", caption)
    caption = re.sub(r"\b[a-z]\b", "", caption)
    caption = re.sub(r"\s+", " ", caption)
    return caption.strip()


def parse_caption_line(line: str):
    """
    Parse a Flickr8k caption line.

    Supports both:
    - image.jpg#0<TAB>caption
    - image.jpg,caption
    """
    line = line.strip()
    if not line:
        return None, None

    if "\t" in line:
        image_id, caption = line.split("\t", 1)
        return image_id.split("#")[0], caption

    if "," in line:
        image_id, caption = line.split(",", 1)
        return image_id.split("#")[0], caption

    return None, None


def read_caption_pairs(captions_file: str, clean: bool = True):
    """Read `(image_name, caption)` pairs from a Flickr8k caption file."""
    pairs = []
    with open(captions_file, "r", encoding="utf-8") as f:
        for line in f:
            image_name, caption = parse_caption_line(line)
            if image_name is None:
                continue
            caption = clean_caption(caption) if clean else caption.strip()
            if caption:
                pairs.append((image_name, caption))
    return pairs


def read_split_list(split_file: str):
    """Read a Flickr8k split file containing one image name per line."""
    with open(split_file, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def build_vocabulary(captions_file: str, freq_threshold: int = 5):
    """
    Build a vocabulary dictionary from a caption text file.
    """
    pairs = read_caption_pairs(captions_file, clean=True)
    print(f"[INFO] Processing {len(pairs)} captions...")

    all_words = []
    for _, caption in pairs:
        all_words.extend(caption.split())

    freq = Counter(all_words)
    words = [w for w, c in freq.items() if c >= freq_threshold]

    print(f"[INFO] Found {len(words)} words with freq >= {freq_threshold}")

    vocab = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
    }
    for i, word in enumerate(sorted(words), start=4):
        vocab[word] = i

    return vocab


def save_vocabulary(vocab: dict, path: str = "data/vocab.json"):
    """Save vocabulary dictionary to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Vocabulary saved to {path}")


def load_vocabulary(path: str) -> dict:
    """Load vocabulary dictionary from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def export_dataset_files(
    captions_file: str,
    train_split: str,
    val_split: str,
    test_split: str,
    train_out: str,
    val_out: str,
    test_json_out: str,
):
    """
    Export processed training/validation CSV files and a JSON mapping for test evaluation.
    """
    pairs = read_caption_pairs(captions_file, clean=True)
    grouped = {}
    for image_name, caption in pairs:
        grouped.setdefault(image_name, []).append(caption)

    train_images = read_split_list(train_split)
    val_images = read_split_list(val_split)
    test_images = read_split_list(test_split)

    os.makedirs(os.path.dirname(train_out), exist_ok=True)
    os.makedirs(os.path.dirname(val_out), exist_ok=True)
    os.makedirs(os.path.dirname(test_json_out), exist_ok=True)

    with open(train_out, "w", encoding="utf-8", newline="") as f_train:
        writer = csv.writer(f_train)
        writer.writerow(["image_name", "caption"])
        for image_name in sorted(train_images):
            for caption in grouped.get(image_name, []):
                writer.writerow([image_name, caption])

    with open(val_out, "w", encoding="utf-8", newline="") as f_val:
        writer = csv.writer(f_val)
        writer.writerow(["image_name", "caption"])
        for image_name in sorted(val_images):
            for caption in grouped.get(image_name, []):
                writer.writerow([image_name, caption])

    test_mapping = {
        image_name: grouped.get(image_name, [])
        for image_name in sorted(test_images)
        if grouped.get(image_name)
    }
    with open(test_json_out, "w", encoding="utf-8") as f_test:
        json.dump(test_mapping, f_test, indent=2, ensure_ascii=False)

    print(f"[INFO] Wrote train split to {train_out}")
    print(f"[INFO] Wrote val split to {val_out}")
    print(f"[INFO] Wrote test captions JSON to {test_json_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build vocabulary for Flickr8k captions.")
    parser.add_argument("--captions_file", type=str, required=True, help="Path to captions file (CSV/TXT).")
    parser.add_argument("--freq_threshold", type=int, default=5, help="Min frequency for word inclusion.")
    parser.add_argument("--output", type=str, default="data/vocab.json", help="Output JSON path.")
    parser.add_argument("--train_split", type=str, default="", help="Optional Flickr8k train split file.")
    parser.add_argument("--val_split", type=str, default="", help="Optional Flickr8k validation split file.")
    parser.add_argument("--test_split", type=str, default="", help="Optional Flickr8k test split file.")
    parser.add_argument("--train_out", type=str, default="data/Flickr8k_text/train.csv", help="Processed train CSV output.")
    parser.add_argument("--val_out", type=str, default="data/Flickr8k_text/val.csv", help="Processed validation CSV output.")
    parser.add_argument("--test_json_out", type=str, default="data/Flickr8k_text/test_captions.json", help="Processed test captions JSON output.")
    args = parser.parse_args()

    vocab = build_vocabulary(args.captions_file, args.freq_threshold)
    save_vocabulary(vocab, args.output)

    if args.train_split and args.val_split and args.test_split:
        export_dataset_files(
            captions_file=args.captions_file,
            train_split=args.train_split,
            val_split=args.val_split,
            test_split=args.test_split,
            train_out=args.train_out,
            val_out=args.val_out,
            test_json_out=args.test_json_out,
        )

    print(f"[INFO] Vocabulary size: {len(vocab)} words")
