"""
utils.py
--------
Utility functions for reproducibility, configuration handling,
and general project support for the CNN–LSTM Image Captioning system.

Author: Giuseppe Dimonte
University of Parma – MSc in Computer Engineering (AI curriculum)
"""

import os
import yaml
import json
import random
import torch
import numpy as np


# ===============================
#   Reproducibility
# ===============================
def set_seed(seed: int = 42):
    """
    Set global random seed to ensure reproducible experiments.

    Args:
        seed: integer value for RNG initialization.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic for exact reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Global random seed set to: {seed}")


# ===============================
#   Model utilities
# ===============================
def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a PyTorch model.

    Args:
        model: PyTorch model

    Returns:
        int: number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===============================
#   Vocabulary decoding
# ===============================
def decode_tokens(tokens, vocab):
    """
    Convert a sequence of token indices back into human-readable words.

    Args:
        tokens (list[int] or Tensor): sequence of token indices
        vocab (dict): vocabulary mapping word -> index

    Returns:
        str: decoded sentence as plain text
    """
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()

    inv_vocab = {idx: word for word, idx in vocab.items()}
    words = []
    for idx in tokens:
        word = inv_vocab.get(idx, "<unk>")
        if word in ["<bos>", "<pad>"]:
            continue
        if word == "<eos>":
            break
        words.append(word)
    return " ".join(words)


# ===============================
#   Configuration utilities
# ===============================
def save_config(config: dict, path: str = "configs/config_saved.yaml"):
    """
    Save configuration dictionary as YAML file.

    Args:
        config: dictionary containing experiment parameters
        path: output file path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)
    print(f"[INFO] Configuration saved to {path}")


def load_config(path: str):
    """
    Load configuration dictionary from YAML file.

    Args:
        path: YAML config file path

    Returns:
        dict: configuration parameters
    """
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


# ===============================
#   File and directory handling
# ===============================
def ensure_dir(path: str):
    """
    Ensure that a directory exists (create it if missing).
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[INFO] Created directory: {path}")


def save_json(data: dict, path: str):
    """
    Save a Python dictionary as a JSON file.

    Args:
        data: dictionary to save
        path: output file path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[INFO] JSON file saved to {path}")


def load_json(path: str) -> dict:
    """
    Load and parse a JSON file.

    Args:
        path: file path

    Returns:
        dict: loaded JSON content
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ===============================
#   Example main block
# ===============================
if __name__ == "__main__":
    """
    Example usage for debugging utilities.
    """
    set_seed(123)
    sample_vocab = {"<bos>": 0, "a": 1, "dog": 2, "runs": 3, "<eos>": 4}
    tokens = [0, 1, 2, 3, 4]
    print("Decoded:", decode_tokens(tokens, sample_vocab))

    cfg = {"train": {"epochs": 30, "batch_size": 64}}
    save_config(cfg, "configs/test_config.yaml")
    loaded = load_config("configs/test_config.yaml")
    print("Loaded config:", loaded)
