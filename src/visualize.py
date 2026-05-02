"""
visualize.py
------------
Visualization script for qualitative evaluation of the CNN-LSTM Image Captioning model.
"""

import os
import json
import random
import torch
import matplotlib.pyplot as plt
from PIL import Image

from model import CNNtoLSTM
from eval import generate_caption
from dataset import get_transforms
from vocab_builder import load_vocabulary


def visualize_predictions(
    model_path: str,
    vocab_path: str,
    data_dir: str,
    captions_json: str,
    output_dir: str = "figures/",
    num_samples: int = 6,
    device: str = "cuda"
):
    """Load a trained model and visualize several examples."""
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    vocab = load_vocabulary(vocab_path)
    ckpt = torch.load(model_path, map_location=device)
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})

    model = CNNtoLSTM(
        embed_size=model_cfg.get("embed_dim", 300),
        hidden_size=model_cfg.get("hidden_dim", 512),
        vocab_size=len(vocab),
        num_layers=model_cfg.get("num_layers", 1),
        dropout=model_cfg.get("dropout", 0.3),
        train_cnn=False,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    with open(captions_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    transform = get_transforms(img_size=299)
    num_samples = min(num_samples, len(data))
    sample_keys = random.sample(list(data.keys()), num_samples)

    print(f"Visualizing {num_samples} samples...")
    for idx, img_name in enumerate(sample_keys):
        img_path = os.path.join(data_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        predicted_caption = generate_caption(model, image_tensor, vocab, max_len=20, top_k=1)
        gt_caption = random.choice(data[img_name])

        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.axis("off")
        plt.title(
            f"Predicted: {predicted_caption}\nGround Truth: {gt_caption}",
            fontsize=9, wrap=True, loc="center"
        )
        save_path = os.path.join(output_dir, f"vis_{idx + 1}_{img_name.replace('.jpg', '.png')}")
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()

        print(f"Saved visualization: {save_path}")

    print(f"\nAll visualizations saved in '{output_dir}'.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize CNN-LSTM caption predictions.")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained model weights.")
    parser.add_argument("--vocab_path", type=str, default="data/vocab.json", help="Path to vocabulary JSON.")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory with test images.")
    parser.add_argument("--captions_json", type=str, required=True, help="JSON with reference captions.")
    parser.add_argument("--output_dir", type=str, default="figures/", help="Where to save visualizations.")
    parser.add_argument("--num_samples", type=int, default=6, help="Number of samples to visualize.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    visualize_predictions(
        model_path=args.weights,
        vocab_path=args.vocab_path,
        data_dir=args.images_dir,
        captions_json=args.captions_json,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        device=args.device,
    )
