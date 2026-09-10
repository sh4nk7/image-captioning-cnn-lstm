"""
eval.py
-------
Evaluation script for the CNN-LSTM Image Captioning model on Flickr8k.
"""

import os
import json
import torch
import argparse
from PIL import Image
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm

from model import CNNtoLSTM
from dataset import get_transforms
from vocab_builder import load_vocabulary


def generate_caption(
    model: CNNtoLSTM,
    image_tensor: torch.Tensor,
    vocab: dict,
    max_len: int = 20,
    top_k: int = 1,
    beam_width: int = 3
) -> str:
    """Generate a caption using deterministic beam search."""
    model.eval()
    inv_vocab = {idx: word for word, idx in vocab.items()}
    eos_idx = vocab.get("<eos>")
    unk_idx = vocab.get("<unk>")
    blocked_ids = {vocab.get("<bos>"), vocab.get("<pad>"), unk_idx}

    with torch.no_grad():
        features = model.encoder(image_tensor)
        beams = [([], 0.0, None, features.unsqueeze(1), False)]

        for _ in range(max_len):
            candidates = []
            for tokens, score, states, inputs, finished in beams:
                if finished:
                    candidates.append((tokens, score, states, inputs, True))
                    continue

                lstm_out, next_states = model.decoder.lstm(inputs, states)
                logits = model.decoder.linear(lstm_out[:, -1, :])
                logits[:, list(blocked_ids - {None})] = -float("inf")
                if tokens:
                    logits[:, tokens[-1]] = -float("inf")

                log_probs = torch.log_softmax(logits, dim=-1)
                values, indices = torch.topk(log_probs, beam_width, dim=-1)
                for value, index in zip(values[0], indices[0]):
                    token_id = index.item()
                    ended = token_id == eos_idx
                    next_tokens = tokens if ended else tokens + [token_id]
                    next_input = model.decoder.embed(index.unsqueeze(0)).unsqueeze(1)
                    candidates.append((next_tokens, score + value.item(), next_states, next_input, ended))

            beams = sorted(
                candidates,
                key=lambda beam: beam[1] / max(len(beam[0]), 1) ** 0.7,
                reverse=True,
            )[:beam_width]
            if all(beam[4] for beam in beams):
                break

        output_indices = max(
            beams,
            key=lambda beam: beam[1] / max(len(beam[0]), 1) ** 0.7,
        )[0]

    caption_words = [
        inv_vocab[idx]
        for idx in output_indices
        if idx in inv_vocab and inv_vocab[idx] not in {"<unk>", "<bos>", "<pad>"}
    ]
    return " ".join(caption_words)


def compute_bleu(references, hypotheses):
    """Compute BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores."""
    smooth_fn = SmoothingFunction().method1
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smooth_fn)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth_fn)
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth_fn)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_fn)
    return {"BLEU-1": bleu1, "BLEU-2": bleu2, "BLEU-3": bleu3, "BLEU-4": bleu4}


def read_captions_map(file_path):
    """Read a JSON mapping from image file name to reference captions."""
    if not file_path.endswith(".json"):
        raise ValueError("Only JSON format is supported for reference captions.")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_model_from_checkpoint(weights_path: str, vocab_size: int, device: torch.device):
    """Load a model using hyperparameters stored in the checkpoint when available."""
    ckpt = torch.load(weights_path, map_location=device)
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})
    model = CNNtoLSTM(
        embed_size=model_cfg.get("embed_dim", 300),
        hidden_size=model_cfg.get("hidden_dim", 512),
        vocab_size=vocab_size,
        num_layers=model_cfg.get("num_layers", 1),
        dropout=model_cfg.get("dropout", 0.3),
        train_cnn=False,
    )
    model.load_state_dict(ckpt["model_state"])
    return model


def evaluate_model(
    model: CNNtoLSTM,
    data_dir: str,
    captions_json: str,
    vocab: dict,
    device: torch.device,
    output_path: str = "results/predictions.json",
    max_len: int = 20,
    top_k: int = 1
):
    """Evaluate the trained model on the Flickr8k test set."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    transform = get_transforms(img_size=299)

    model.to(device)
    model.eval()

    references = []
    hypotheses = []
    predictions = {}

    ref_data = read_captions_map(captions_json)

    print(f"Evaluating on {len(ref_data)} images...")
    for img_name, ref_caps in tqdm(ref_data.items(), desc="Generating captions"):
        img_path = os.path.join(data_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        pred_caption = generate_caption(model, image, vocab, max_len=max_len, top_k=top_k)
        predictions[img_name] = pred_caption

        hypotheses.append(pred_caption.split())
        references.append([ref.lower().split() for ref in ref_caps])

    bleu_scores = compute_bleu(references, hypotheses)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"bleu": bleu_scores, "predictions": predictions}, f, indent=2)

    print("\n--- BLEU Scores ---")
    for k, v in bleu_scores.items():
        print(f"{k}: {v:.4f}")

    return bleu_scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate CNN-LSTM model on Flickr8k.")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to test images.")
    parser.add_argument("--captions_json", type=str, required=True, help="JSON with reference captions.")
    parser.add_argument("--vocab_path", type=str, default="data/vocab.json", help="Vocabulary file.")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained model weights (.pt).")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--output", type=str, default="results/predictions.json", help="Output JSON path.")
    parser.add_argument("--max_len", type=int, default=20, help="Maximum caption length.")
    parser.add_argument("--top_k", type=int, default=1, help="Top-k sampling parameter.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    vocab = load_vocabulary(args.vocab_path)
    model = build_model_from_checkpoint(args.weights, vocab_size=len(vocab), device=device)

    bleu_scores = evaluate_model(
        model=model,
        data_dir=args.images_dir,
        captions_json=args.captions_json,
        vocab=vocab,
        device=device,
        output_path=args.output,
        max_len=args.max_len,
        top_k=args.top_k,
    )

    print("\nFinal BLEU Results:", bleu_scores)


if __name__ == "__main__":
    main()
