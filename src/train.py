"""
train.py
--------
Full training script for the CNN-LSTM Image Captioning model on Flickr8k.
"""

import os
import math
import time
import yaml
import random
import argparse
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import FlickrDataset, get_transforms, collate_fn
from model import CNNtoLSTM
from vocab_builder import load_vocabulary


def set_seed(seed: int = 42):
    """Set global random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def decoder_forward_scheduled(
    decoder: nn.Module,
    features: torch.Tensor,
    captions: torch.Tensor,
    teacher_forcing_ratio: float,
) -> torch.Tensor:
    """
    Run the decoder step-by-step to apply scheduled sampling.
    """
    batch_size, seq_len = captions.size()
    device = captions.device

    outputs = []
    states = None

    inputs = features.unsqueeze(1)
    lstm_out, states = decoder.lstm(decoder.dropout(inputs), states)
    step_logits = decoder.linear(lstm_out.squeeze(1))
    outputs.append(step_logits.unsqueeze(1))

    prev_token = captions[:, 0]
    pred_token = prev_token

    for t in range(1, seq_len):
        use_teacher = (torch.rand(batch_size, device=device) < teacher_forcing_ratio)
        if t == 1:
            use_teacher = torch.ones_like(use_teacher, dtype=torch.bool, device=device)

        inp_token = prev_token if t == 1 else torch.where(use_teacher, captions[:, t - 1], pred_token)

        emb = decoder.embed(inp_token).unsqueeze(1)
        lstm_out, states = decoder.lstm(decoder.dropout(emb), states)
        step_logits = decoder.linear(lstm_out.squeeze(1))
        outputs.append(step_logits.unsqueeze(1))
        pred_token = step_logits.argmax(dim=-1)

    return torch.cat(outputs, dim=1)


def train_one_epoch(
    model: CNNtoLSTM,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    ss_start: float,
    ss_end: float,
    log_interval: int = 50,
    use_scheduled_sampling: bool = True
) -> Tuple[float, float]:
    """Train for a single epoch and return average loss and teacher forcing ratio."""
    model.train()
    running_loss = 0.0

    if use_scheduled_sampling:
        tf_ratio = max(ss_end, ss_start - (epoch / max(1, total_epochs - 1)) * (ss_start - ss_end))
    else:
        tf_ratio = 1.0

    for step, (images, captions, lengths) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        features = model.encoder(images)
        logits = decoder_forward_scheduled(
            decoder=model.decoder,
            features=features,
            captions=captions,
            teacher_forcing_ratio=tf_ratio,
        )

        _, _, vocab_size = logits.shape
        logits_flat = logits[:, 1:, :].reshape(-1, vocab_size)
        targets_flat = captions[:, 1:].reshape(-1)

        loss = criterion(logits_flat, targets_flat)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

        if (step + 1) % log_interval == 0:
            print(
                f"[Epoch {epoch + 1:02d}] Step {step + 1:04d}/{len(loader)} "
                f"Loss: {running_loss / (step + 1):.4f}  TF: {tf_ratio:.3f}"
            )

    avg_loss = running_loss / max(1, len(loader))
    return avg_loss, tf_ratio


@torch.no_grad()
def evaluate_loss(
    model: CNNtoLSTM,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Evaluate average loss on the validation set."""
    model.eval()
    running_loss = 0.0

    for images, captions, lengths in loader:
        images = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)

        outputs = model(images, captions)
        _, _, vocab_size = outputs.shape
        loss = criterion(outputs[:, 1:, :].reshape(-1, vocab_size), captions[:, 1:].reshape(-1))
        running_loss += loss.item()

    return running_loss / max(1, len(loader))


def save_checkpoint(state: Dict, is_best: bool, ckpt_dir: str, filename: str = "last.pt"):
    """Save training checkpoint and optionally copy as best."""
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, filename)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(ckpt_dir, "model_best.pt")
        torch.save(state, best_path)


def main():
    parser = argparse.ArgumentParser(description="Train CNN-LSTM Image Captioning model.")
    parser.add_argument("--config", type=str, default="configs/config.yml", help="Path to YAML config.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--resume", type=str, default="", help="Path to a checkpoint to resume from.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    data_dir = cfg["data"]["images_dir"]
    captions_csv_train = cfg["data"]["captions_train"]
    captions_csv_val = cfg["data"]["captions_val"]
    vocab_path = cfg["data"]["vocab_path"]
    ckpt_dir = cfg["train"]["ckpt_dir"]
    log_dir = cfg["train"]["log_dir"]

    vocab = load_vocabulary(vocab_path)
    vocab_size = len(vocab)
    pad_idx = vocab.get("<pad>", 0)

    transform = get_transforms(img_size=cfg["data"].get("img_size", 299))

    train_set = FlickrDataset(
        root_dir=data_dir,
        captions_file=captions_csv_train,
        vocab=vocab,
        transform=transform
    )
    val_set = FlickrDataset(
        root_dir=data_dir,
        captions_file=captions_csv_val,
        vocab=vocab,
        transform=transform
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 4),
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 4),
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )

    model = CNNtoLSTM(
        embed_size=cfg["model"]["embed_dim"],
        hidden_size=cfg["model"]["hidden_dim"],
        vocab_size=vocab_size,
        num_layers=cfg["model"].get("num_layers", 1),
        dropout=cfg["model"].get("dropout", 0.3),
        train_cnn=cfg["model"].get("train_cnn", False),
    ).to(device)

    print(f"Trainable parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_idx,
        label_smoothing=cfg["train"].get("label_smoothing", 0.1)
    )

    base_lr = cfg["train"]["learning_rate"]
    if cfg["model"].get("train_cnn", False):
        decoder_params = list(model.decoder.parameters()) + list(model.encoder.fc.parameters()) + list(model.encoder.bn.parameters())
        encoder_params = [
            p for name, p in model.encoder.named_parameters()
            if p.requires_grad and not name.startswith("fc.") and not name.startswith("bn.")
        ]
        optimizer = optim.Adam([
            {"params": encoder_params, "lr": base_lr * cfg["model"].get("cnn_lr_factor", 0.1)},
            {"params": decoder_params, "lr": base_lr},
        ])
    else:
        optimizer = optim.Adam(model.parameters(), lr=base_lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg["train"].get("lr_factor", 0.5),
        patience=cfg["train"].get("lr_patience", 2)
    )

    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    start_epoch = 0
    best_val = math.inf

    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        scheduler.load_state_dict(ckpt["sched_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", best_val)

    ss_start = cfg["train"].get("teacher_forcing_start", 0.7)
    ss_end = cfg["train"].get("teacher_forcing_end", 0.3)
    use_ss = cfg["train"].get("use_scheduled_sampling", True)
    epochs = cfg["train"]["epochs"]

    print("=== Training started ===")
    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        train_loss, tf_ratio = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
            ss_start=ss_start,
            ss_end=ss_end,
            log_interval=cfg["train"].get("log_interval", 50),
            use_scheduled_sampling=use_ss
        )

        val_loss = evaluate_loss(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device
        )

        scheduler.step(val_loss)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Train/teacher_forcing_ratio", tf_ratio, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        dt = time.time() - t0
        print(
            f"[Epoch {epoch + 1:02d}/{epochs}] "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  tf={tf_ratio:.3f}  time={dt:.1f}s"
        )

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss

        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "sched_state": scheduler.state_dict(),
            "best_val": best_val,
            "config": cfg,
            "vocab_size": vocab_size,
            "seed": seed,
        }
        save_checkpoint(state, is_best=False, ckpt_dir=ckpt_dir, filename="last.pt")
        if is_best:
            save_checkpoint(state, is_best=True, ckpt_dir=ckpt_dir, filename=f"epoch_{epoch:03d}.pt")

    writer.close()
    print("=== Training finished ===")


if __name__ == "__main__":
    main()
