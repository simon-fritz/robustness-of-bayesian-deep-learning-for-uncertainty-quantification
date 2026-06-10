"""Generic training loop (MAP). Method-agnostic."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..evaluation.metrics import accuracy, auroc


class Trainer:
    """Standard supervised trainer with TensorBoard logging and best-val checkpointing."""

    def __init__(
        self,
        cfg,
        device: str | None = None,
        log_dir: str | Path | None = None,
        ckpt_path: str | Path | None = None,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.log_dir = str(log_dir) if log_dir is not None else None
        self.ckpt_path = str(ckpt_path) if ckpt_path is not None else None
        self.class_weights = class_weights
        # Optional mixed precision (CUDA only); opt-in via the training config.
        self.use_amp = bool(getattr(self.cfg, "use_amp", False)) and self.device.startswith("cuda")

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        name = str(getattr(self.cfg, "optimizer", "adam")).lower()
        lr = float(self.cfg.lr)
        wd = float(getattr(self.cfg, "weight_decay", 0.0))
        if name == "adam":
            return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        if name == "sgd":
            return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        raise ValueError(f"unknown optimizer: {name}")

    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> nn.Module:
        model = model.to(self.device)
        opt = self._build_optimizer(model)
        weights = self.class_weights.to(self.device) if self.class_weights is not None else None
        criterion = nn.CrossEntropyLoss(weight=weights)

        epochs = int(self.cfg.epochs)
        patience = int(getattr(self.cfg, "early_stopping_patience", 0))
        writer = SummaryWriter(self.log_dir) if self.log_dir else None
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        if self.use_amp:
            print("mixed precision (AMP) enabled", flush=True)

        best_val = float("inf")
        best_state: dict | None = None
        no_improve = 0

        for epoch in range(epochs):
            model.train()
            running, n = 0.0, 0
            for x, y in train_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True).long()
                opt.zero_grad()
                with torch.autocast(device_type="cuda", enabled=self.use_amp):
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                running += loss.item() * x.size(0)
                n += x.size(0)
            train_loss = running / max(n, 1)

            val_metrics: dict[str, float] = {}
            if val_loader is not None:
                val_metrics = self._evaluate(model, val_loader, criterion)

            msg = f"epoch {epoch + 1}/{epochs} train_loss={train_loss:.4f}"
            if val_metrics:
                msg += " " + " ".join(f"val_{k}={v:.4f}" for k, v in val_metrics.items())
            print(msg, flush=True)

            if writer is not None:
                # TensorBoard event files live on a shared network mount that can
                # raise transient OSError (Errno 5) on append. A logging hiccup
                # must never abort training: drop the writer and carry on.
                try:
                    writer.add_scalar("train/loss", train_loss, epoch)
                    for k, v in val_metrics.items():
                        writer.add_scalar(f"val/{k}", v, epoch)
                except OSError as exc:
                    print(f"[trainer] tensorboard logging disabled after I/O error: {exc}", flush=True)
                    try:
                        writer.close()
                    except OSError:
                        pass
                    writer = None

            score = val_metrics.get("loss", train_loss)
            if score < best_val:
                best_val = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
                if self.ckpt_path:
                    os.makedirs(os.path.dirname(self.ckpt_path), exist_ok=True)
                    torch.save(best_state, self.ckpt_path)
            else:
                no_improve += 1
                if patience > 0 and no_improve >= patience:
                    print(f"early stop at epoch {epoch + 1} (best val_loss={best_val:.4f})")
                    break

        if writer is not None:
            try:
                writer.close()
            except OSError:
                pass
        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    @torch.no_grad()
    def _evaluate(self, model: nn.Module, loader: DataLoader, criterion: nn.Module) -> dict[str, float]:
        model.eval()
        running, n = 0.0, 0
        all_probs, all_y = [], []
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y_dev = y.to(self.device, non_blocking=True).long()
            logits = model(x)
            running += criterion(logits, y_dev).item() * x.size(0)
            n += x.size(0)
            all_probs.append(torch.softmax(logits, dim=-1).cpu())
            all_y.append(y_dev.cpu())
        probs = torch.cat(all_probs)
        targets = torch.cat(all_y)
        return {
            "loss": running / max(n, 1),
            "acc": accuracy(targets, probs),
            "auroc": auroc(targets, probs),
        }
