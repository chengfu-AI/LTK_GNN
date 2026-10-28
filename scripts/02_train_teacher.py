#!/usr/bin/env python
"""Train a leakage-safe ERA5 + terrain -> HYSPLIT transport-kernel teacher."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from _bootstrap import default_config
from ltkg_2021_2022.config import load_settings
from ltkg_2021_2022.dataset import load_manifest, load_sample
from ltkg_2021_2022.io import write_csv
from ltkg_2021_2022.model import (
    TransportDataset, build_teacher, device, fit_weather_normalization, save_teacher_run, seed_everything,
)


def epoch(model, loader, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    total, denominator = 0.0, 0
    for weather, label in loader:
        weather, label = weather.to(device()), label.to(device())
        if training:
            optimizer.zero_grad(set_to_none=True)
        prediction = model(weather)
        valid = label.sum(dim=3) > 0.0
        cross_entropy = -(label * prediction.clamp_min(1e-8).log()).sum(dim=3)
        loss = cross_entropy[valid].mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss; inspect ERA5 and HYSPLIT inputs.")
        if training:
            loss.backward()
            optimizer.step()
        count = int(valid.sum().item())
        total += float(loss.detach()) * count
        denominator += count
    if denominator == 0:
        raise RuntimeError("No non-empty HYSPLIT labels in this split.")
    return total / denominator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config())
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--pm10-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--variant", default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    settings = load_settings(args.config, data_root=args.data_root, pm10_root=args.pm10_root, output_root=args.output_root)
    if args.variant not in settings.teacher_variants:
        parser.error(f"--variant must be one of {tuple(settings.teacher_variants)}")
    seed_everything(args.seed)
    rows = load_manifest(settings)
    train, valid = [row for row in rows if row["split"] == "train"], [row for row in rows if row["split"] == "valid"]
    variables = tuple(settings.teacher_variants[args.variant]["weather"])
    mean, std = fit_weather_normalization(settings, train, variables)
    _, label = load_sample(settings, train[0])
    model = build_teacher(settings, args.variant, tuple(label.shape), args.hidden).to(device())
    loaders = [
        DataLoader(TransportDataset(settings, subset, variables, mean, std), args.batch_size, shuffle=index == 0)
        for index, subset in enumerate((train, valid))
    ]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, best_loss, stale, history = None, float("inf"), 0, []
    for number in range(1, args.epochs + 1):
        train_loss = epoch(model, loaders[0], optimizer)
        with torch.no_grad():
            valid_loss = epoch(model, loaders[1])
        history.append({"epoch": number, "train_cross_entropy": train_loss, "valid_cross_entropy": valid_loss})
        print(f"Epoch {number:03d}: train CE={train_loss:.5f}; valid CE={valid_loss:.5f}")
        if valid_loss < best_loss:
            best, best_loss, best_epoch, stale = copy.deepcopy(model.state_dict()), valid_loss, number, 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stop at epoch {number}; selected epoch {best_epoch}.")
                break
    if best is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    model.load_state_dict(best)
    run_dir = settings.teacher_run_dir(args.variant, args.seed)
    metadata = {
        "variant": args.variant, "seed": args.seed, "hidden": args.hidden, "label_shape": list(label.shape),
        "weather_variables": list(variables), "use_terrain_barrier": settings.teacher_variants[args.variant]["use_terrain_barrier"],
        "train_samples": len(train), "valid_samples": len(valid), "best_epoch": best_epoch, "best_validation_cross_entropy": best_loss,
        "train_end_utc": settings.train_end.strftime("%Y-%m-%dT%H:%M:%SZ"), "valid_end_utc": settings.valid_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    save_teacher_run(run_dir, model, mean, std, metadata)
    write_csv(run_dir / "training_history.csv", history)
    print(f"Saved selected teacher model: {run_dir}")


if __name__ == "__main__":
    main()
