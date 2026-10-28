#!/usr/bin/env python
"""Forecast PM10 with teacher, HYSPLIT and static graph alternatives."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from _bootstrap import default_config
from ltkg_2021_2022.config import load_settings, parse_utc
from ltkg_2021_2022.dataset import load_manifest, load_sample, load_weather
from ltkg_2021_2022.io import read_csv, write_csv, write_json
from ltkg_2021_2022.model import LaggedTransportPM10, device, seed_everything


class PM10Reader:
    def __init__(self, settings, sites: list[str]):
        self.settings, self.sites, self.index = settings, sites, {site: number for number, site in enumerate(sites)}

    @lru_cache(maxsize=20000)
    def at(self, timestamp) -> np.ndarray:
        path = self.settings.pm10_root / f"{(timestamp + timedelta(hours=self.settings.pm10_utc_offset_hours)):%Y%m%d%H}.csv"
        rows = read_csv(path)
        values = np.full(len(self.sites), np.nan, dtype=np.float32)
        for row in rows:
            if row.get("site_code", "").strip() in self.index and row.get("PM10", "").strip():
                values[self.index[row["site_code"].strip()]] = float(row["PM10"])
        if not np.isfinite(values).all(): raise ValueError(f"Incomplete PM10: {path}")
        return values


def normalize_sources(matrix: np.ndarray) -> np.ndarray:
    matrix = np.maximum(matrix.astype(np.float32), 0.0)
    total = matrix.sum(axis=0, keepdims=True)
    empty = total[0] <= 1e-12
    if np.any(empty): matrix[:, empty] = 1.0 / matrix.shape[0]; total = matrix.sum(axis=0, keepdims=True)
    return matrix / total


class GraphProvider:
    def __init__(self, settings, source: str, variant: str, teacher_seed: int, knn: int):
        self.settings, self.source, self.variant, self.teacher_seed, self.knn = settings, source, variant, teacher_seed, knn
        with np.load(settings.dataset_root / "static_graph.npz") as static:
            self.sites = [str(value) for value in static["site_code"]]; self.longitude, self.latitude = static["longitude"], static["latitude"]
            distances = static["edge_static"][:, 0].reshape(len(self.sites), len(self.sites))
            self.elevation = static["node_static"][:, 0].astype(np.float32)
        _, example_label = load_sample(settings, load_manifest(settings)[0]); self.shape = example_label.shape
        self.distance = distances.astype(np.float32)
        if source == "teacher": self.directory = settings.output_root / "teacher_graphs" / variant / f"seed_{teacher_seed}"

    def static_graph(self, kind: str) -> np.ndarray:
        distance = self.distance.copy(); np.fill_diagonal(distance, np.inf)
        if kind == "distance": matrix = 1.0 / np.maximum(distance, 1e-3)
        elif kind == "knn":
            matrix = np.zeros_like(distance)
            for target in range(distance.shape[1]): matrix[np.argsort(distance[:, target])[:self.knn], target] = 1.0
        else: raise ValueError(kind)
        np.fill_diagonal(matrix, 0.0)
        return np.broadcast_to(normalize_sources(matrix), self.shape).copy()

    def wind_graph(self, row) -> np.ndarray:
        weather = load_weather(self.settings, row["weather_time_key"], ("u10", "v10"))
        wind_east, wind_north = float(weather[0].mean()), float(weather[1].mean())
        dx = self.longitude[None, :] - self.longitude[:, None]; dy = self.latitude[None, :] - self.latitude[:, None]
        matrix = np.maximum(dx * wind_east + dy * wind_north, 0.0) / np.maximum(np.hypot(dx, dy), 1e-6)
        np.fill_diagonal(matrix, 0.0)
        return np.broadcast_to(normalize_sources(matrix), self.shape).copy()

    def get(self, row) -> np.ndarray:
        if self.source == "hysplit": return load_sample(self.settings, row)[1]
        if self.source == "teacher":
            path = self.directory / f"{row['weather_time_key']}.npz"
            if not path.is_file(): raise FileNotFoundError(f"Missing predicted graph: {path}; run 04_export_teacher_graphs.py --split all.")
            with np.load(path) as values: return values["transport_weight"].astype(np.float32)
        if self.source in {"distance", "knn"}: return self.static_graph(self.source)
        return self.wind_graph(row)


class ForecastDataset(Dataset):
    def __init__(self, rows, reader, provider, mean, std, history_hours, horizons):
        self.rows, self.reader, self.provider, self.mean, self.std = rows, reader, provider, mean, std
        self.history_hours, self.horizons = history_hours, horizons
        elevation = provider.elevation; self.elevation = (elevation - elevation.mean()) / max(float(elevation.std()), 1e-6)

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]; end = parse_utc(row["end_time_utc"])
        history = np.stack([self.reader.at(end - timedelta(hours=offset)) for offset in range(self.history_hours, 0, -1)])
        target = np.stack([self.reader.at(end + timedelta(hours=horizon)) for horizon in self.horizons])
        return (
            torch.from_numpy(((history - self.mean) / self.std).astype(np.float32)), torch.from_numpy(self.provider.get(row)),
            torch.from_numpy(self.elevation.astype(np.float32)), torch.from_numpy(((target - self.mean) / self.std).astype(np.float32)),
        )


def epoch(model, loader, optimizer=None):
    training = optimizer is not None; model.train(training); total = 0.0
    for history, graph, elevation, target in loader:
        history, graph, elevation, target = [value.to(device()) for value in (history, graph, elevation, target)]
        if training: optimizer.zero_grad(set_to_none=True)
        loss = ((model(history, graph, elevation) - target) ** 2).mean()
        if training: loss.backward(); optimizer.step()
        total += float(loss.detach()) * history.shape[0]
    return total / len(loader.dataset)


def evaluate(model, loader, mean, std):
    model.eval(); actual, predicted = [], []
    with torch.no_grad():
        for history, graph, elevation, target in loader:
            predicted.append(model(history.to(device()), graph.to(device()), elevation.to(device())).cpu().numpy() * std + mean)
            actual.append(target.numpy() * std + mean)
    return np.concatenate(actual), np.concatenate(predicted)


def metrics(actual, predicted):
    error = predicted - actual; total = np.sum((actual - actual.mean()) ** 2)
    return {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt((error ** 2).mean())), "r2": 1.0 - float((error ** 2).sum() / total) if total > 1e-12 else None}


def eligible_rows(settings, rows, horizons):
    result = {"train": [], "valid": [], "test": []}
    for row in rows:
        end = parse_utc(row["end_time_utc"]); last_target = end + timedelta(hours=max(horizons))
        if not settings.contains(last_target): continue
        if settings.split_name(last_target) != row["split"]: continue
        result[row["split"]].append(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config()); parser.add_argument("--data-root", type=Path)
    parser.add_argument("--pm10-root", type=Path); parser.add_argument("--output-root", type=Path)
    parser.add_argument("--graph-source", choices=("teacher", "hysplit", "distance", "knn", "wind_rule"), default="teacher")
    parser.add_argument("--teacher-variant", default="full"); parser.add_argument("--teacher-seed", type=int, default=42); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history-hours", type=int, default=24); parser.add_argument("--horizons", type=int, nargs="+", default=(1, 3, 6, 12))
    parser.add_argument("--knn", type=int, default=5); parser.add_argument("--epochs", type=int, default=100); parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16); parser.add_argument("--hidden", type=int, default=64); parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args(); settings = load_settings(args.config, data_root=args.data_root, pm10_root=args.pm10_root, output_root=args.output_root)
    horizons = tuple(sorted(set(args.horizons))); seed_everything(args.seed)
    provider = GraphProvider(settings, args.graph_source, args.teacher_variant, args.teacher_seed, args.knn); reader = PM10Reader(settings, provider.sites)
    if args.history_hours < provider.shape[1]: parser.error(f"--history-hours must be >= label lag count ({provider.shape[1]})")
    rows = eligible_rows(settings, load_manifest(settings), horizons)
    if any(not rows[name] for name in rows): raise RuntimeError(f"A forecast split has no complete examples: { {name: len(value) for name, value in rows.items()} }")
    train_values = np.concatenate([np.stack([reader.at(parse_utc(row['end_time_utc']) - timedelta(hours=offset)) for offset in range(args.history_hours, 0, -1)]).ravel() for row in rows["train"]])
    mean, std = float(train_values.mean()), max(float(train_values.std()), 1e-6)
    datasets = [ForecastDataset(rows[name], reader, provider, mean, std, args.history_hours, horizons) for name in ("train", "valid", "test")]
    loaders = [DataLoader(dataset, args.batch_size, shuffle=index == 0) for index, dataset in enumerate(datasets)]
    model = LaggedTransportPM10(args.history_hours, provider.shape[0], provider.shape[1], len(horizons), args.hidden).to(device()); optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, best_loss, stale, history = None, float("inf"), 0, []
    for number in range(1, args.epochs + 1):
        train_loss, valid_loss = epoch(model, loaders[0], optimizer), epoch(model, loaders[1])
        history.append({"epoch": number, "train_mse": train_loss, "valid_mse": valid_loss}); print(f"Epoch {number:03d}: train={train_loss:.5f}; valid={valid_loss:.5f}")
        if valid_loss < best_loss: best, best_loss, best_epoch, stale = copy.deepcopy(model.state_dict()), valid_loss, number, 0
        else:
            stale += 1
            if stale >= args.patience: break
    model.load_state_dict(best); actual, predicted = evaluate(model, loaders[2], mean, std)
    persistence = np.repeat(np.stack([reader.at(parse_utc(row["end_time_utc"])) for row in rows["test"]])[:, None, :], len(horizons), axis=1)
    metric_rows = []
    for method, values in ((args.graph_source, predicted), ("persistence", persistence)):
        metric_rows.append({"method": method, "horizon": "overall", **metrics(actual, values)})
        metric_rows.extend({"method": method, "horizon": horizon, **metrics(actual[:, index], values[:, index])} for index, horizon in enumerate(horizons))
    output = settings.pm10_run_dir(args.graph_source, args.seed); output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, output / "best_model.pt"); write_csv(output / "training_history.csv", history); write_csv(output / "pm10_metrics.csv", metric_rows)
    np.savez_compressed(output / "test_predictions.npz", actual=actual, predicted=predicted, persistence=persistence, horizons=np.asarray(horizons), site_code=np.asarray(provider.sites))
    figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for method in {row["method"] for row in metric_rows}:
        points = [row for row in metric_rows if row["method"] == method and row["horizon"] != "overall"]; axis.plot([int(row["horizon"]) for row in points], [row["rmse"] for row in points], marker="o", label=method)
    axis.set(xlabel="Forecast horizon (h)", ylabel="PM10 RMSE", title="2022 H2 PM10 forecast"); axis.legend(); figure.savefig(output / "pm10_rmse_by_horizon.png", dpi=220); plt.close(figure)
    write_json(output / "run_metadata.json", {"graph_source": args.graph_source, "teacher_variant": args.teacher_variant if args.graph_source == "teacher" else None, "seed": args.seed, "best_epoch": best_epoch, "best_validation_mse": best_loss, "horizons_h": list(horizons), "examples": {name: len(value) for name, value in rows.items()}})
    print(json.dumps(metric_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
