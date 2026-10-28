"""Neural models and reproducible feature loading for the LTK-GNN workflow."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from .config import Settings
from .dataset import load_sample, load_weather


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def z_score(values: np.ndarray, *, keep_last_binary: bool = False) -> np.ndarray:
    output = values.astype(np.float32).copy()
    stop = output.shape[1] - int(keep_last_binary)
    mean, std = output[:, :stop].mean(0), output[:, :stop].std(0)
    std[std < 1e-6] = 1.0
    output[:, :stop] = (output[:, :stop] - mean) / std
    return output


def load_static_features(settings: Settings, variant: str) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if variant not in settings.teacher_variants:
        raise ValueError(f"Unknown teacher variant {variant!r}")
    with np.load(settings.dataset_root / "static_graph.npz") as static:
        sites = [str(value) for value in static["site_code"]]
        longitude, latitude = static["longitude"].astype(np.float32), static["latitude"].astype(np.float32)
        node = np.column_stack([longitude, latitude, static["node_static"]]).astype(np.float32)
        edge = static["edge_static"].astype(np.float32)
        names = [str(value) for value in static["edge_feature_names"]]
    if not settings.teacher_variants[variant]["use_terrain_barrier"]:
        location = names.index("terrain_barrier_m")
        edge, names = np.delete(edge, location, axis=1), names[:location] + names[location + 1 :]
    return sites, longitude, latitude, z_score(node), z_score(edge, keep_last_binary=True)


def fit_weather_normalization(settings: Settings, rows: list[dict[str, str]], variables: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Fit grid-cell-wise statistics from training rows only, in streaming form."""
    total = None
    squared = None
    count = 0
    for row in rows:
        weather = load_weather(settings, row["weather_time_key"], variables).astype(np.float64)
        if total is None:
            total, squared = np.zeros_like(weather), np.zeros_like(weather)
        total += weather
        squared += weather * weather
        count += 1
    if count == 0 or total is None or squared is None:
        raise ValueError("No training weather rows available for normalization.")
    mean = total / count
    variance = np.maximum(squared / count - mean * mean, 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def normalize_weather(weather: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if weather.shape != mean.shape or weather.shape != std.shape:
        raise ValueError(f"Weather/normalization shape mismatch: {weather.shape}, {mean.shape}, {std.shape}")
    return ((weather - mean) / np.maximum(std, 1e-6)).astype(np.float32)


class TransportDataset(Dataset):
    def __init__(self, settings: Settings, rows: list[dict[str, str]], variables: tuple[str, ...], mean: np.ndarray, std: np.ndarray):
        self.settings, self.rows, self.variables, self.mean, self.std = settings, rows, variables, mean, std

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        _, label = load_sample(self.settings, row)
        weather = normalize_weather(load_weather(self.settings, row["weather_time_key"], self.variables), self.mean, self.std)
        return torch.from_numpy(weather), torch.from_numpy(label)


class TransportTeacher(nn.Module):
    """ERA5 field encoder with directed terrain features and lag/height embeddings."""

    def __init__(self, weather_channels: int, node_features: np.ndarray, edge_features: np.ndarray, heights: int, lags: int, hidden: int):
        super().__init__()
        self.nodes, self.heights, self.lags = node_features.shape[0], heights, lags
        self.register_buffer("node_features", torch.tensor(node_features))
        self.register_buffer("edge_features", torch.tensor(edge_features))
        self.weather_encoder = nn.Sequential(
            nn.Conv2d(weather_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, hidden, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.node_encoder = nn.Sequential(nn.Linear(node_features.shape[1], hidden), nn.ReLU())
        self.edge_encoder = nn.Sequential(nn.Linear(edge_features.shape[1], hidden), nn.ReLU())
        self.lag_embedding, self.height_embedding = nn.Embedding(lags, hidden), nn.Embedding(heights, hidden)
        self.score = nn.Sequential(nn.Linear(hidden * 6, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, weather: torch.Tensor) -> torch.Tensor:
        batch, edges = weather.shape[0], self.nodes * self.nodes
        source = torch.arange(self.nodes, device=weather.device).repeat_interleave(self.nodes)
        target = torch.arange(self.nodes, device=weather.device).repeat(self.nodes)
        nodes, edge = self.node_encoder(self.node_features), self.edge_encoder(self.edge_features)
        geometry = torch.cat([nodes[source], nodes[target], edge], dim=1)
        parts = [
            self.weather_encoder(weather)[:, None, None, None, :].expand(batch, self.heights, self.lags, edges, -1),
            geometry[None, None, None, :, :].expand(batch, self.heights, self.lags, edges, -1),
            self.lag_embedding.weight[None, None, :, None, :].expand(batch, self.heights, -1, edges, -1),
            self.height_embedding.weight[None, :, None, None, :].expand(batch, -1, self.lags, edges, -1),
        ]
        logits = self.score(torch.cat(parts, dim=-1)).squeeze(-1)
        logits = logits.view(batch, self.heights, self.lags, self.nodes, self.nodes)
        return torch.softmax(logits, dim=3)


def build_teacher(settings: Settings, variant: str, label_shape: tuple[int, ...], hidden: int) -> TransportTeacher:
    _, _, _, node, edge = load_static_features(settings, variant)
    return TransportTeacher(
        len(settings.teacher_variants[variant]["weather"]), node, edge,
        heights=label_shape[0], lags=label_shape[1], hidden=hidden,
    )


def save_teacher_run(run_dir: Path, model: nn.Module, mean: np.ndarray, std: np.ndarray, metadata: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, run_dir / "best_model.pt")
    np.savez_compressed(run_dir / "normalization.npz", weather_mean=mean, weather_std=std)
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_torch_load(path: Path, map_location: str) -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_teacher_run(settings: Settings, variant: str, seed: int) -> tuple[TransportTeacher, tuple[str, ...], np.ndarray, np.ndarray, dict]:
    run_dir = settings.teacher_run_dir(variant, seed)
    needed = [run_dir / "best_model.pt", run_dir / "normalization.npz", run_dir / "run_metadata.json"]
    if any(not path.is_file() for path in needed):
        raise FileNotFoundError(f"Missing teacher run in {run_dir}; train it first.")
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    model = build_teacher(settings, variant, tuple(metadata["label_shape"]), int(metadata["hidden"]))
    model.load_state_dict(_safe_torch_load(run_dir / "best_model.pt", device())["state_dict"])
    model.to(device()).eval()
    with np.load(run_dir / "normalization.npz") as values:
        mean, std = values["weather_mean"], values["weather_std"]
    return model, tuple(metadata["weather_variables"]), mean, std, metadata


def predict_kernel(settings: Settings, row: dict[str, str], model: nn.Module, variables: tuple[str, ...], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    weather = normalize_weather(load_weather(settings, row["weather_time_key"], variables), mean, std)
    with torch.no_grad():
        prediction = model(torch.from_numpy(weather).unsqueeze(0).to(device()))
    return prediction[0].detach().cpu().numpy().astype(np.float32)


class LaggedTransportPM10(nn.Module):
    """Lag-matched dynamic graph aggregation followed by a shared station decoder."""

    def __init__(self, history_hours: int, heights: int, lags: int, horizons: int, hidden: int):
        super().__init__()
        self.lags = lags
        self.decoder = nn.Sequential(
            nn.Linear(history_hours + heights * lags + 1, hidden), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, horizons),
        )

    def forward(self, history: torch.Tensor, graph: torch.Tensor, elevation: torch.Tensor) -> torch.Tensor:
        source_at_lag = torch.stack([history[:, -lag, :] for lag in range(1, self.lags + 1)], dim=1)
        incoming = torch.einsum("bhlst,bls->bhlt", graph, source_at_lag)
        incoming = incoming.permute(0, 3, 1, 2).reshape(history.shape[0], history.shape[2], -1)
        local = history.permute(0, 2, 1)
        features = torch.cat([local, incoming, elevation[:, :, None]], dim=-1)
        return self.decoder(features).permute(0, 2, 1)
