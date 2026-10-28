#!/usr/bin/env python
"""Select a graph-edge threshold on validation only and report final test metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import default_config
from ltkg_2021_2022.config import load_settings
from ltkg_2021_2022.dataset import load_manifest, load_sample
from ltkg_2021_2022.io import write_csv, write_json
from ltkg_2021_2022.model import load_teacher_run, predict_kernel


def binary_metrics(reference: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    actual, predicted = reference.astype(bool).ravel(), (score >= threshold).ravel()
    tp = int(np.count_nonzero(actual & predicted)); fp = int(np.count_nonzero(~actual & predicted)); fn = int(np.count_nonzero(actual & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def roc_auc(reference: np.ndarray, score: np.ndarray) -> float | None:
    actual, values = reference.astype(bool).ravel(), score.ravel()
    positives, negatives = int(actual.sum()), int((~actual).sum())
    if not positives or not negatives:
        return None
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values, start = values[order], 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return float((ranks[actual].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def select_threshold(reference: np.ndarray, score: np.ndarray) -> float:
    candidates = np.unique(np.quantile(score, np.linspace(0.02, 0.98, 97)))
    best_threshold, best_f1 = float(candidates[0]), -1.0
    for threshold in candidates:
        f1 = binary_metrics(reference, score, float(threshold))["f1"]
        if f1 > best_f1:
            best_threshold, best_f1 = float(threshold), f1
    return best_threshold


def select_entries(reference: np.ndarray, prediction: np.ndarray, *, nonself: bool) -> tuple[np.ndarray, np.ndarray]:
    # Reference/prediction: [sample, height, lag, source, target].
    valid = reference.sum(axis=3) > 0.0
    include = np.broadcast_to(valid[:, :, :, None, :], reference.shape).copy()
    if nonself:
        source_target = ~np.eye(reference.shape[3], dtype=bool)
        include &= source_target[None, None, None, :, :]
    return reference[include], prediction[include]


def kernel_metrics(reference: np.ndarray, prediction: np.ndarray, *, nonself: bool, threshold: float | None = None) -> dict:
    values, scores = select_entries(reference, prediction, nonself=nonself)
    valid_groups = reference.sum(axis=3) > 0.0
    group_ce = -(reference * np.log(np.maximum(prediction, 1e-8))).sum(axis=3)
    result = {
        "cross_entropy": float(np.mean(group_ce[valid_groups])),
        "kernel_mae": float(np.mean(np.abs(scores - values))),
        "kernel_rmse": float(np.sqrt(np.mean((scores - values) ** 2))),
    }
    result["pearson_r"] = float(np.corrcoef(values, scores)[0, 1]) if values.std() > 0 and scores.std() > 0 else None
    truth = values >= 0.05
    if threshold is not None:
        result.update(binary_metrics(truth, scores, threshold))
        result["roc_auc"] = roc_auc(truth, scores)
    valid = reference.sum(axis=3) > 0.0
    actual_peak, predicted_peak = reference.argmax(axis=3), prediction.argmax(axis=3)
    if nonself:
        mask = ~np.eye(reference.shape[3], dtype=bool)[None, None, None, :, :]
        # Collapse the source axis after selecting maximum source; target self links use index equality.
        diagonal_targets = np.arange(reference.shape[3]) == actual_peak
        valid_for_top = valid & ~diagonal_targets
    else:
        valid_for_top = valid
    result["top1_accuracy"] = float(np.mean((actual_peak == predicted_peak)[valid_for_top])) if valid_for_top.any() else None

    reference_lag = reference.argmax(axis=2)
    prediction_lag = prediction.argmax(axis=2)
    relation_exists = reference.max(axis=2) > 0.0
    if nonself:
        relation_exists &= ~np.eye(reference.shape[3], dtype=bool)[None, None, :, :]
    result["delay_mae_h"] = float(np.mean(np.abs(reference_lag[relation_exists] - prediction_lag[relation_exists]))) if relation_exists.any() else None
    return result


def collect(settings, rows, model, variables, mean, std) -> tuple[np.ndarray, np.ndarray]:
    labels, predictions = [], []
    for number, row in enumerate(rows, start=1):
        _, label = load_sample(settings, row)
        labels.append(label)
        predictions.append(predict_kernel(settings, row, model, variables, mean, std))
        if number % 100 == 0 or number == len(rows):
            print(f"Predicted {number}/{len(rows)} samples")
    return np.stack(labels), np.stack(predictions)


def plot_curves(reference: np.ndarray, prediction: np.ndarray, threshold: float, path: Path) -> None:
    values, scores = select_entries(reference, prediction, nonself=True)
    truth = values >= 0.05
    thresholds = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 60)))
    points = [binary_metrics(truth, scores, float(value)) for value in thresholds]
    fig, axis = plt.subplots(figsize=(5, 4), constrained_layout=True)
    axis.plot([item["recall"] for item in points], [item["precision"] for item in points], label="validation PR")
    chosen = binary_metrics(truth, scores, threshold)
    axis.scatter([chosen["recall"]], [chosen["precision"]], color="crimson", label=f"threshold={threshold:.3f}")
    axis.set(xlabel="Recall", ylabel="Precision", title="Non-self transport-edge validation")
    axis.legend(); fig.savefig(path, dpi=220); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config()); parser.add_argument("--data-root", type=Path)
    parser.add_argument("--pm10-root", type=Path); parser.add_argument("--output-root", type=Path)
    parser.add_argument("--variant", default="full"); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    settings = load_settings(args.config, data_root=args.data_root, pm10_root=args.pm10_root, output_root=args.output_root)
    model, variables, mean, std, metadata = load_teacher_run(settings, args.variant, args.seed)
    rows = load_manifest(settings); valid = [row for row in rows if row["split"] == "valid"]; test = [row for row in rows if row["split"] == "test"]
    valid_reference, valid_prediction = collect(settings, valid, model, variables, mean, std)
    threshold_values, threshold_scores = select_entries(valid_reference, valid_prediction, nonself=True)
    threshold = select_threshold(threshold_values >= 0.05, threshold_scores)
    test_reference, test_prediction = collect(settings, test, model, variables, mean, std)
    results = {
        "variant": args.variant, "seed": args.seed, "prediction_edge_threshold": threshold,
        "test_all": kernel_metrics(test_reference, test_prediction, nonself=False, threshold=threshold),
        "test_nonself": kernel_metrics(test_reference, test_prediction, nonself=True, threshold=threshold),
        "validation_nonself": kernel_metrics(valid_reference, valid_prediction, nonself=True, threshold=threshold),
    }
    run_dir = settings.teacher_run_dir(args.variant, args.seed)
    write_json(run_dir / "evaluation.json", results)
    write_csv(run_dir / "evaluation_flat.csv", [{"scope": scope, **values} for scope, values in results.items() if isinstance(values, dict)])
    plot_curves(valid_reference, valid_prediction, threshold, run_dir / "validation_pr_curve.png")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
