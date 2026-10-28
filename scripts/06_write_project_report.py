#!/usr/bin/env python
"""Create a paper-ready Markdown summary from completed real 2021–2022 runs only."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from _bootstrap import default_config
from ltkg_2021_2022.config import load_settings
from ltkg_2021_2022.io import read_csv, write_csv, write_json


def mean_std(values: list[float | None]) -> tuple[str, str]:
    finite = [float(value) for value in values if value is not None]
    if not finite: return "未运行", "0"
    return f"{np.mean(finite):.4f} ± {np.std(finite, ddof=0):.4f}", str(len(finite))


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n" + "\n".join("| " + " | ".join(row) + " |" for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config()); parser.add_argument("--data-root", type=Path)
    parser.add_argument("--pm10-root", type=Path); parser.add_argument("--output-root", type=Path)
    parser.add_argument("--minimum-seeds", type=int, default=5)
    args = parser.parse_args(); settings = load_settings(args.config, data_root=args.data_root, pm10_root=args.pm10_root, output_root=args.output_root)

    teacher = defaultdict(list)
    for path in settings.teachers_root.glob("*/seed_*/evaluation.json"):
        result = json.loads(path.read_text(encoding="utf-8")); teacher[result["variant"]].append(result)
    teacher_rows = []
    for variant in settings.teacher_variants:
        results = teacher[variant]; metric = [result.get("test_nonself", {}) for result in results]
        f1, count = mean_std([item.get("f1") for item in metric]); delay, _ = mean_std([item.get("delay_mae_h") for item in metric]); kernel, _ = mean_std([item.get("kernel_rmse") for item in metric])
        teacher_rows.append([variant, count, f1, delay, kernel, "完成" if len(results) >= args.minimum_seeds else "需补充随机种子"])

    pm10 = defaultdict(list)
    for path in settings.pm10_runs_root.glob("*/seed_*/pm10_metrics.csv"):
        for row in read_csv(path):
            if row["method"] != "persistence": pm10[(row["method"], row["horizon"])].append(row)
    pm10_rows = []
    for (method, horizon), results in sorted(pm10.items()):
        mae, count = mean_std([float(row["mae"]) for row in results]); rmse, _ = mean_std([float(row["rmse"]) for row in results]); r2, _ = mean_std([float(row["r2"]) if row["r2"] else None for row in results])
        pm10_rows.append([method, horizon, count, mae, rmse, r2])

    readiness = {
        "study_period_utc": ["2021-01-01T00:00:00Z", "2022-12-31T23:00:00Z"],
        "minimum_seeds": args.minimum_seeds,
        "teacher_completed_seeds": {variant: len(values) for variant, values in teacher.items()},
        "teacher_variants_missing_runs": [variant for variant in settings.teacher_variants if len(teacher[variant]) < args.minimum_seeds],
        "pm10_completed_comparisons": len(pm10_rows),
        "rule": "No missing result is converted to a numerical table value.",
    }
    output = settings.output_root / "paper_results"; output.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 2021–2022 LTK-GNN 项目结果摘要", "",
        "本文件只汇总 `outputs/` 中真实运行产生的结果。训练期为 2021 年，验证期为 2022 年上半年，测试期为 2022 年下半年。", "",
        "## 动态输送图重建（测试集、去自环）", "", markdown_table(["变体", "种子数", "F1", "Delay-MAE (h)", "Kernel-RMSE", "状态"], teacher_rows), "",
        "## PM10 下游预测（测试集）", "", markdown_table(["图结构", "时效 (h)", "种子数", "MAE", "RMSE", "R²"], pm10_rows or [["未运行", "-", "0", "未运行", "未运行", "未运行"]]), "",
        "## 报告约束", "", "- HYSPLIT 是模拟派生的物理参考，不能表述为真实源贡献。", "- 边阈值只能由验证集选择；测试集不用于调参。", "- 主表应在每个配置至少完成 5 个随机种子后使用均值±标准差。", "- 未完成对照保持“未运行”，不得填入模板或合成数值。", "",
    ]
    (output / "项目结果摘要.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(output / "experiment_readiness.json", readiness)
    write_csv(output / "teacher_summary.csv", [{"variant": row[0], "seeds": row[1], "f1": row[2], "delay_mae_h": row[3], "kernel_rmse": row[4], "status": row[5]} for row in teacher_rows])
    print(f"Wrote: {output / '项目结果摘要.md'}")


if __name__ == "__main__":
    main()
