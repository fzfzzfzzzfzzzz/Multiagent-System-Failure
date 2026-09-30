"""Episode-level analysis for the frozen P2 development diagnostic.

P2 is exploratory.  This script describes content and routing effects without
changing the preregistered H1/H4 conclusions or selecting a new test design.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from analyze_core_failures import BOOLEAN_METRICS, aggregate, derive_run, read_jsonl, write_csv


CELLS = ("model_auto", "reviewed_auto", "model_reviewed_route", "reviewed_reviewed_route")
METRICS = ("first_success", "final_success", "abstained", "recurrence",
           "target_avoided_but_failed", "any_new_error", "any_tool_error", "budget_ceiling")


def cluster(row):
    return row["family_id"].split("/")[1]


def paired_factor(rows, metric, stratum="all", bootstrap=5000, seed=260930):
    groups = {name: {} for name in CELLS}
    opportunity = {}
    for row in rows:
        key = (row["meta_id"], row["kind"], row["target_id"], row["seed"])
        groups[row["method"]][key] = row
        opportunity[(key, row["method"])] = row["opportunity"]
    keys = set.intersection(*(set(group) for group in groups.values()))
    if metric == "recurrence":
        keys = {key for key in keys if all(opportunity[(key, method)] for method in CELLS)}
    effects = defaultdict(list)
    unit_rows = []
    for key in sorted(keys):
        a, b, c, d = (float(groups[name][key][metric]) for name in CELLS)
        values = {"content": ((b-a)+(d-c))/2, "routing": ((c-a)+(d-b))/2,
                  "interaction": d-b-c+a}
        family = cluster(groups[CELLS[0]][key])
        for effect, value in values.items():
            effects[(effect, family)].append(value)
            unit_rows.append({"stratum": stratum, "metric": metric, "effect": effect,
                              "meta_id": key[0], "kind": key[1], "target_id": key[2], "seed": key[3],
                              "family": family, "value": value})
    output = []
    rng = np.random.default_rng(seed)
    for effect in ("content", "routing", "interaction"):
        means = np.array([np.mean(values) for (name, _), values in sorted(effects.items()) if name == effect])
        if not len(means):
            output.append({"stratum": stratum, "metric": metric, "effect": effect,
                           "n_units": 0, "n_clusters": 0, "estimate": None, "ci_low": None,
                           "ci_high": None, "p_exact_sign_flip": None})
            continue
        estimate = float(means.mean())
        draws = np.mean(rng.choice(means, (bootstrap, len(means)), replace=True), axis=1)
        signs = np.array(list(itertools.product((-1, 1), repeat=len(means))), dtype=float)
        null = np.mean(signs * means, axis=1)
        p_value = float(np.mean(np.abs(null) >= abs(estimate)-1e-12))
        output.append({"stratum": stratum, "metric": metric, "effect": effect,
                       "n_units": len(keys), "n_clusters": len(means), "estimate": estimate,
                       "ci_low": float(np.quantile(draws, .025)), "ci_high": float(np.quantile(draws, .975)),
                       "p_exact_sign_flip": p_value})
    return output, unit_rows


def report(derived, summaries, effects):
    total = len(derived)
    categories = Counter(row["outcome_category"] for row in derived)
    overall = aggregate(derived, ("experiment", "method"))
    lines = [
        "# P2 报告内容 × 收件路由诊断", "",
        "本分析属于开发集探索性诊断；协议审阅报告是确定性参考干预，不是独立人工标注。结果不修改 H1–H4。", "",
        f"共 {total} 个真实 Qwen3-8B 回合，四组各 {total//4} 个；首次成功和最终成功均为 0。", "",
        "## 四组总体结果", "",
        "| 组别 | n | 未提交 | 目标复发 | 避免目标错误但仍失败 | 新错误 | 触及调用上限 |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        lines.append(f"| {row['method']} | {row['n']} | {row['abstained_rate']:.3f} | {row['recurrence_rate']:.3f} | "
                     f"{row['target_avoided_but_failed_rate']:.3f} | {row['any_new_error_rate']:.3f} | {row['budget_ceiling_rate']:.3f} |")
    lines += ["", "## 2×2 配对因子效应", "",
              "正值表示该干预提高对应指标；对未提交、复发和新错误而言，正值是更差。区间按六个源规则家族聚类 bootstrap；p 值为家族级精确符号翻转，仅作探索性描述。", "",
              "| 范围 | 指标 | 因子 | 效应 | 95% CI | p |", "|---|---|---|---:|---:|---:|"]
    for row in effects:
        if row["metric"] not in ("first_success", "final_success", "abstained", "recurrence"):
            continue
        estimate = "NA" if row["estimate"] is None else f"{row['estimate']:+.3f}"
        interval = "NA" if row["ci_low"] is None else f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]"
        p_value = "NA" if row["p_exact_sign_flip"] is None else f"{row['p_exact_sign_flip']:.3f}"
        lines.append(f"| {row['stratum']} | {row['metric']} | {row['effect']} | {estimate} | {interval} | {p_value} |")
    lines += ["", "## 互斥失败构成", "", "| 类别 | 回合数 | 比例 |", "|---|---:|---:|"]
    for category, count in categories.most_common():
        lines.append(f"| {category} | {count} | {count/total:.3f} |")
    lines += ["", "## 解释", "",
              "- 更好的参考内容和预设收件角色都没有产生一次完整成功，因此 P2 不支持把零成功主要归因于报告措辞或初始收件人选择。",
              "- 参考内容对目标复发只有小幅总体下降；需要结合未提交和其他错误判断，不能把少复发直接解释成学习成功。",
              "- 3 次调用的形式约束可解，但真实模型普遍触及预算上限；剩余瓶颈是跨角色执行与时序控制，而不是单独的报告内容或路由。",
              "- 下一步若继续，只应在开发集上分离执行预算/能力与记忆干预，例如固定最佳 P2 单元后比较 3/5/7 次调用；不得用本结果回改冻结 H1/H4。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="P2 directory containing results.jsonl and episodes")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.out); out.mkdir(parents=True, exist_ok=True)
    source_rows = read_jsonl(root / "results.jsonl")
    derived, missing = [], []
    for row in source_rows:
        path = root / "episodes" / f"{row['run_id']}.json"
        if not path.exists():
            missing.append(row["run_id"]); continue
        value = derive_run(row, json.loads(path.read_text(encoding="utf-8")))
        value.update({"meta_id": row["meta_id"], "target_id": row["target_id"]})
        derived.append(value)
    with (out / "run_diagnostics.jsonl").open("w", encoding="utf-8") as handle:
        for row in derived: handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summaries = aggregate(derived, ("experiment", "kind", "method"))
    write_csv(out / "summary.csv", summaries)
    write_csv(out / "mechanism_summary.csv", aggregate(derived, ("experiment", "mechanism", "kind", "method")))
    write_csv(out / "family_summary.csv", aggregate(derived, ("experiment", "family_id", "kind", "method")))
    effects, unit_rows = [], []
    strata = [("all", derived), ("same", [r for r in derived if r["kind"] == "same"]),
              ("transfer", [r for r in derived if r["kind"] == "transfer"])]
    strata += [(mechanism, [r for r in derived if r["mechanism"] == mechanism]) for mechanism in ("F1", "F2", "F3", "F4")]
    for name, rows in strata:
        for metric in METRICS:
            output, units = paired_factor(rows, metric, name)
            effects.extend(output); unit_rows.extend(units)
    write_csv(out / "factor_effects.csv", effects)
    write_csv(out / "factor_unit_effects.csv", unit_rows)
    failed_checks = Counter(check for row in derived for check in row["first_failed_checks"])
    errors = Counter((row["first_tool_error_role"], row["first_tool_error_tool"], row["first_tool_error"])
                     for row in derived if row["first_tool_error_role"])
    (out / "failure_counts.json").write_text(json.dumps({"outcomes": Counter(r["outcome_category"] for r in derived),
        "first_failed_checks": failed_checks, "first_tool_errors": {json.dumps(k): v for k, v in errors.items()}}, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"complete": len(derived) == 192 and not missing, "exploratory": True, "rows": len(derived),
                "missing_episodes": missing, "source_results_sha256": __import__("hashlib").sha256((root/"results.jsonl").read_bytes()).hexdigest(),
                "cluster": "source aggregation-rule family", "warning": "cannot revise preregistered H1-H4"}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "REPORT.md").write_text(report(derived, summaries, effects), encoding="utf-8")
    print(json.dumps({"out": str(out), **manifest}, ensure_ascii=False, indent=2))
    if not manifest["complete"]: raise SystemExit(2)


if __name__ == "__main__":
    main()
