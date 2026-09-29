"""Export and interpret the terminal H1/H4 confirmation without changing it."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

try:
    from remote import connection
except ModuleNotFoundError:  # Imported as scripts.finalize_core by the test suite.
    from scripts.remote import connection


PRIMARY = (
    ("H1", "E1", "components_SOCDKPU", "components_SP"),
    ("H4", "E3", "matched", "cycle"),
    ("H4_validation", "E3", "matched", "random"),
)
OUTCOMES = ("first_success", "recurrence")


def holm(rows):
    """Add Holm adjusted p-values across the declared primary family."""
    ordered = sorted(enumerate(rows), key=lambda item: item[1]["p_block_permutation"])
    running = 0.0
    for rank, (index, row) in enumerate(ordered):
        running = max(running, min(1.0, row["p_block_permutation"] * (len(rows) - rank)))
        rows[index]["p_holm_primary_family"] = running
    return rows


def classify(row, alpha=0.05):
    """Classify direction and uncertainty; recurrence is beneficial when lower."""
    if not row.get("estimable"):
        return "not_estimable"
    effect = row["difference"]
    low, high = row["ci95"]
    sign = 1 if row["metric"] == "first_success" else -1
    if abs(effect) < 1e-12:
        return "no_observed_difference"
    adjusted = row["p_holm_primary_family"] <= alpha
    directional_interval = low > 0 if sign > 0 else high < 0
    opposite_interval = high < 0 if sign > 0 else low > 0
    if sign * effect > 0 and adjusted and directional_interval:
        return "supports_expected_direction"
    if sign * effect < 0 and adjusted and opposite_interval:
        return "supports_opposite_direction"
    return "inconclusive"


def select_primary(e1, e3):
    by_experiment = {"E1": e1, "E3": e3}
    selected = []
    for hypothesis, experiment, left, right in PRIMARY:
        for metric in OUTCOMES:
            matches = [row for row in by_experiment[experiment]
                       if row.get("experiment") == experiment
                       and row.get("kind") == "same"
                       and row.get("a") == left and row.get("b") == right
                       and row.get("metric") == metric]
            if len(matches) != 1:
                raise RuntimeError(
                    f"expected one {experiment}/{left}/{right}/{metric} comparison, got {len(matches)}"
                )
            selected.append({"hypothesis": hypothesis, **matches[0]})
    holm(selected)
    for row in selected:
        row["expected_direction"] = "positive" if row["metric"] == "first_success" else "negative"
        row["evidence"] = classify(row)
    return selected


def read_remote_json(sftp, path):
    with sftp.open(path, "r") as handle:
        raw = handle.read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def read_remote_csv(sftp, path):
    with sftp.open(path, "r") as handle:
        raw = handle.read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return list(csv.DictReader(io.StringIO(raw)))


def same_condition_rates(e1, e3):
    wanted = {
        "E1": {"components_SOCDKPU", "components_SP"},
        "E3": {"matched", "cycle", "random"},
    }
    rows = []
    for experiment, source in (("E1", e1), ("E3", e3)):
        for row in source:
            if row["kind"] == "same" and row["method"] in wanted[experiment]:
                rows.append({
                    "experiment": experiment,
                    "method": row["method"],
                    "n_runs": int(row["n_runs"]),
                    "first_success": float(row["first_success"]),
                    "recurrence": float(row["recurrence"]),
                    "mean_total_tokens": float(row["mean_total_tokens"]),
                })
    return rows


def conditional_transfer_results(e1, e3):
    declared = (("E1", "components_SOCDKPU", "components_SP"),
                ("E3", "matched", "cycle"),
                ("E3", "matched", "random"))
    sources = {"E1": e1, "E3": e3}
    selected = []
    for experiment, left, right in declared:
        matches = [row for row in sources[experiment]
                   if row.get("kind") == "transfer" and row.get("a") == left
                   and row.get("b") == right and row.get("metric") == "recurrence"]
        if len(matches) != 1:
            raise RuntimeError(f"expected one transfer recurrence comparison for {experiment}/{left}/{right}")
        selected.append(matches[0])
    return selected


def hypothesis_summary(rows):
    result = {}
    for name in ("H1", "H4", "H4_validation"):
        states = [row["evidence"] for row in rows if row["hypothesis"] == name]
        if any(state == "supports_opposite_direction" for state in states):
            label = "contradicted_or_mixed"
        elif all(state == "supports_expected_direction" for state in states):
            label = "supported_on_both_primary_outcomes"
        elif any(state == "supports_expected_direction" for state in states):
            label = "partially_supported"
        else:
            label = "not_supported_by_this_confirmation"
        result[name] = {"classification": label, "outcome_evidence": states}
    return result


def markdown_report(payload):
    lines = [
        "# H1/H4 独立测试确认结果",
        "",
        "本文件只解释预先冻结的 same 条件主比较。首次成功差值为正、复发率差值为负才是预期方向。六项主检验共同进行 Holm 校正。",
        "",
        "| 假设 | 对照（A−B） | 指标 | 家族数 | 差值 | 95% CI | 原始 p | Holm p | 证据 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["primary_family"]:
        interval = f"[{row['ci95'][0]:.4f}, {row['ci95'][1]:.4f}]"
        lines.append(
            f"| {row['hypothesis']} | {row['a']} − {row['b']} | {row['metric']} | "
            f"{row.get('n_clusters', 0)} | {row['difference']:.4f} | {interval} | "
            f"{row['p_block_permutation']:.4g} | {row['p_holm_primary_family']:.4g} | {row['evidence']} |"
        )
    lines.extend([
        "",
        "## 判定",
        "",
    ])
    for name, value in payload["hypotheses"].items():
        lines.append(f"- **{name}**：`{value['classification']}`")
    lines.extend([
        "",
        "## same 条件绝对结果",
        "",
        "| 实验 | 方法 | 回合数 | 首次成功率 | 复发率 | 平均总 tokens |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in payload["same_condition_arm_rates"]:
        lines.append(
            f"| {row['experiment']} | {row['method']} | {row['n_runs']} | "
            f"{row['first_success']:.4f} | {row['recurrence']:.4f} | {row['mean_total_tokens']:.1f} |"
        )
    lines.extend([
        "",
        "## 预先保留的 transfer 条件分析",
        "",
        "这些结果不改变 same 条件主判定。这里的 Holm p 来自各实验输出的完整注册比较族。",
        "",
        "| 实验 | 对照（A−B） | 复发率差值 | 95% CI | Holm p |",
        "|---|---|---:|---:|---:|",
    ])
    for row in payload["conditional_transfer"]:
        interval = f"[{row['ci95'][0]:.4f}, {row['ci95'][1]:.4f}]"
        lines.append(
            f"| {row['experiment']} | {row['a']} − {row['b']} | {row['difference']:.4f} | "
            f"{interval} | {row['p_holm']:.4g} |"
        )
    lines.extend([
        "",
        "该判定限于 Qwen3-8B、当前受控 workflow 环境、冻结任务与预算。没有预注册非劣效界限，区间跨零时不能声称等效或非劣。transfer/reversal 与成本结果保留在导出的完整分析中，作为条件分析而不改变主判定。",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--credentials-pdf", default="fangc-README.pdf")
    parser.add_argument("--launch", default="artifacts/core_launch_config.json")
    parser.add_argument("--out", default="artifacts/core_confirmation")
    args = parser.parse_args()
    launch = json.loads(Path(args.launch).read_text(encoding="utf-8"))
    root = launch["root"]
    remote = root + "/core"
    names = {
        "controller_status": root + "/controller/status.json",
        "analysis_recovery": root + "/controller/analysis_recovery.json",
        "design": remote + "/design.json",
        "test_source_gate": remote + "/test_source_gate.json",
        "completion": remote + "/completion.json",
        "E1": remote + "/analysis/E1/paired_comparisons.json",
        "E3": remote + "/analysis/E3/paired_comparisons.json",
        "E1_manifest": remote + "/analysis/E1/analysis_manifest.json",
        "E3_manifest": remote + "/analysis/E3/analysis_manifest.json",
    }
    with connection(launch["host"], args.credentials_pdf) as client:
        with client.open_sftp() as sftp:
            exported = {name: read_remote_json(sftp, path) for name, path in names.items()}
            summaries = same_condition_rates(
                read_remote_csv(sftp, remote + "/analysis/E1/summary.csv"),
                read_remote_csv(sftp, remote + "/analysis/E3/summary.csv"),
            )
    status = exported["controller_status"]
    recovery = exported["analysis_recovery"]
    completion = exported["completion"]
    original_completed = status.get("state") == "completed" and status.get("gpu_released")
    analysis_recovered = (recovery.get("complete") and recovery.get("unit_exit_code") == 0
                          and status.get("gpu_released")
                          and not status.get("owned_gpu_pids_remaining"))
    if not (original_completed or analysis_recovered):
        raise RuntimeError(f"core service/recovery is not terminal with released GPUs: {status}, {recovery}")
    if not completion.get("complete"):
        raise RuntimeError(f"core experiment is incomplete: {completion}")
    selected = select_primary(exported["E1"], exported["E3"])
    payload = {
        "purpose": "outcome-blind export of the preregistered H1/H4 same-condition family",
        "automatic_hypothesis_acceptance": False,
        "controller_status": status,
        "analysis_recovery": recovery,
        "design": exported["design"],
        "test_source_gate": exported["test_source_gate"],
        "completion": completion,
        "primary_family": selected,
        "hypotheses": hypothesis_summary(selected),
        "same_condition_arm_rates": summaries,
        "conditional_transfer": conditional_transfer_results(exported["E1"], exported["E3"]),
        "analysis_manifests": {"E1": exported["E1_manifest"], "E3": exported["E3_manifest"]},
        "scope_caution": "Qwen3-8B, frozen workflow tasks and budget only; no non-inferiority margin was registered.",
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "REPORT.md").write_text(markdown_report(payload), encoding="utf-8")
    print(json.dumps({"out": str(out), "hypotheses": payload["hypotheses"],
                      "rows": completion}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
