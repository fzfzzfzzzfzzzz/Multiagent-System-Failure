"""Explain terminal core failures from saved result rows and full episodes.

This is a post hoc diagnostic.  It never changes the registered H1/H4 result
and deliberately labels output as exploratory.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


BOOLEAN_METRICS = (
    "first_success", "final_success", "abstained", "strict_composition_failure",
    "preventive_action", "unnecessary_wait", "any_new_error", "any_tool_error",
    "budget_ceiling", "target_avoided_but_failed", "recurred_then_repaired",
)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_action(text):
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return {"tool": "invalid_json", "args": {}}


def budget_ceiling_roles(transcripts, turns):
    grouped = defaultdict(list)
    for row in transcripts:
        grouped[(row["role"], row["round"])].append(row)
    roles = set()
    for (role, _), rows in grouped.items():
        if len(rows) >= turns and parse_action(rows[-1].get("output", "")).get("tool") != "finish":
            roles.add(role)
    return sorted(roles)


def outcome_category(row):
    if row["first_success"]:
        return "first_success"
    if row["final_success"] and row["recurrence"]:
        return "recurred_then_repaired"
    if row["final_success"]:
        return "late_success_without_target_recurrence"
    if row["abstained"]:
        return "abstained"
    if row["recurrence"]:
        return "target_recurrence_failure"
    return "other_failure_without_target_recurrence"


def target_recurrence_event(row, events):
    if not row.get("recurrence"):
        return None
    mechanism = row["mechanism"]
    if mechanism == "F2":
        for event in events:
            analysis = event.get("observation", {}).get("analysis", {})
            if event.get("action", {}).get("tool") == "calculate" and analysis.get("artifact_status_seen") != "confirmed":
                return {"role": event["role"], "tool": "calculate", "reason": "calculated_before_confirmation"}
    if mechanism == "F3":
        latest = 1
        for event in events:
            observation = event.get("observation", {})
            latest = max(latest, int(observation.get("source_revision_now", latest)))
            analysis = observation.get("analysis", {})
            if event.get("action", {}).get("tool") == "calculate" and analysis.get("version") != latest:
                return {"role": event["role"], "tool": "calculate", "reason": "calculated_stale_version"}
    if mechanism == "F1":
        for event in events:
            if event.get("action", {}).get("tool") == "calculate" and not event.get("observation", {}).get("error"):
                return {"role": event["role"], "tool": "calculate", "reason": "incorrect_transformation_detected_at_grading"}
    if mechanism == "F4":
        allocations = []
        for event in events:
            if event.get("action", {}).get("tool") == "allocate" and not event.get("observation", {}).get("error"):
                allocations.append(event)
                if len(allocations) == 2:
                    return {"role": event["role"], "tool": "allocate", "reason": "joint_conflict_formed"}
    for event in events:
        if event.get("action", {}).get("tool") == "commit":
            return {"role": event["role"], "tool": "commit", "reason": f"{mechanism}_failed_at_commit"}
    return {"role": "unknown", "tool": "none", "reason": f"{mechanism}_recurrence_without_commit"}


def derive_run(row, episode):
    events = episode.get("events", [])
    errors = [event for event in events if event.get("observation", {}).get("error")]
    action_counts = Counter((event["role"], event.get("action", {}).get("tool", "missing")) for event in events)
    checks = row.get("commit_checks", [])
    first_failed = sorted(key for key, value in (checks[0] if checks else {}).items() if not value)
    final_failed = sorted(key for key, value in (checks[-1] if checks else {}).items() if not value)
    ceiling = budget_ceiling_roles(episode.get("transcripts", []), int(episode.get("turns_per_role", 0)))
    target_event = target_recurrence_event(row, events)
    calls_by_role = Counter(item.get("role") for item in row.get("cost", []) if item.get("phase") == "execution")
    result = {
        "run_id": row["run_id"], "experiment": row["experiment"], "family_id": row["family_id"],
        "mechanism": row["mechanism"], "kind": row["kind"], "method": row["method"],
        "seed": row["seed"], "first_success": bool(row["first_success"]),
        "final_success": bool(row["final_success"]), "abstained": bool(row["abstained"]),
        "opportunity": bool(row["opportunity"]), "recurrence": bool(row["recurrence"]),
        "strict_composition_failure": bool(row["strict_composition_failure"]),
        "preventive_action": bool(row["preventive_action"]),
        "unnecessary_wait": bool(row["unnecessary_wait"]),
        "any_new_error": bool(row.get("new_errors")), "new_errors": row.get("new_errors", []),
        "outcome_category": outcome_category(row),
        "target_avoided_but_failed": bool(row["opportunity"] and not row["recurrence"] and not row["final_success"]),
        "recurred_then_repaired": bool(row["recurrence"] and row["final_success"]),
        "commits": len(checks), "first_failed_checks": first_failed, "final_failed_checks": final_failed,
        "any_tool_error": bool(errors), "tool_error_count": len(errors),
        "first_tool_error_role": errors[0]["role"] if errors else None,
        "first_tool_error_tool": errors[0].get("action", {}).get("tool") if errors else None,
        "first_tool_error": errors[0].get("observation", {}).get("error") if errors else None,
        "target_event_role": target_event["role"] if target_event else None,
        "target_event_tool": target_event["tool"] if target_event else None,
        "target_event_reason": target_event["reason"] if target_event else None,
        "budget_ceiling": bool(ceiling), "budget_ceiling_roles": ceiling,
        "model_calls": sum(calls_by_role.values()), "calls_by_role": dict(calls_by_role),
        "input_tokens": sum(item.get("input_tokens", 0) for item in row.get("cost", [])),
        "output_tokens": sum(item.get("output_tokens", 0) for item in row.get("cost", [])),
        "action_counts": {f"{role}:{tool}": count for (role, tool), count in action_counts.items()},
    }
    return result


def aggregate(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for values, items in sorted(groups.items()):
        record = dict(zip(keys, values))
        n = len(items)
        opportunities = sum(row["opportunity"] for row in items)
        record.update({"n": n, "opportunities": opportunities,
                       "recurrence_count": sum(row["recurrence"] for row in items),
                       "recurrence_rate": (sum(row["recurrence"] for row in items) / opportunities) if opportunities else None,
                       "mean_commits": sum(row["commits"] for row in items) / n,
                       "mean_model_calls": sum(row["model_calls"] for row in items) / n,
                       "mean_input_tokens": sum(row["input_tokens"] for row in items) / n})
        for metric in BOOLEAN_METRICS:
            record[metric + "_rate"] = sum(bool(row[metric]) for row in items) / n
        for category in ("first_success", "recurred_then_repaired", "late_success_without_target_recurrence",
                         "abstained", "target_recurrence_failure", "other_failure_without_target_recurrence"):
            record["outcome_" + category] = sum(row["outcome_category"] == category for row in items)
        output.append(record)
    return output


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def report(rows, summary):
    total = len(rows)
    opportunities = sum(row["opportunity"] for row in rows)
    categories = Counter(row["outcome_category"] for row in rows)
    failed_checks = Counter(check for row in rows for check in row["first_failed_checks"])
    lines = [
        "# 核心实验失败构成诊断（探索性）", "",
        "本报告只解释已冻结 H1/H4 实验为何没有首次成功，不改变已注册主结论。", "",
        f"共分析 {total} 个真实模型回合及对应完整 episode；首次成功 {sum(r['first_success'] for r in rows)}，"
        f"最终成功 {sum(r['final_success'] for r in rows)}，未提交 {sum(r['abstained'] for r in rows)}。",
        f"在 {opportunities} 个存在目标错误机会的回合中，复发 {sum(r['recurrence'] for r in rows)}。", "",
        "## 互斥结果构成", "", "| 结果类别 | 回合数 | 比例 |", "|---|---:|---:|",
    ]
    for category, count in categories.most_common():
        lines.append(f"| {category} | {count} | {count/total:.3f} |")
    lines += ["", "## 第一次提交失败的检查项", "", "| 检查项 | 回合数 |", "|---|---:|"]
    for check, count in failed_checks.most_common():
        lines.append(f"| {check} | {count} |")
    lines += ["", "## 按目标条件汇总", "",
              "| 实验 | 条件 | 方法 | n | 首次成功 | 最终成功 | 未提交 | 目标复发 | 避免目标错误但仍失败 | 触及调用上限 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(
            f"| {row['experiment']} | {row['kind']} | {row['method']} | {row['n']} | "
            f"{row['first_success_rate']:.3f} | {row['final_success_rate']:.3f} | {row['abstained_rate']:.3f} | "
            f"{row['recurrence_rate'] if row['recurrence_rate'] is not None else 'NA'} | "
            f"{row['target_avoided_but_failed_rate']:.3f} | {row['budget_ceiling_rate']:.3f} |")
    lines += ["", "`budget_ceiling` 表示至少一个角色在某轮用完全部调用槽且没有主动 finish；它是预算紧张迹象，不自动证明失败由预算造成。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Core directory containing test_E1 and test_E3")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    derived = []
    missing = []
    for experiment in ("E1", "E3"):
        directory = root / f"test_{experiment}"
        for row in read_jsonl(directory / "results.jsonl"):
            path = directory / "episodes" / f"{row['run_id']}.json"
            if not path.exists():
                missing.append(row["run_id"])
                continue
            derived.append(derive_run(row, json.loads(path.read_text(encoding="utf-8"))))
    with (out / "run_diagnostics.jsonl").open("w", encoding="utf-8") as handle:
        for row in derived:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = aggregate(derived, ("experiment", "kind", "method"))
    detailed = aggregate(derived, ("experiment", "mechanism", "kind", "method"))
    family = aggregate(derived, ("experiment", "family_id", "kind", "method"))
    write_csv(out / "summary.csv", summary)
    write_csv(out / "mechanism_summary.csv", detailed)
    write_csv(out / "family_summary.csv", family)
    action_rows = []
    for keys, items in sorted(defaultdict(list, {
        key: [row for row in derived if (row["experiment"], row["mechanism"], row["kind"], row["method"]) == key]
        for key in {(row["experiment"], row["mechanism"], row["kind"], row["method"]) for row in derived}
    }).items()):
        actions = Counter()
        present = Counter()
        for row in items:
            for action, count in row["action_counts"].items():
                actions[action] += count
                present[action] += count > 0
        for action in sorted(actions):
            role, tool = action.split(":", 1)
            action_rows.append({"experiment": keys[0], "mechanism": keys[1], "kind": keys[2], "method": keys[3],
                                "role": role, "tool": tool, "n": len(items),
                                "run_rate": present[action] / len(items), "mean_count": actions[action] / len(items)})
    write_csv(out / "action_rates.csv", action_rows)
    error_rows = []
    errors = Counter()
    for row in derived:
        if row["first_tool_error_role"]:
            errors[(row["experiment"], row["mechanism"], row["kind"], row["method"], "tool_error",
                    row["first_tool_error_role"], row["first_tool_error_tool"], row["first_tool_error"])] += 1
        if row["target_event_role"]:
            errors[(row["experiment"], row["mechanism"], row["kind"], row["method"], "target_recurrence",
                    row["target_event_role"], row["target_event_tool"], row["target_event_reason"])] += 1
    for key, count in sorted(errors.items()):
        error_rows.append(dict(zip(("experiment", "mechanism", "kind", "method", "error_type", "role", "tool", "reason"), key), count=count))
    write_csv(out / "first_error_rates.csv", error_rows)
    manifest = {"exploratory": True, "rows": len(derived), "missing_episodes": missing,
                "experiments": sorted({row["experiment"] for row in derived}),
                "warning": "diagnostic output cannot revise the preregistered H1/H4 result"}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "REPORT.md").write_text(report(derived, summary), encoding="utf-8")
    print(json.dumps({"out": str(out), **manifest}, ensure_ascii=False, indent=2))
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
