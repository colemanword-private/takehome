"""Condense a campaign's raw reports into small, committable evidence.

Reads the gitignored raw load/quality reports for one run ID and emits a
compact JSON summary plus a rendered markdown table under evidence/. Only
numeric aggregates, configuration, and per-source SHA-256 hashes are copied:
no prompts, generated text, error messages, or project identifiers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

# Google list prices for gemini-2.5-flash, USD per million tokens. Thinking
# tokens are billed at the output rate, so observed output+thought totals
# price directly. Verified 2026-08-17; failed attempts that returned no usage
# metadata are absent from the totals, so every figure is a lower bound.
PRICING = {
    "as_of": "2026-08-17",
    "input_usd_per_million": 0.30,
    "output_usd_per_million": 2.50,
}

_LOAD_FIELDS = (
    "offered_rps",
    "realized_arrival_rps",
    "arrival_overlap_fraction",
    "requests",
    "successes",
    "failures",
    "failure_modes",
    "retries",
    "retries_by_status",
    "retry_exhaustions",
    "max_active_requests",
    "observed_input_tokens",
    "observed_output_tokens",
    "observed_thought_tokens",
)
_PROVIDER_FIELDS = (
    "max_output_tokens",
    "thinking_budget",
    "max_retries",
    "timeout_seconds",
    "request_deadline_seconds",
)


def _usd(input_tokens: float, output_tokens: float) -> float:
    return round(
        input_tokens / 1e6 * PRICING["input_usd_per_million"]
        + output_tokens / 1e6 * PRICING["output_usd_per_million"],
        4,
    )


def _load_row(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"file": path.name, "sha256": _sha256(path)}
    for field in _LOAD_FIELDS:
        if report.get(field) is not None:
            row[field] = report[field]
    for key in ("service_latency_ms", "end_to_end_latency_ms", "scheduler_lag_ms"):
        row[key] = report[key]
    provider = report.get("provider", {})
    for key in _PROVIDER_FIELDS:
        row[key] = provider.get(key)
    row["config"] = report.get("config")
    if report.get("shards") is not None:
        row["shards"] = len(report["shards"])
        row["shard_scheduler_lag_p95_ms"] = max(
            shard["scheduler_lag_p95_ms"] for shard in report["shards"]
        )
        row["shard_max_active_requests"] = max(
            shard["max_active_requests"] for shard in report["shards"]
        )
        # Hash the raw per-shard reports too, so pooled numbers stay
        # traceable to the exact sample sets they were merged from. The set
        # on disk must match the merged report exactly: a stray shard file
        # from another run must never be presented as campaign evidence.
        shards_dir = path.parent / f"{path.stem}-shards"
        shard_hashes = {
            shard_path.name: _sha256(shard_path)
            for shard_path in sorted(shards_dir.glob("*.json"))
        }
        expected = {
            f"shard-{index}.json" for index in range(len(report["shards"]))
        }
        if set(shard_hashes) != expected:
            raise ValueError(
                f"{path.name}: shard files on disk ({sorted(shard_hashes)}) "
                f"do not match the merged report's {len(report['shards'])} "
                "shards; refusing to summarize a mixed campaign"
            )
        row["shard_report_sha256"] = shard_hashes
    row["breaches"] = report.get("acceptance", {}).get("breaches", [])
    return row


def _quality_row(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    provider = report.get("provider", {})
    return {
        "file": path.name,
        "sha256": _sha256(path),
        "thinking_budget": provider.get("thinking_budget"),
        "max_output_tokens": provider.get("max_output_tokens"),
        "cases": report["cases"],
        "passed": report["passed"],
        "failed": report["failed"],
        "elapsed_seconds": round(report["elapsed_seconds"], 2),
        "observed_input_tokens": report["observed_input_tokens"],
        "observed_output_tokens": report["observed_output_tokens"],
        "observed_thought_tokens": report["observed_thought_tokens"],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(directory: Path, builder: Any) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(directory.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        rows.append(builder(path, report))
    return rows


def _cost_summary(
    load_rows: list[dict[str, Any]], quality_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    cost: dict[str, Any] = {"pricing": PRICING}

    operating = [
        row for row in load_rows if row["file"].startswith("operating-point")
    ]
    if operating:
        requests = sum(row["requests"] for row in operating)
        usd = _usd(
            sum(row["observed_input_tokens"] for row in operating),
            sum(row["observed_output_tokens"] for row in operating),
        )
        rps = operating[0]["offered_rps"]
        cost["operating_point"] = {
            "offered_rps": rps,
            "usd_per_1k_requests": round(usd / requests * 1_000, 4),
            "usd_per_day_at_offered_rps": round(
                usd / requests * rps * 86_400, 2
            ),
        }

    def cell_cost(thinking_on: bool) -> dict[str, Any] | None:
        rows = [
            row
            for row in quality_rows
            if (row["thinking_budget"] is None) == thinking_on
        ]
        if not rows:
            return None
        usd = _usd(
            sum(row["observed_input_tokens"] for row in rows),
            sum(row["observed_output_tokens"] for row in rows),
        )
        return {"runs": len(rows), "usd_per_run": round(usd / len(rows), 4)}

    thinking_default = cell_cost(thinking_on=True)
    thinking_off = cell_cost(thinking_on=False)
    if thinking_default and thinking_off:
        cost["quality_thinking_comparison"] = {
            "default_thinking": thinking_default,
            "thinking_disabled": thinking_off,
            "cost_multiplier": round(
                thinking_default["usd_per_run"] / thinking_off["usd_per_run"], 1
            ),
        }

    rows = load_rows + quality_rows
    cost["campaign_total"] = {
        "requests": sum(row.get("requests", row.get("cases", 0)) for row in rows),
        "input_tokens": sum(row["observed_input_tokens"] for row in rows),
        "output_tokens": sum(row["observed_output_tokens"] for row in rows),
        "usd": _usd(
            sum(row["observed_input_tokens"] for row in rows),
            sum(row["observed_output_tokens"] for row in rows),
        ),
    }
    return cost


def build_summary(project_dir: Path, run_id: str, model: str) -> dict[str, Any]:
    load_rows = _rows(
        project_dir / "load-results" / "gemini" / model / run_id, _load_row
    )
    quality_rows = _rows(
        project_dir / "eval-results" / "gemini" / model / run_id, _quality_row
    )
    synthetic_rows = _rows(
        project_dir / "load-results" / "synthetic" / "local" / run_id, _load_row
    )
    if not load_rows and not quality_rows:
        raise ValueError(f"no reports found for run {run_id}")
    load_rows.sort(key=lambda row: (row.get("offered_rps", 0), row["file"]))
    return {
        "run_id": run_id,
        "model": model,
        "load_reports": load_rows,
        "quality_reports": quality_rows,
        "synthetic_reports": synthetic_rows,
        "cost": _cost_summary(load_rows, quality_rows),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# Evidence summary: {summary['model']} run {summary['run_id']}",
        "",
        "## Load reports",
        "",
        "| Report | RPS | Realized | Requests | Fail | Service p50/p95/p99 ms"
        " | Sched p95 ms | Retries | Breaches |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["load_reports"] + summary["synthetic_reports"]:
        service = row["service_latency_ms"]
        lines.append(
            f"| {row['file']} | {row.get('offered_rps', 0):g}"
            f" | {row.get('realized_arrival_rps', 0):.2f}"
            f" | {row.get('requests', 0)} | {row.get('failures', 0)}"
            f" | {service['p50']:.0f}/{service['p95']:.0f}/{service['p99']:.0f}"
            f" | {row['scheduler_lag_ms']['p95']:.2f}"
            f" | {row.get('retries', 0)}"
            f" | {', '.join(row['breaches']) or 'none'} |"
        )
    if summary["quality_reports"]:
        lines += [
            "",
            "## Quality reports",
            "",
            "| Report | Thinking | Cap | Passed | Output tokens | Thought tokens |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
        for row in summary["quality_reports"]:
            thinking = row["thinking_budget"]
            lines.append(
                f"| {row['file']} | {'default' if thinking is None else thinking}"
                f" | {row['max_output_tokens'] or 'none'}"
                f" | {row['passed']}/{row['cases']}"
                f" | {row['observed_output_tokens']}"
                f" | {row['observed_thought_tokens']} |"
            )
    cost = summary["cost"]
    lines += ["", "## Cost (list price, lower bound)", ""]
    if "operating_point" in cost:
        point = cost["operating_point"]
        lines.append(
            f"- Operating point ({point['offered_rps']:g} RPS):"
            f" ${point['usd_per_1k_requests']}/1k requests,"
            f" ${point['usd_per_day_at_offered_rps']}/day sustained"
        )
    if "quality_thinking_comparison" in cost:
        compare = cost["quality_thinking_comparison"]
        lines.append(
            f"- Quality runs: default thinking"
            f" ${compare['default_thinking']['usd_per_run']}/run vs"
            f" ${compare['thinking_disabled']['usd_per_run']}/run disabled"
            f" ({compare['cost_multiplier']}x)"
        )
    total = cost["campaign_total"]
    lines.append(
        f"- Campaign total: {total['requests']} requests,"
        f" {total['input_tokens']} in / {total['output_tokens']} out tokens,"
        f" ${total['usd']}"
    )
    lines += [
        "",
        f"Prices as of {PRICING['as_of']}:"
        f" ${PRICING['input_usd_per_million']}/M input,"
        f" ${PRICING['output_usd_per_million']}/M output"
        " (thinking tokens billed as output).",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    summary = build_summary(args.project_dir, args.run_id, args.model)
    out_dir = args.project_dir / "evidence" / "gemini" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.run_id}-summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path = out_dir / f"{args.run_id}-summary.md"
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    print(f"wrote {json_path} and {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
