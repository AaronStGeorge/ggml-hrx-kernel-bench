from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .common import (
    SCRIPT_BUILD_SUMMARY_SCHEMA,
    timestamp,
    write_json_file,
)


def _case_shape(case: dict[str, Any]) -> dict[str, Any]:
    shape = case.get("shape")
    if isinstance(shape, dict):
        return dict(shape)
    prepared_entry = case.get("prepared_entry")
    if not isinstance(prepared_entry, dict):
        return {}
    descriptor = case.get("descriptor")
    if isinstance(descriptor, dict):
        metadata = descriptor.get("metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get("shape"), dict):
            return dict(metadata["shape"])
    values = prepared_entry.get("case_values")
    return {"case_values": values} if isinstance(values, list) else {}


def _markdown_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        items = ", ".join(f"{key}={current}" for key, current in sorted(value.items()))
        return items
    return str(value)


def _route_summary(route_manifest: dict[str, Any]) -> dict[str, Any]:
    cases = []
    for case in route_manifest["cases"]:
        cases.append(
            {
                "case_id": case["case_id"],
                "benchmark_symbol": case["benchmark_symbol"],
                "descriptor_path": case["descriptor_path"],
                "shape": _case_shape(case),
                "shape_bucket": case.get("shape_bucket"),
                "estimated_flops": case.get("estimated_flops"),
                "descriptor_execution_digest": case.get("descriptor_execution_digest"),
                "run_case_dir_name": case.get("run_case_dir_name"),
            }
        )
    return {
        "op": route_manifest["op"],
        "route_id": route_manifest["route_id"],
        "implementation_id": route_manifest["implementation_id"],
        "root": route_manifest["root"],
        "kernel_source": route_manifest["kernel_source"],
        "normalized_kernel_source": route_manifest["normalized_kernel_source"],
        "source_content_hash": route_manifest.get("source_content_hash"),
        "case_count": route_manifest["case_count"],
        "cases": cases,
    }


def build_benchmark_route_summary(
    *,
    output_root: Path,
    catalog_root: Path,
    summary_output_dir: Path,
    routes: Sequence[dict[str, Any]],
    suite_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    route_summaries = [_route_summary(route) for route in routes]
    ops = sorted({route["op"] for route in route_summaries})
    return {
        "schema": SCRIPT_BUILD_SUMMARY_SCHEMA,
        "timestamp": timestamp(),
        "output_root": str(output_root),
        "catalog_root": str(catalog_root),
        "summary_output_dir": str(summary_output_dir),
        "suite": suite_metadata,
        "ops": ops,
        "route_count": len(route_summaries),
        "case_count": sum(int(route["case_count"]) for route in route_summaries),
        "routes": route_summaries,
    }


def benchmark_route_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Loom Benchmark Route Summary",
        "",
        f"- Suite: `{(summary.get('suite') or {}).get('name', '')}`",
        f"- Output root: `{summary.get('output_root')}`",
        f"- Operations: `{', '.join(summary.get('ops') or [])}`",
        f"- Routes: `{summary.get('route_count')}`",
        f"- Cases: `{summary.get('case_count')}`",
        "",
        "| op | case | route | kernel | root | shape |",
        "|---|---|---|---|---|---|",
    ]
    for route in summary["routes"]:
        for case in route["cases"]:
            lines.append(
                "| {op} | `{case}` | `{route}` | `{kernel}` | `{root}` | {shape} |".format(
                    op=route["op"],
                    case=case["case_id"],
                    route=route["route_id"],
                    kernel=route["normalized_kernel_source"],
                    root=route["root"],
                    shape=_markdown_value(case.get("shape")),
                )
            )
    lines.append("")
    return "\n".join(lines)


def write_benchmark_route_summary(
    *,
    output_root: Path,
    catalog_root: Path,
    summary_output_dir: Path,
    routes: Sequence[dict[str, Any]],
    suite_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    summary = build_benchmark_route_summary(
        output_root=output_root,
        catalog_root=catalog_root,
        summary_output_dir=summary_output_dir,
        routes=routes,
        suite_metadata=suite_metadata,
    )
    write_json_file(summary_output_dir / "benchmark-route-summary.json", summary)
    (summary_output_dir / "benchmark-route-summary.md").write_text(
        benchmark_route_summary_markdown(summary),
        encoding="utf-8",
    )
    return summary
