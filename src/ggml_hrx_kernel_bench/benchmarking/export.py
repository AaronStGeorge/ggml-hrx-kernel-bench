from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

from ggml_hrx_kernel_bench.required_tools import require_tool, resolve_tool
from ggml_hrx_kernel_bench.loom_execution_descriptor import load_descriptor

from .common import (
    SCRIPT_CASE_MANIFEST_SCHEMA,
    json_scalar as _json_scalar,
    load_json as _load_json,
    sha1_file as _sha1_file,
    timestamp as _timestamp,
    write_executable_script as _write_executable_script,
    write_json_file as _write_json_file,
)
from .discovery import (
    DescriptorCase,
    _binding_element_count,
    _expectation_element_count,
    _resolve_kernel_path,
    descriptor_execution_digest,
    estimate_case_flops,
    shape_bucket_for_case,
)
from .workbench import _copy_or_write_fixture, _dtype_element_type, _symbol_base


EXPORT_SCHEMA = "ggml_hrx_kernel_bench.kernel_export.v1"


def _rel(path: Path, *, base: Path) -> str:
    return str(path.resolve().relative_to(base.resolve()))


def _fixture_digest(path: Path) -> str | None:
    return _sha1_file(path)


def _copy_baseline_kernel(source: Path, output: Path) -> None:
    if not source.is_file():
        raise RuntimeError(f"kernel source does not exist: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)


def _binding_name(binding: dict[str, Any]) -> str:
    return _symbol_base(str(binding.get("name") or f"binding{binding['position']}"))


def _write_template_and_fixtures(
    *,
    case: DescriptorCase,
    export_root: Path,
) -> tuple[Path, str, str, list[dict[str, Any]]]:
    descriptor = case.descriptor
    descriptor_dir = case.descriptor_path.parent
    fixture_dir = export_root / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    case_symbol = f"@case_{_symbol_base(case.route_id)}_{case.execution_digest[:12]}"
    bench_symbol = f"@bench_{_symbol_base(case.route_id)}_{case.execution_digest[:12]}"
    values: dict[int, tuple[str, str]] = {}
    exported_bindings: list[dict[str, Any]] = []
    expectation_lines: list[str] = []
    lines: list[str] = ["{{LINKED_SOURCE}}", "", f"check.case public {case_symbol} {{"]

    for scalar in sorted(descriptor.get("scalars") or [], key=lambda item: item["position"]):
        name = _symbol_base(str(scalar.get("name") or f"scalar{scalar['position']}"))
        var = f"%{name}"
        values[int(scalar["position"])] = (var, str(scalar["dtype"]))
        lines.append(f"  {var} = check.literal value({_json_scalar(scalar['value'])}) : {scalar['dtype']}")

    for binding in sorted(descriptor["bindings"], key=lambda item: item["position"]):
        name = _binding_name(binding)
        var = f"%{name}"
        element_type = _dtype_element_type(str(binding["dtype"]))
        element_count = _binding_element_count(binding, descriptor, descriptor_dir)
        fixture_path = _copy_or_write_fixture(
            source=binding,
            descriptor_dir=descriptor_dir,
            fixture_dir=fixture_dir,
            name=name,
            descriptor_dtype=str(binding["dtype"]),
        )
        fixture_rel = _rel(fixture_path, base=export_root)
        values[int(binding["position"])] = (var, f"tensor<{element_count}x{element_type}>")
        lines.append(
            f"  {var} = check.file.read.npy path(\"{fixture_path.resolve()}\") : "
            f"tensor<{element_count}x{element_type}>"
        )
        exported: dict[str, Any] = {
            "position": binding.get("position"),
            "kind": binding.get("kind"),
            "name": name,
            "dtype": binding.get("dtype"),
            "element_count": element_count,
            "fixture": fixture_rel,
            "fixture_sha1": _fixture_digest(fixture_path),
        }

        expect = binding.get("expect")
        if isinstance(expect, dict):
            expected_name = f"expected_{name}"
            expected_var = f"%{expected_name}"
            expected_count = _expectation_element_count(binding, descriptor, descriptor_dir)
            expected_path = _copy_or_write_fixture(
                source=expect,
                descriptor_dir=descriptor_dir,
                fixture_dir=fixture_dir,
                name=expected_name,
                descriptor_dtype=str(binding["dtype"]),
            )
            expected_rel = _rel(expected_path, base=export_root)
            expectation_lines.append(
                f"  {expected_var} = check.file.read.npy path(\"{expected_path.resolve()}\") : "
                f"tensor<{expected_count}x{element_type}>"
            )
            mode = str(expect.get("mode") or "close")
            if mode == "close":
                atol = _json_scalar(expect.get("atol", 0.0))
                rtol = _json_scalar(expect.get("rtol", 0.0))
                expectation_lines.append(
                    f"  check.expect.close actual({var}) expected({expected_var}) "
                    f"atol({atol}) rtol({rtol}) nan(same) : tensor<{expected_count}x{element_type}>"
                )
            elif mode in {"equal", "bitwise"}:
                expectation_lines.append(
                    f"  check.expect.{mode} actual({var}) expected({expected_var}) : "
                    f"tensor<{expected_count}x{element_type}>"
                )
            else:
                raise RuntimeError(f"unsupported expectation mode {mode!r}")
            exported["expect"] = {
                "mode": mode,
                "atol": expect.get("atol"),
                "rtol": expect.get("rtol"),
                "element_count": expected_count,
                "fixture": expected_rel,
                "fixture_sha1": _fixture_digest(expected_path),
            }
        exported_bindings.append(exported)

    call_values = [values[position] for position in sorted(values)]
    args_text = ", ".join(value[0] for value in call_values)
    types_text = ", ".join(value[1] for value in call_values)
    lines.append(f"  func.call {descriptor['root']}({args_text}) : ({types_text})")
    lines.extend(expectation_lines)
    lines.extend(["  check.return", "}", "", f"check.benchmark<{case_symbol}> {bench_symbol}", ""])
    template_path = export_root / "benchmark.loom.template"
    template_path.write_text("\n".join(lines), encoding="utf-8")
    return template_path, case_symbol, bench_symbol, exported_bindings


def _runner_script_text() -> str:
    return r'''from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


RESULT_SCHEMA = "ggml_hrx_kernel_bench.kernel_export_result.v1"
SUMMARY_SCHEMA = "ggml_hrx_kernel_bench.kernel_export_summary.v1"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _benchmark_summary(path: Path, *, estimated_flops: int | None) -> dict[str, Any]:
    rows = _load_jsonl(path)
    benchmark_rows = [
        row for row in rows
        if isinstance(row.get("benchmark_result"), dict)
        or row.get("row") == "benchmark"
    ]
    compile_rows = [
        row for row in rows
        if isinstance(row.get("compile_report"), dict)
        or row.get("row") == "compile"
    ]
    result = benchmark_rows[-1].get("benchmark_result") if benchmark_rows else None
    result = result if isinstance(result, dict) else {}
    measurement = result.get("measurement") if isinstance(result.get("measurement"), dict) else {}
    timing = measurement.get("operation_timing_ns")
    timing = timing if isinstance(timing, dict) else {}
    compile_report = compile_rows[-1].get("compile_report") if compile_rows else None
    status = result.get("state") or ("ok" if benchmark_rows else "no_benchmark_rows")
    summary: dict[str, Any] = {
        "status": status,
        "raw_row_count": len(rows),
        "benchmark_row_count": len(benchmark_rows),
        "operation_timing_ns": timing,
        "compile_summary": compile_report,
    }
    p50 = timing.get("p50")
    mean = timing.get("mean")
    if estimated_flops is not None and isinstance(p50, int | float):
        summary["throughput"] = {
            "estimated_flops_per_operation": estimated_flops,
            "flops_per_second_from_p50_ns": estimated_flops * 1_000_000_000.0 / float(p50),
            **(
                {"flops_per_second_from_mean_ns": estimated_flops * 1_000_000_000.0 / float(mean)}
                if isinstance(mean, int | float)
                else {}
            ),
        }
    return summary


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_result_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _extra_args(values: list[str]) -> list[str]:
    return values[1:] if values and values[0] == "--" else values


def _run(args: argparse.Namespace) -> int:
    export_path = args.export.resolve()
    export = _load_json(export_path)
    export_root = export_path.parent
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    kernel = args.kernel.resolve()
    if not kernel.is_file():
        raise RuntimeError(f"kernel source does not exist: {kernel}")

    abi = export.get("abi") if isinstance(export.get("abi"), dict) else {}
    root_symbol = str(abi.get("root_symbol") or "")
    if root_symbol and root_symbol not in kernel.read_text(encoding="utf-8", errors="replace"):
        raise RuntimeError(f"candidate kernel does not contain required root symbol {root_symbol}")

    tools = export.get("tools") if isinstance(export.get("tools"), dict) else {}
    defaults = export.get("defaults") if isinstance(export.get("defaults"), dict) else {}
    benchmark_runner = args.benchmark_runner or str(tools.get("benchmark_runner") or "iree-benchmark-loom")
    loom_link = args.loom_link or str(tools.get("loom_link") or "loom-link")
    benchmark_device = args.device or str(defaults.get("benchmark_device") or "amdgpu")
    benchmark_measure = args.measure or str(defaults.get("benchmark_measure") or "dispatch_complete")
    benchmark_symbol = str(export["benchmark_symbol"])
    linked_path = out_dir / "linked.loom"
    benchmark_path = out_dir / "benchmark.loom"
    raw_output = out_dir / "benchmark-results.jsonl"
    (out_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    link_command = [
        loom_link,
        str(kernel),
        "--mode=link",
        "--to=text",
        "--require-resolved-config",
        f"--root={root_symbol}",
        f"--output={linked_path}",
    ]
    for key, value in sorted((abi.get("configs") or {}).items()):
        link_command.append(f"--config={key}={value}")
    link_result = subprocess.run(
        link_command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    (out_dir / "link-command.txt").write_text("\n".join(link_command) + "\n", encoding="utf-8")
    (out_dir / "link-stdout.txt").write_text(link_result.stdout, encoding="utf-8")
    (out_dir / "link-stderr.txt").write_text(link_result.stderr, encoding="utf-8")
    if link_result.returncode != 0:
        (out_dir / "returncode.txt").write_text(str(link_result.returncode) + "\n", encoding="utf-8")
        return link_result.returncode

    template = (export_root / str(export["template"])).read_text(encoding="utf-8")
    benchmark_path.write_text(
        template.replace("{{LINKED_SOURCE}}", linked_path.read_text(encoding="utf-8").rstrip()),
        encoding="utf-8",
    )
    command = [
        benchmark_runner,
        str(benchmark_path),
        f"--device={benchmark_device}",
        f"--measure={benchmark_measure}",
        f"--benchmark={benchmark_symbol}",
        "--output-format=jsonl",
        f"--output={raw_output}",
        f"--artifact-bundle-dir={out_dir / 'artifacts'}",
        *_extra_args(list(args.benchmark_args or [])),
    ]
    (out_dir / "command.txt").write_text("\n".join(command) + "\n", encoding="utf-8")
    result = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (out_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (out_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")
    (out_dir / "returncode.txt").write_text(str(result.returncode) + "\n", encoding="utf-8")

    estimated_flops = export.get("estimated_flops")
    summary = _benchmark_summary(
        raw_output,
        estimated_flops=estimated_flops if isinstance(estimated_flops, int) else None,
    )
    status = "pass" if result.returncode == 0 and summary.get("status") == "ok" else "failed"
    row = {
        "schema": RESULT_SCHEMA,
        "candidate_name": args.candidate_name or kernel.stem,
        "status": status,
        "process_returncode": result.returncode,
        "suite": export.get("suite"),
        "op": export.get("op"),
        "route_id": export.get("route_id"),
        "case_id": export.get("case_id"),
        "target_arch": export.get("target_arch") or export.get("gpu_arch"),
        "gpu_arch": export.get("gpu_arch") or export.get("target_arch"),
        "kernel_source": str(kernel),
        "root": root_symbol,
        "benchmark_symbol": benchmark_symbol,
        "benchmark_output_path": str(raw_output),
        "benchmark_summary": summary,
        "estimated_flops": estimated_flops,
        "shape_bucket": export.get("shape_bucket") if isinstance(export.get("shape_bucket"), dict) else {},
        "command": command,
        "stdout": _read_text(out_dir / "stdout.txt"),
        "stderr": _read_text(out_dir / "stderr.txt"),
    }
    _write_result_jsonl(out_dir / "results.jsonl", row)
    _write_json(
        out_dir / "summary.json",
        {
            "schema": SUMMARY_SCHEMA,
            "candidate_name": row["candidate_name"],
            "status": status,
            "case_count": 1,
            "passed_count": 1 if status == "pass" else 0,
            "failed_count": 0 if status == "pass" else 1,
            "result_path": str(out_dir / "results.jsonl"),
            "case_id": export.get("case_id"),
            "route_id": export.get("route_id"),
        },
    )
    return 0 if status == "pass" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one exported Loom kernel benchmark case.")
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidate-name")
    parser.add_argument("--benchmark-runner")
    parser.add_argument("--loom-link")
    parser.add_argument("--device")
    parser.add_argument("--measure")
    parser.add_argument("benchmark_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        return _run(args)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _run_script_text() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

EXPORT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$EXPORT_DIR/runner.py" --export "$EXPORT_DIR/export.json" "$@"
"""


def _metadata_string(descriptor: dict[str, Any], key: str, default: str = "") -> str:
    metadata = descriptor.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get(key), str):
        return str(metadata[key])
    return default


def _source_hash(path: Path) -> str | None:
    return _sha1_file(path)


def _descriptor_case_from_manifest(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    repo_root_override: Path | None,
) -> tuple[DescriptorCase, Path, dict[str, Any], str | None]:
    descriptor_path = Path(str(manifest["descriptor_path"])).resolve()
    descriptor = load_descriptor(descriptor_path)
    repo_root = (
        repo_root_override.resolve()
        if repo_root_override
        else Path(str(manifest.get("repo_root") or descriptor_path.parent)).resolve()
    )
    kernel_path = _resolve_kernel_path(descriptor["kernel"], repo_root=repo_root)
    prepared_entry = manifest.get("prepared_entry")
    prepared_entry = prepared_entry if isinstance(prepared_entry, dict) else {}
    normalized_kernel_source = str(manifest.get("normalized_kernel_source") or descriptor["kernel"])
    source_hash = (
        str(manifest["source_content_hash"])
        if isinstance(manifest.get("source_content_hash"), str)
        else _source_hash(kernel_path)
    )
    execution_digest = (
        str(manifest["descriptor_execution_digest"])
        if isinstance(manifest.get("descriptor_execution_digest"), str)
        else descriptor_execution_digest(
            descriptor,
            descriptor_path=descriptor_path,
            normalized_kernel_source=normalized_kernel_source,
        )
    )
    case = DescriptorCase(
        op=str(manifest.get("op") or "unknown"),
        run_manifest_path=Path(str(manifest.get("run_manifest_path") or manifest_path)).resolve(),
        descriptor_path=descriptor_path,
        descriptor=descriptor,
        prepared_entry=prepared_entry,
        normalized_kernel_source=normalized_kernel_source,
        source_content_hash=source_hash,
        implementation_id=str(manifest.get("implementation_id") or ""),
        execution_digest=execution_digest,
    )
    defaults = manifest.get("defaults")
    defaults = defaults if isinstance(defaults, dict) else {}
    preparation = manifest.get("preparation")
    preparation = preparation if isinstance(preparation, dict) else {}
    loom_link = None
    link_command = preparation.get("link_command")
    if isinstance(link_command, list) and link_command and isinstance(link_command[0], str):
        loom_link = link_command[0]
    return case, repo_root, defaults, loom_link


def _descriptor_case_from_descriptor(
    *,
    descriptor_path: Path,
    descriptor: dict[str, Any],
    repo_root_override: Path | None,
) -> tuple[DescriptorCase, Path, dict[str, Any], str | None]:
    repo_root = repo_root_override.resolve() if repo_root_override else descriptor_path.parent.resolve()
    kernel_path = _resolve_kernel_path(descriptor["kernel"], repo_root=repo_root)
    route_id = _metadata_string(descriptor, "route_id")
    case_id = _metadata_string(descriptor, "case_id", descriptor_path.stem)
    normalized_kernel_source = str(descriptor["kernel"])
    execution_digest = descriptor_execution_digest(
        descriptor,
        descriptor_path=descriptor_path,
        normalized_kernel_source=normalized_kernel_source,
    )
    case = DescriptorCase(
        op=_metadata_string(descriptor, "op", "unknown"),
        run_manifest_path=descriptor_path,
        descriptor_path=descriptor_path,
        descriptor=descriptor,
        prepared_entry={"case_id": case_id, "route_id": route_id},
        normalized_kernel_source=normalized_kernel_source,
        source_content_hash=_source_hash(kernel_path),
        implementation_id="",
        execution_digest=execution_digest,
    )
    return case, repo_root, {}, None


def _load_export_case(args: argparse.Namespace) -> tuple[DescriptorCase, Path, dict[str, Any], str | None]:
    case_path = args.case.resolve()
    payload = _load_json(case_path)
    repo_root_override = args.repo_root.resolve() if args.repo_root else None
    if payload.get("schema") == SCRIPT_CASE_MANIFEST_SCHEMA:
        return _descriptor_case_from_manifest(
            manifest_path=case_path,
            manifest=payload,
            repo_root_override=repo_root_override,
        )
    return _descriptor_case_from_descriptor(
        descriptor_path=case_path,
        descriptor=payload,
        repo_root_override=repo_root_override,
    )


def command_export(args: argparse.Namespace) -> int:
    out_root = args.out.resolve()
    case, repo_root, case_defaults, case_loom_link = _load_export_case(args)
    kernel_path = (
        args.kernel.resolve()
        if args.kernel
        else _resolve_kernel_path(case.descriptor["kernel"], repo_root=repo_root)
    )
    baseline_path = out_root / "baseline.loom"
    benchmark_runner = (
        args.benchmark_runner
        or case_defaults.get("benchmark_runner")
        or resolve_tool("iree-benchmark-loom", tool_dir=args.tool_dir)
    )
    benchmark_runner = benchmark_runner or "iree-benchmark-loom"
    loom_link = args.loom_link or case_loom_link or require_tool("loom-link", tool_dir=args.tool_dir)
    benchmark_device = args.benchmark_device or case_defaults.get("benchmark_device") or "amdgpu"
    benchmark_measure = args.benchmark_measure or case_defaults.get("benchmark_measure") or "dispatch_complete"
    target_arch = getattr(args, "target_arch", None) or getattr(args, "gpu_arch", None)
    if not target_arch:
        raise RuntimeError("target architecture is required; pass --target-arch <arch>")

    out_root.mkdir(parents=True, exist_ok=True)
    _copy_baseline_kernel(kernel_path, baseline_path)
    template_path, case_symbol, benchmark_symbol, bindings = _write_template_and_fixtures(
        case=case,
        export_root=out_root,
    )
    flop_estimate = estimate_case_flops(case)
    estimated_flops = flop_estimate.get("estimated_flops")
    descriptor_copy = out_root / "descriptor.json"
    _write_json_file(descriptor_copy, case.descriptor)
    export = {
        "schema": EXPORT_SCHEMA,
        "timestamp": _timestamp(),
        "op": case.op,
        "route_id": case.route_id,
        "case_id": case.case_id,
        "target_arch": target_arch,
        "gpu_arch": target_arch,
        "repo_root": str(repo_root),
        "descriptor": _rel(descriptor_copy, base=out_root),
        "template": _rel(template_path, base=out_root),
        "baseline_kernel": _rel(baseline_path, base=out_root),
        "baseline_kernel_sha1": _sha1_file(baseline_path),
        "case_symbol": case_symbol,
        "benchmark_symbol": benchmark_symbol,
        "estimated_flops": estimated_flops if isinstance(estimated_flops, int) else None,
        "flop_estimate": flop_estimate,
        "shape_bucket": shape_bucket_for_case(case, flop_estimate=flop_estimate),
        "abi": {
            "root_symbol": case.root,
            "configs": {
                key: _json_scalar(value) for key, value in sorted((case.descriptor.get("configs") or {}).items())
            },
            "workgroup_count": case.descriptor.get("workgroup_count"),
            "bindings": bindings,
        },
        "tools": {
            "benchmark_runner": benchmark_runner,
            "loom_link": loom_link,
        },
        "defaults": {
            "benchmark_device": benchmark_device,
            "benchmark_measure": benchmark_measure,
        },
        "source": {
            "descriptor_path": str(case.descriptor_path),
            "run_manifest_path": str(case.run_manifest_path),
            "route_id": case.route_id,
            "kernel": str(kernel_path),
            "kernel_sha1": _sha1_file(kernel_path),
            "descriptor_execution_digest": case.execution_digest,
        },
    }
    _write_json_file(out_root / "export.json", export)
    (out_root / "runner.py").write_text(_runner_script_text(), encoding="utf-8")
    _write_executable_script(out_root / "run.sh", _run_script_text())
    if args.archive:
        archive_dir = out_root / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        _write_json_file(archive_dir / "prepared-entry.json", case.prepared_entry)
    print(json.dumps(export, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export one built Loom benchmark case for run-many optimization.")
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kernel", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument(
        "--target-arch",
        dest="target_arch",
        metavar="ARCH",
        required=True,
        help="Required target GPU architecture, for example gfx1100.",
    )
    parser.add_argument("--tool-dir")
    parser.add_argument("--benchmark-runner")
    parser.add_argument("--benchmark-device")
    parser.add_argument("--benchmark-measure")
    parser.add_argument("--loom-link")
    parser.add_argument("--archive", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return command_export(args)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
