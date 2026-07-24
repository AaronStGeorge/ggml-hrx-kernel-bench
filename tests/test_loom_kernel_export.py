from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ggml_hrx_kernel_bench.benchmarking import export  # noqa: E402


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_executable(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)
    return path


def _write_prepared_case(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo_root = tmp_path / "repo"
    asset_root = repo_root / "build" / "generated" / "assets"
    kernel = asset_root / "kernels" / "v2" / "add" / "f32.loom"
    kernel.parent.mkdir(parents=True, exist_ok=True)
    kernel.write_text("kernel.def @add_f32() {} launch(%src0: buffer) { kernel.return }\n", encoding="utf-8")

    descriptor_root = tmp_path / "descriptors" / "ADD"
    for fixture_name in ("src0.npy", "src1.npy", "dst_init.npy", "expected.npy"):
        fixture_path = descriptor_root / "fixtures" / fixture_name
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        fixture_path.write_bytes(fixture_name.encode("utf-8"))
    descriptor_path = descriptor_root / "case.json"
    _write_json(
        descriptor_path,
        {
            "schema": "ggml_hrx_kernel_bench.loom_execution_descriptor.v1",
            "kernel": str(kernel),
            "root": "@add_f32",
            "target": "gfx1100",
            "workgroup_count": [1, 1, 1],
            "configs": {
                "@shape.pointwise.ne0": "4",
                "@tuning.pointwise.workgroup_size": "256",
            },
            "scalars": [],
            "bindings": [
                {
                    "position": 0,
                    "kind": "input",
                    "dtype": "f32",
                    "name": "src0",
                    "path": "fixtures/src0.npy",
                },
                {
                    "position": 1,
                    "kind": "input",
                    "dtype": "f32",
                    "name": "src1",
                    "path": "fixtures/src1.npy",
                },
                {
                    "position": 2,
                    "kind": "output",
                    "dtype": "f32",
                    "name": "dst",
                    "path": "fixtures/dst_init.npy",
                    "expect": {
                        "mode": "close",
                        "path": "fixtures/expected.npy",
                        "atol": 1e-6,
                        "rtol": 1e-6,
                    },
                },
            ],
            "metadata": {
                "case_id": "d04",
                "case_values": [4],
                "shape": {"d0": 4},
                "route_id": "add_f32_contiguous_4d",
                "candidate_id": "add_f32_contiguous_4d_candidate",
                "element_counts": {
                    "src0": 4,
                    "src1": 4,
                    "dst": 4,
                },
                "oracle_array_element_counts": {
                    "src0": 4,
                    "src1": 4,
                    "dst_init": 4,
                    "expected": 4,
                },
            },
        },
    )

    prepare_root = tmp_path / "prepare"
    _write_json(
        prepare_root / "ADD" / "loom-execution-runs.json",
        {
            "schema": "ggml_hrx_kernel_bench.loom_execution_runs.v1",
            "descriptor_manifest_path": str(descriptor_root / "loom-execution-descriptors.json"),
            "entries": [
                {
                    "descriptor_path": str(descriptor_path),
                    "case_id": "d04",
                    "case_values": [4],
                    "kernel": "add_f32",
                    "route_id": "add_f32_contiguous_4d",
                    "status": "prepared",
                }
            ],
        },
    )
    case_manifest = tmp_path / "built-case" / "manifest.json"
    _write_json(
        case_manifest,
        {
            "schema": "ggml_hrx_kernel_bench.loom_benchmark_script_case.v1",
            "op": "ADD",
            "route_id": "add_f32_contiguous_4d",
            "case_id": "d04",
            "descriptor_path": str(descriptor_path),
            "run_manifest_path": str(prepare_root / "ADD" / "loom-execution-runs.json"),
            "prepared_entry": {
                "descriptor_path": str(descriptor_path),
                "case_id": "d04",
                "case_values": [4],
                "kernel": "add_f32",
                "route_id": "add_f32_contiguous_4d",
                "status": "prepared",
            },
            "root": "@add_f32",
            "kernel_source": str(kernel),
            "normalized_kernel_source": "asset:kernels/v2/add/f32.loom",
            "source_content_hash": None,
            "descriptor_execution_digest": "case-digest",
            "repo_root": str(repo_root),
            "defaults": {
                "benchmark_device": "amdgpu",
                "benchmark_measure": "dispatch_complete",
            },
        },
    )
    return repo_root, kernel, case_manifest


def _write_fake_tools(tmp_path: Path) -> tuple[Path, Path]:
    loom_link = _write_executable(
        tmp_path / "tools" / "loom-link",
        """#!/usr/bin/env bash
set -euo pipefail
OUT=""
for ARG in "$@"; do
  case "$ARG" in
    --output=*) OUT="${ARG#--output=}" ;;
  esac
done
if [[ -z "$OUT" ]]; then
  echo "missing output" >&2
  exit 2
fi
cp "$1" "$OUT"
""",
    )
    benchmark_runner = _write_executable(
        tmp_path / "tools" / "iree-benchmark-loom",
        """#!/usr/bin/env bash
set -euo pipefail
OUT=""
for ARG in "$@"; do
  case "$ARG" in
    --output=*) OUT="${ARG#--output=}" ;;
  esac
done
if [[ -z "$OUT" ]]; then
  echo "missing output" >&2
  exit 2
fi
mkdir -p "$(dirname "$OUT")"
printf '%s\n' '{"row":"benchmark","benchmark_result":{"state":"ok","measurement":{"operation_timing_ns":{"mean":20,"p50":10}}}}' > "$OUT"
""",
    )
    return loom_link, benchmark_runner


def test_export_writes_standalone_capsule_without_catalog_tree(tmp_path: Path) -> None:
    _repo_root, kernel, case_manifest = _write_prepared_case(tmp_path)
    loom_link, benchmark_runner = _write_fake_tools(tmp_path)
    export_root = tmp_path / "export"

    status = export.command_export(
        SimpleNamespace(
            case=case_manifest,
            out=export_root,
            kernel=None,
            target_arch="gfx1100",
            repo_root=None,
            tool_dir=None,
            benchmark_runner=str(benchmark_runner),
            benchmark_device=None,
            benchmark_measure=None,
            loom_link=str(loom_link),
            archive=False,
        )
    )

    manifest = json.loads((export_root / "export.json").read_text(encoding="utf-8"))
    template = (export_root / "benchmark.loom.template").read_text(encoding="utf-8")

    assert status == 0
    assert manifest["schema"] == export.EXPORT_SCHEMA
    assert manifest["route_id"] == "add_f32_contiguous_4d"
    assert manifest["case_id"] == "d04"
    assert manifest["target_arch"] == "gfx1100"
    assert manifest["gpu_arch"] == "gfx1100"
    assert manifest["abi"]["root_symbol"] == "@add_f32"
    assert manifest["abi"]["configs"]["@shape.pointwise.ne0"] == "4"
    assert (export_root / "baseline.loom").read_text(encoding="utf-8") == kernel.read_text(encoding="utf-8")
    assert (export_root / "fixtures" / "src0.npy").is_file()
    assert (export_root / "run.sh").is_file()
    assert (export_root / "runner.py").is_file()
    assert not (export_root / "catalog").exists()
    assert "func.call @add_f32" in template
    assert template.index("func.call @add_f32") < template.index("check.expect.close")


def test_export_requires_target_architecture(tmp_path: Path) -> None:
    _repo_root, _kernel, case_manifest = _write_prepared_case(tmp_path)
    loom_link, benchmark_runner = _write_fake_tools(tmp_path)

    with pytest.raises(RuntimeError, match="target architecture is required"):
        export.command_export(
            SimpleNamespace(
                case=case_manifest,
                out=tmp_path / "export",
                kernel=None,
                repo_root=None,
                tool_dir=None,
                benchmark_runner=str(benchmark_runner),
                benchmark_device=None,
                benchmark_measure=None,
                loom_link=str(loom_link),
                archive=False,
            )
        )


def test_export_runner_relinks_candidate_and_writes_one_result(tmp_path: Path) -> None:
    _repo_root, kernel, case_manifest = _write_prepared_case(tmp_path)
    loom_link, benchmark_runner = _write_fake_tools(tmp_path)
    export_root = tmp_path / "export"
    export.command_export(
        SimpleNamespace(
            case=case_manifest,
            out=export_root,
            kernel=None,
            target_arch="gfx1100",
            repo_root=None,
            tool_dir=None,
            benchmark_runner=str(benchmark_runner),
            benchmark_device=None,
            benchmark_measure=None,
            loom_link=str(loom_link),
            archive=False,
        )
    )
    candidate = tmp_path / "candidate.loom"
    candidate.write_text(
        "kernel.def @add_f32() {} launch(%src0: buffer) { kernel.return }\n// candidate\n",
        encoding="utf-8",
    )
    run_dir = export_root / "runs" / "candidate-001"

    result = subprocess.run(
        [str(export_root / "run.sh"), "--kernel", str(candidate), "--out", str(run_dir)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    assert result.returncode == 0, result.stderr
    assert len(rows) == 1
    assert rows[0]["schema"] == "ggml_hrx_kernel_bench.kernel_export_result.v1"
    assert rows[0]["status"] == "pass"
    assert rows[0]["case_id"] == "d04"
    assert rows[0]["target_arch"] == "gfx1100"
    assert rows[0]["gpu_arch"] == "gfx1100"
    assert rows[0]["benchmark_summary"]["operation_timing_ns"]["p50"] == 10
    assert summary["passed_count"] == 1
    assert (run_dir / "benchmark.loom").read_text(encoding="utf-8").startswith("kernel.def @add_f32")
