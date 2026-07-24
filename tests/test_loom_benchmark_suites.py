from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ggml_hrx_kernel_bench.benchmarking import suites  # noqa: E402


def test_builtin_suites_include_f32_f32_matmul() -> None:
    names = [suite.name for suite in suites.list_suites()]

    assert "llama-cpp-mul-mat-f32-f32" in names


def test_get_suite_resolves_paths_under_repo_root(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"

    suite = suites.get_suite("llama-cpp-mul-mat-f32-f32", repo_root=repo_root)

    assert suite.prepare_root == (
        repo_root
        / "build"
        / "benchmarks"
        / "artifacts"
        / "kernel-prepare-llama-cpp-mul-mat-src0-f32-src1-f32-dst-f32-v2"
    ).resolve()
    assert suite.asset_root == (repo_root / "build" / "generated" / "assets").resolve()
    assert suite.default_op == "MUL_MAT"
    assert suite.script_target == "kernel-benchmark-llama-cpp-mul-mat-src0-f32-src1-f32-dst-f32-v2-scripts"


def test_get_suite_reports_available_names(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="available suites:"):
        suites.get_suite("missing", repo_root=tmp_path)


def test_missing_artifact_message_includes_build_target(tmp_path: Path) -> None:
    suite = suites.get_suite("llama-cpp-mul-mat-f32-f32", repo_root=tmp_path)

    message = suites.missing_artifact_message(
        suite=suite,
        path=suite.prepare_root,
        artifact="prepare root",
    )

    assert "prepare root does not exist" in message
    assert "cmake --build build --target kernel-benchmark-llama-cpp-mul-mat-src0-f32-src1-f32-dst-f32-v2-scripts" in message
