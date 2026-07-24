from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkSuite:
    name: str
    description: str
    prepare_root: Path
    asset_root: Path
    default_op: str | None
    script_target: str | None
    route_import_target: str | None
    benchmark_yaml: Path | None

    def resolved(self, *, repo_root: Path) -> "BenchmarkSuite":
        root = repo_root.resolve()
        return BenchmarkSuite(
            name=self.name,
            description=self.description,
            prepare_root=_resolve_under(root, self.prepare_root),
            asset_root=_resolve_under(root, self.asset_root),
            default_op=self.default_op,
            script_target=self.script_target,
            route_import_target=self.route_import_target,
            benchmark_yaml=_resolve_under(root, self.benchmark_yaml) if self.benchmark_yaml else None,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "prepare_root": str(self.prepare_root),
            "asset_root": str(self.asset_root),
            "default_op": self.default_op,
            "script_target": self.script_target,
            "route_import_target": self.route_import_target,
            "benchmark_yaml": str(self.benchmark_yaml) if self.benchmark_yaml else None,
        }


def _resolve_under(repo_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


BUILTIN_SUITES: dict[str, BenchmarkSuite] = {
    "llama-3.3-8b-mul-mat": BenchmarkSuite(
        name="llama-3.3-8b-mul-mat",
        description="Llama 3.3 8B MUL_MAT benchmark suite.",
        prepare_root=Path("build/benchmarks/artifacts/kernel-prepare-llama-3.3-8b-mul-mat-v2"),
        asset_root=Path("build/generated/assets"),
        default_op="MUL_MAT",
        script_target="kernel-benchmark-llama-3-3-8b-mul-mat-v2-scripts",
        route_import_target="benchmark-llama-3-3-8b-mul-mat-yaml-route-import-v2",
        benchmark_yaml=Path("benchmarks/llama-3.3-8b-mul-mat.v2.yaml"),
    ),
    "llama-3.3-8b-mul-mat-q8-0-f32": BenchmarkSuite(
        name="llama-3.3-8b-mul-mat-q8-0-f32",
        description="Llama 3.3 8B MUL_MAT q8_0/f32 benchmark suite.",
        prepare_root=Path("build/benchmarks/artifacts/kernel-prepare-llama-3.3-8b-mul-mat-q8-0-f32-v2"),
        asset_root=Path("build/generated/assets"),
        default_op="MUL_MAT",
        script_target="kernel-benchmark-llama-3-3-8b-mul-mat-q8-0-f32-v2-scripts",
        route_import_target="benchmark-llama-3-3-8b-mul-mat-q8-0-f32-yaml-route-import-v2",
        benchmark_yaml=Path("benchmarks/llama-3.3-8b-mul-mat-q8-0-f32.v2.yaml"),
    ),
    "llama-cpp-mul-mat-f32-f32": BenchmarkSuite(
        name="llama-cpp-mul-mat-f32-f32",
        description="llama.cpp MUL_MAT src0=f32 src1=f32 dst=f32 benchmark suite.",
        prepare_root=Path(
            "build/benchmarks/artifacts/kernel-prepare-llama-cpp-mul-mat-src0-f32-src1-f32-dst-f32-v2"
        ),
        asset_root=Path("build/generated/assets"),
        default_op="MUL_MAT",
        script_target="kernel-benchmark-llama-cpp-mul-mat-src0-f32-src1-f32-dst-f32-v2-scripts",
        route_import_target="benchmark-llama-cpp-mul-mat-src0-f32-src1-f32-dst-f32-yaml-route-import-v2",
        benchmark_yaml=Path("benchmarks/llama-cpp-mul-mat-src0-f32-src1-f32-dst-f32.v2.yaml"),
    ),
}


def list_suites() -> list[BenchmarkSuite]:
    return [BUILTIN_SUITES[name] for name in sorted(BUILTIN_SUITES)]


def get_suite(name: str, *, repo_root: Path) -> BenchmarkSuite:
    suite = BUILTIN_SUITES.get(name)
    if suite is None:
        available = ", ".join(sorted(BUILTIN_SUITES))
        raise RuntimeError(f"unknown benchmark suite {name!r}; available suites: {available}")
    return suite.resolved(repo_root=repo_root)


def missing_artifact_message(*, suite: BenchmarkSuite, path: Path, artifact: str) -> str:
    message = f"suite {suite.name!r} {artifact} does not exist: {path}"
    if suite.script_target:
        message += f"\nBuild it with: cmake --build build --target {suite.script_target}"
    elif suite.route_import_target:
        message += f"\nBuild it with: cmake --build build --target {suite.route_import_target}"
    return message
