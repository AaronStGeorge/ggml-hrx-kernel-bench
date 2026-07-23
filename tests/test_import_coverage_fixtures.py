from __future__ import annotations

import json
from pathlib import Path

import yaml


EXPECTED_TOP_LEVEL_KEYS = [
    "schema",
    "operation_count",
    "total_pass_case_count",
    "total_fail_case_count",
    "operations",
]

EXPECTED_OPERATION_KEYS = [
    "op",
    "pass_case_count",
    "fail_case_count",
]


def test_expected_import_coverage_fixtures_preserve_canonical_key_order() -> None:
    fixture_paths = [
        Path("benchmarks/llama-3.3-8b-mul-mat.import-coverage.json"),
        Path("tests/kernels/data/llamacpp.import-coverage.json"),
        Path("tests/models/data/llama-8b-q8.import-coverage.json"),
    ]
    fixture_paths.extend(
        sorted(Path("tests/kernels/data/llamacpp/mul_mat").glob("*.import-coverage.json"))
    )

    for fixture_path in fixture_paths:
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        assert list(payload.keys()) == EXPECTED_TOP_LEVEL_KEYS
        assert payload["operations"], f"{fixture_path} must contain operation coverage rows"
        for row in payload["operations"]:
            assert list(row.keys()) == EXPECTED_OPERATION_KEYS


def test_llamacpp_mul_mat_shards_cover_source_cases_once() -> None:
    source = yaml.safe_load(
        Path("tests/kernels/data/llamacpp_full_unsharded_source.v2.yaml").read_text(
            encoding="utf-8"
        )
    )
    core = yaml.safe_load(
        Path("tests/kernels/data/llamacpp_core.v2.yaml").read_text(
            encoding="utf-8"
        )
    )

    source_cases = source["ops"]["MUL_MAT"]
    assert "MUL_MAT" not in core["ops"]

    shard_case_count = 0
    for shard_path in sorted(Path("tests/kernels/data/llamacpp/mul_mat").glob("*.v2.yaml")):
        shard = yaml.safe_load(shard_path.read_text(encoding="utf-8"))
        assert list(shard["ops"]) == ["MUL_MAT"]
        shard_case_count += len(shard["ops"]["MUL_MAT"])

    assert shard_case_count == len(source_cases)
