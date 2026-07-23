from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


SCHEMA = "ggml_hrx_kernel_bench.import_test_coverage.v1"


class GroupedYamlDumper(yaml.SafeDumper):
    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        return super().increase_indent(flow, False)


def _represent_list(dumper: yaml.Dumper, data: list[Any]) -> yaml.SequenceNode:
    flow_style = all(not isinstance(item, (dict, list)) for item in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow_style)


GroupedYamlDumper.add_representer(list, _represent_list)


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"expected YAML object in {path}")
    return data


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"expected JSON object in {path}")
    return data


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.dump(payload, Dumper=GroupedYamlDumper, sort_keys=False),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _case_signature(case: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    inputs = case.get("inputs") or []
    destinations = case.get("destinations") or []
    return (
        tuple(str(tensor["dtype"]).upper() for tensor in inputs),
        tuple(str(tensor["dtype"]).upper() for tensor in destinations),
    )


def _slug_dtype(dtype: str) -> str:
    return dtype.lower().replace("_", "-")


def _signature_slug(signature: tuple[tuple[str, ...], tuple[str, ...]]) -> str:
    inputs, destinations = signature
    input_parts = [f"src{index}-{_slug_dtype(dtype)}" for index, dtype in enumerate(inputs)]
    destination_parts = [
        f"dst-{_slug_dtype(dtype)}" if index == 0 else f"dst{index}-{_slug_dtype(dtype)}"
        for index, dtype in enumerate(destinations)
    ]
    return "-".join([*input_parts, *destination_parts])


def _signature_sort_key(signature: tuple[tuple[str, ...], tuple[str, ...]]) -> tuple[str, ...]:
    inputs, destinations = signature
    return (*inputs, *destinations)


def _coverage_payload(*, pass_count: int, fail_count: int) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "operation_count": 1,
        "total_pass_case_count": pass_count,
        "total_fail_case_count": fail_count,
        "operations": [
            {
                "op": "MUL_MAT",
                "pass_case_count": pass_count,
                "fail_case_count": fail_count,
            }
        ],
    }


def _native_route_count_payload(*, pass_count: int) -> dict[str, Any]:
    return {"native_route_count": pass_count}


def _row_signature(row: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    case = row.get("case")
    if not isinstance(case, dict):
        raise RuntimeError("route artifact row is missing case object")
    return _case_signature(case)


def _count_by_signature(path: Path) -> Counter[tuple[tuple[str, ...], tuple[str, ...]]]:
    payload = _load_json(path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError(f"expected rows list in {path}")
    return Counter(_row_signature(row) for row in rows if isinstance(row, dict))


def shard_mul_mat(
    *,
    source_yaml: Path,
    route_matches: Path,
    route_unmatched: Path,
    core_yaml: Path,
    shard_dir: Path,
    aggregate_coverage: Path,
) -> list[str]:
    data = _load_yaml(source_yaml)
    ops = data.get("ops")
    if not isinstance(ops, dict):
        raise RuntimeError(f"expected ops object in {source_yaml}")
    mul_mat_cases = ops.get("MUL_MAT")
    if not isinstance(mul_mat_cases, list):
        raise RuntimeError(f"expected ops.MUL_MAT list in {source_yaml}")

    core_data = copy.deepcopy(data)
    del core_data["ops"]["MUL_MAT"]
    _write_yaml(core_yaml, core_data)

    cases_by_signature: dict[tuple[tuple[str, ...], tuple[str, ...]], list[dict[str, Any]]] = {}
    for case in mul_mat_cases:
        if not isinstance(case, dict):
            raise RuntimeError("MUL_MAT cases must be objects")
        cases_by_signature.setdefault(_case_signature(case), []).append(case)

    pass_counts = _count_by_signature(route_matches)
    fail_counts = _count_by_signature(route_unmatched)
    shard_ids: list[str] = []
    for signature in sorted(cases_by_signature, key=_signature_sort_key):
        slug = _signature_slug(signature)
        shard_ids.append(slug)
        shard_yaml = shard_dir / f"llamacpp-mul-mat-{slug}.v2.yaml"
        coverage_json = shard_dir / f"llamacpp-mul-mat-{slug}.import-coverage.json"
        native_route_count_json = shard_dir / f"llamacpp-mul-mat-{slug}.native-route-count.json"
        cases = cases_by_signature[signature]
        pass_count = pass_counts[signature]
        fail_count = fail_counts[signature]
        if pass_count + fail_count != len(cases):
            raise RuntimeError(
                f"{slug} route artifact count mismatch: "
                f"{pass_count} pass + {fail_count} fail != {len(cases)} cases"
            )
        _write_yaml(shard_yaml, {"ops": {"MUL_MAT": cases}})
        _write_json(coverage_json, _coverage_payload(pass_count=pass_count, fail_count=fail_count))
        _write_json(native_route_count_json, _native_route_count_payload(pass_count=pass_count))

    aggregate = _load_json(aggregate_coverage)
    operations = aggregate.get("operations")
    if not isinstance(operations, list):
        raise RuntimeError(f"expected operations list in {aggregate_coverage}")
    core_operations = []
    removed_pass = 0
    removed_fail = 0
    for row in operations:
        if row.get("op") == "MUL_MAT":
            removed_pass += int(row["pass_case_count"])
            removed_fail += int(row["fail_case_count"])
            continue
        core_operations.append(row)
    aggregate["operation_count"] = len(core_operations)
    aggregate["total_pass_case_count"] = int(aggregate["total_pass_case_count"]) - removed_pass
    aggregate["total_fail_case_count"] = int(aggregate["total_fail_case_count"]) - removed_fail
    aggregate["operations"] = core_operations
    _write_json(aggregate_coverage, aggregate)
    return shard_ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Shard llama.cpp MUL_MAT grouped-YAML cases by dtype signature.")
    parser.add_argument("--source-yaml", type=Path, required=True)
    parser.add_argument("--route-matches", type=Path, required=True)
    parser.add_argument("--route-unmatched", type=Path, required=True)
    parser.add_argument("--core-yaml", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--aggregate-coverage", type=Path, required=True)
    args = parser.parse_args()

    shard_ids = shard_mul_mat(
        source_yaml=args.source_yaml,
        route_matches=args.route_matches,
        route_unmatched=args.route_unmatched,
        core_yaml=args.core_yaml,
        shard_dir=args.shard_dir,
        aggregate_coverage=args.aggregate_coverage,
    )
    for shard_id in shard_ids:
        print(shard_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
