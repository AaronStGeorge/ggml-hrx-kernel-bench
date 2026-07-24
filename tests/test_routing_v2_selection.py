from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

import pytest

import ggml_hrx_kernel_bench.routing.v2.selection as selection_module
from ggml_hrx_kernel_bench.routing.v2.models import (
    ConcreteTensor,
    ConcreteTensorDimension,
    ConstraintCheck,
    RouteConstraints,
    TensorDescriptor,
    V2Route,
)
from ggml_hrx_kernel_bench.routing.v2.query import RouteCatalog
from ggml_hrx_kernel_bench.routing.v2.selection import (
    ROUTE_QUERY_SCHEMA,
    RouteQuery,
    route_query_from_json,
    route_query_to_json,
    select_route_query,
)


def _tensor(
    *,
    dtype: str = "F32",
    sizes: tuple[int, ...] = (4,),
    strides: tuple[int, ...] = (1,),
) -> ConcreteTensor:
    return ConcreteTensor(
        dtype=dtype,
        dimensions=tuple(
            ConcreteTensorDimension(name=f"d{index}", size=size, stride=stride)
            for index, (size, stride) in enumerate(zip(sizes, strides, strict=True))
        ),
    )


def _route(
    route_id: str,
    *,
    op: str = "TEST",
    tensors: Mapping[str, TensorDescriptor] | None = None,
    constraints: tuple[ConstraintCheck, ...] = (),
    attributes: Mapping[str, Any] | None = None,
) -> V2Route:
    return V2Route(
        id=route_id,
        family="test",
        op=op,
        source_id="test",
        kernel_path=f"test/{route_id}.loom",
        root_symbol=f"@{route_id}",
        export_name=route_id,
        tensors=(
            tensors
            if tensors is not None
            else {
                "src0": TensorDescriptor(
                    dtype="F32",
                    dimensions_capture="dimensions",
                    strides_capture="strides",
                )
            }
        ),
        values=(),
        constraints=RouteConstraints(checks=constraints),
        launch={},
        bindings=(),
        attributes={} if attributes is None else attributes,
    )


def _catalog(*routes: V2Route) -> RouteCatalog:
    by_op: dict[str, list[V2Route]] = defaultdict(list)
    by_family: dict[str, list[V2Route]] = defaultdict(list)
    for route in routes:
        by_op[route.op].append(route)
        by_family[route.family].append(route)
    return RouteCatalog(
        routes=routes,
        routes_by_op={op: tuple(op_routes) for op, op_routes in by_op.items()},
        routes_by_family={family: tuple(family_routes) for family, family_routes in by_family.items()},
        routes_by_id={route.id: route for route in routes},
    )


def test_route_query_json_round_trip_matches_native_query_shape() -> None:
    payload = {
        "op": "ADD",
        "tensors": {
            "src0": {
                "dtype": "F32",
                "dimensions": [4, 2],
                "strides": [1, 8],
                # The native parser accepts a permutation whose length differs from the rank.
                "permutation": [1, 0, 2],
            }
        },
        "attributes": {
            "nothing": None,
            "enabled": True,
            "minimum": -(1 << 63),
            "maximum": (1 << 63) - 1,
            "ratio": 1.25,
            "name": "example",
            "items": [False, 3, {"nested": "value"}],
        },
    }

    query = RouteQuery.from_json(payload)

    assert ROUTE_QUERY_SCHEMA == "ggml_hrx_kernel_bench.route_query.v1"
    assert query.operation == "ADD"
    assert query.tensors["src0"].permutation == (1, 0, 2)
    assert query.to_json() == payload
    assert route_query_to_json(route_query_from_json(payload)) == payload


def test_route_query_json_treats_missing_attributes_and_null_permutation_as_absent() -> None:
    query = route_query_from_json(
        {
            "op": "EMPTY",
            "tensors": {
                "src0": {
                    "dtype": "",
                    "dimensions": [],
                    "strides": [],
                    "permutation": None,
                }
            },
        }
    )

    assert query.attributes == {}
    assert query.tensors["src0"].permutation is None
    assert route_query_to_json(query) == {
        "op": "EMPTY",
        "tensors": {"src0": {"dtype": "", "dimensions": [], "strides": []}},
        "attributes": {},
    }


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        ([], "route query must be an object"),
        ({"op": "ADD", "tensors": {}, "allowed_route_ids": []}, "unknown field"),
        ({"op": "ADD", "tensors": {}, "unexpected": True}, "unknown field"),
        ({"op": 7, "tensors": {}}, "field 'op' must be a string"),
        ({"op": "ADD", "tensors": [], "attributes": {}}, "field 'tensors' must be an object"),
        ({"op": "ADD", "tensors": {}, "attributes": None}, "field 'attributes' must be an object"),
        (
            {
                "op": "ADD",
                "tensors": {
                    "src0": {"dtype": "F32", "dimensions": [True], "strides": [1]}
                },
            },
            "must be a signed 64-bit integer",
        ),
        (
            {
                "op": "ADD",
                "tensors": {
                    "src0": {
                        "dtype": "F32",
                        "dimensions": [1 << 63],
                        "strides": [1],
                    }
                },
            },
            "outside the signed 64-bit integer range",
        ),
        (
            {
                "op": "ADD",
                "tensors": {
                    "src0": {"dtype": "F32", "dimensions": [1], "strides": []}
                },
            },
            "dimensions and strides must have equal length",
        ),
        (
            {"op": "ADD", "tensors": {}, "attributes": {"value": 1 << 63}},
            "outside the signed 64-bit integer range",
        ),
        (
            {"op": "ADD", "tensors": {}, "attributes": {"value": float("inf")}},
            "must be a finite floating-point number",
        ),
    ),
)
def test_route_query_json_rejects_non_native_query_values(
    payload: Any,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        route_query_from_json(payload)


def test_select_route_query_uses_first_catalog_match_before_later_exact_rank() -> None:
    range_route = _route(
        "range",
        constraints=(ConstraintCheck(name="dimensions", rank_min=1, rank_max=4),),
    )
    exact_route = _route(
        "exact",
        constraints=(ConstraintCheck(name="dimensions", length=1),),
    )
    query = RouteQuery(operation="test", tensors={"src0": _tensor()}, attributes={})

    selection = select_route_query(_catalog(range_route, exact_route), query)

    assert selection.status == "matched"
    assert selection.route_ids == ("range",)
    assert selection.candidate_route_ids == ("range",)


def test_select_route_query_resolves_overlapping_routes_to_first_catalog_match() -> None:
    first = _route("first", constraints=(ConstraintCheck(name="dimensions", length=1),))
    second = _route("second", constraints=(ConstraintCheck(name="dimensions", length=1),))

    selection = select_route_query(
        _catalog(first, second),
        RouteQuery(operation="TEST", tensors={"src0": _tensor()}, attributes={}),
    )

    assert selection.status == "matched"
    assert selection.route_ids == ("first",)
    assert selection.candidate_route_ids == ("first",)


def test_select_route_query_does_not_evaluate_routes_after_first_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _route("first")
    second = _route("second")
    evaluated_route_ids: list[str] = []

    def record_attribute_evaluation(route: V2Route, attributes: Mapping[str, Any]) -> bool:
        evaluated_route_ids.append(route.id)
        return True

    monkeypatch.setattr(
        selection_module,
        "route_accepts_attributes",
        record_attribute_evaluation,
    )

    selection = select_route_query(
        _catalog(first, second),
        RouteQuery(operation="TEST", tensors={"src0": _tensor()}, attributes={}),
    )

    assert selection.route_ids == ("first",)
    assert evaluated_route_ids == ["first"]


def test_select_route_query_requires_exact_tensor_roles() -> None:
    route = _route(
        "exact-roles",
        tensors={
            "src0": TensorDescriptor("F32", "src_dimensions", "src_strides"),
            "dst": TensorDescriptor("F32", "dst_dimensions", "dst_strides"),
        },
    )
    catalog = _catalog(route)
    exact_tensors = {
        "src0": _tensor(),
        "dst": _tensor(),
    }

    matched = select_route_query(
        catalog,
        RouteQuery(operation="TEST", tensors=exact_tensors),
    )
    missing_role = select_route_query(
        catalog,
        RouteQuery(operation="TEST", tensors={"src0": _tensor()}),
    )
    extra_role = select_route_query(
        catalog,
        RouteQuery(
            operation="TEST",
            tensors={**exact_tensors, "mask": _tensor()},
        ),
    )

    assert matched.status == "matched"
    assert matched.route_ids == ("exact-roles",)
    assert missing_role.status == "unmatched"
    assert missing_role.route_ids == ()
    assert extra_role.status == "unmatched"
    assert extra_role.route_ids == ()


def test_select_route_query_reports_unknown_operation_and_empty_tensors_as_unmatched() -> None:
    selection = select_route_query(
        _catalog(_route("only-route")),
        RouteQuery(operation="UNKNOWN", tensors={}, attributes={}),
    )

    assert selection.status == "unmatched"
    assert selection.route_ids == ()
    assert selection.candidate_route_ids == ()
