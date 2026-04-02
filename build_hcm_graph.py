#!/usr/bin/env python3
"""Build and persist a directed Oracle HCM relationship graph."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import Tuple

import networkx as nx


JSON_PATH = Path("hcm_structured.json")
GRAPH_PATH = Path("hcm_graph.pkl")
HCM_CORE_KEYS = [
    "PERSON_ID",
    "ASSIGNMENT_ID",
    "DEPARTMENT_ID",
    "ORGANIZATION_ID",
    "PERIOD_OF_SERVICE_ID",
    "PAYROLL_ID",
    "SALARY_ID",
    "GRADE_ID",
    "JOB_ID",
    "POSITION_ID",
]
CORE_HR_TABLES = {
    "PER_ALL_PEOPLE_F",
    "PER_ALL_ASSIGNMENTS_M",
    "PER_DEPARTMENTS",
    "CMP_SALARY",
}
PERIPHERAL_TABLE_MARKERS = ("_ACTIONS", "_B", "_TL", "HWR_", "BEN_")


def canonical_name(name: str | None) -> str:
    return (name or "").strip().upper()


def normalize_category(default_category: str, raw_category: str | None) -> str:
    value = canonical_name(raw_category)
    if value.startswith("TABLE"):
        return "TABLE"
    if value.startswith("VIEW"):
        return "VIEW"
    return default_category


def load_schema(json_path: Path) -> Dict:
    with json_path.open("r", encoding="utf-8") as infile:
        return json.load(infile)


def edge_weight_for_tables(source: str, target: str) -> float:
    source_name = canonical_name(source)
    target_name = canonical_name(target)

    if source_name in CORE_HR_TABLES and target_name in CORE_HR_TABLES:
        return 0.1

    if any(marker in source_name for marker in PERIPHERAL_TABLE_MARKERS):
        return 5.0
    if any(marker in target_name for marker in PERIPHERAL_TABLE_MARKERS):
        return 5.0

    return 1.0


def extract_column_names(obj: Dict) -> list[str]:
    return [
        canonical_name(column.get("name"))
        for column in obj.get("columns", [])
        if column.get("name")
    ]


def iter_objects(schema: Dict) -> Iterable[Tuple[Dict, Dict, str]]:
    for module in schema.get("modules", []):
        for key, default_category in (("tables", "TABLE"), ("views", "VIEW")):
            for obj in module.get(key, []):
                yield module, obj, default_category


def collect_nodes(schema: Dict) -> Tuple[nx.Graph, Dict[str, str]]:
    graph = nx.Graph()
    name_lookup: Dict[str, str] = {}

    for module, obj, default_category in iter_objects(schema):
        actual_name = str(obj.get("name", "")).strip()
        if not actual_name:
            continue

        canonical = canonical_name(actual_name)
        module_name = str(module.get("module_name") or module.get("module") or "Unknown").strip()
        category = normalize_category(default_category, obj.get("category"))

        graph.add_node(
            actual_name,
            module=module_name,
            category=category,
            description=str(obj.get("description", "") or "").strip(),
            columns=extract_column_names(obj),
        )
        name_lookup[canonical] = actual_name

    return graph, name_lookup


def add_graph_edge(
    graph: nx.Graph,
    source: str,
    target: str,
    *,
    join_on: str,
    foreign_key_column: str,
    edge_type: str,
) -> bool:
    """Add an undirected edge while preserving stronger metadata."""
    proposed_weight = edge_weight_for_tables(source, target)
    if graph.has_edge(source, target):
        existing_edge = graph[source][target]
        existing_type = str(existing_edge.get("edge_type", ""))
        existing_join = str(existing_edge.get("join_on", ""))
        existing_weight = float(existing_edge.get("weight", 1.0))

        # Keep explicit metadata when an inferred edge targets the same pair.
        if existing_type == "explicit" and edge_type == "inferred":
            return False

        # Avoid rewriting the same relationship.
        if existing_type == edge_type and existing_join == join_on:
            return False

        # Keep the cheaper business-preferred weight if an edge already exists.
        proposed_weight = min(existing_weight, proposed_weight)

    graph.add_edge(
        source,
        target,
        join_on=join_on,
        foreign_key_column=foreign_key_column,
        edge_type=edge_type,
        weight=proposed_weight,
    )
    return True


def add_explicit_relationship_edges(schema: Dict, graph: nx.Graph, name_lookup: Dict[str, str]) -> int:
    explicit_edge_count = 0

    for _, obj, _ in iter_objects(schema):
        object_name = str(obj.get("name", "")).strip()
        foreign_keys = obj.get("foreign_keys", []) or []

        for foreign_key in foreign_keys:
            raw_source = str(foreign_key.get("table", "")).strip() or object_name
            raw_target = str(foreign_key.get("foreign_table", "")).strip()
            foreign_key_column = str(foreign_key.get("foreign_key_column", "")).strip()

            source = name_lookup.get(canonical_name(raw_source))
            target = name_lookup.get(canonical_name(raw_target))

            if not source:
                print(f"Warning: source table/view '{raw_source}' was not found in JSON. Skipping.", file=sys.stderr)
                continue
            if not target:
                print(
                    f"Warning: foreign key target '{raw_target}' referenced by '{raw_source}' was not found in JSON. Skipping.",
                    file=sys.stderr,
                )
                continue
            if not foreign_key_column:
                print(
                    f"Warning: foreign key column missing for relationship '{raw_source}' -> '{raw_target}'. Skipping.",
                    file=sys.stderr,
                )
                continue

            join_on = f"{source}.{foreign_key_column} = {target}.{foreign_key_column}"
            if add_graph_edge(
                graph,
                source,
                target,
                join_on=join_on,
                foreign_key_column=foreign_key_column,
                edge_type="explicit",
            ):
                explicit_edge_count += 1

    return explicit_edge_count


def build_primary_key_map(schema: Dict, name_lookup: Dict[str, str]) -> dict[str, list[str]]:
    primary_key_map: dict[str, list[str]] = {key: [] for key in HCM_CORE_KEYS}
    allowlist = set(HCM_CORE_KEYS)

    for _, obj, _ in iter_objects(schema):
        raw_name = str(obj.get("name", "")).strip()
        object_name = name_lookup.get(canonical_name(raw_name))
        if not object_name:
            continue

        primary_key_columns = obj.get("primary_key", {}).get("columns", []) or []
        for primary_key_column in primary_key_columns:
            pk_name = canonical_name(primary_key_column)
            if pk_name in allowlist and object_name not in primary_key_map[pk_name]:
                primary_key_map[pk_name].append(object_name)

    return primary_key_map


def add_inferred_relationship_edges(
    graph: nx.Graph,
    primary_key_map: dict[str, list[str]],
) -> int:
    inferred_edge_count = 0

    for source, node_data in graph.nodes(data=True):
        columns = set(node_data.get("columns", []))
        for column_name in HCM_CORE_KEYS:
            if column_name not in columns:
                continue

            for target in primary_key_map.get(column_name, []):
                if source == target:
                    continue

                join_on = f"{source}.{column_name} = {target}.{column_name}"
                if add_graph_edge(
                    graph,
                    source,
                    target,
                    join_on=join_on,
                    foreign_key_column=column_name,
                    edge_type="inferred",
                ):
                    inferred_edge_count += 1

    return inferred_edge_count


def save_graph(graph: nx.Graph, graph_path: Path) -> None:
    with graph_path.open("wb") as outfile:
        pickle.dump(graph, outfile, protocol=pickle.HIGHEST_PROTOCOL)


def load_graph(graph_path: Path) -> nx.Graph:
    with graph_path.open("rb") as infile:
        return pickle.load(infile)


def build_hcm_graph(json_path: Path = JSON_PATH, graph_path: Path = GRAPH_PATH) -> nx.Graph:
    schema = load_schema(json_path)
    graph, name_lookup = collect_nodes(schema)
    explicit_edge_count = add_explicit_relationship_edges(schema, graph, name_lookup)
    primary_key_map = build_primary_key_map(schema, name_lookup)
    inferred_edge_count = add_inferred_relationship_edges(graph, primary_key_map)
    save_graph(graph, graph_path)

    total_edges = graph.number_of_edges()
    print(f"Explicit edges created: {explicit_edge_count}")
    print(f"Inferred edges created: {inferred_edge_count}")
    print(f"Saved HCM graph to '{graph_path}' with {graph.number_of_nodes()} nodes and {total_edges} edges.")
    return graph


def find_hcm_path(start_table: str, end_table: str, graph_path: Path = GRAPH_PATH) -> list[str] | None:
    graph = load_graph(graph_path)
    canonical_lookup = {canonical_name(node): node for node in graph.nodes}

    start_node = canonical_lookup.get(canonical_name(start_table))
    end_node = canonical_lookup.get(canonical_name(end_table))

    if not start_node:
        print(f"Start table/view '{start_table}' was not found in the graph.")
        return None
    if not end_node:
        print(f"End table/view '{end_table}' was not found in the graph.")
        return None

    try:
        path = nx.shortest_path(graph, source=start_node, target=end_node, weight="weight")
    except nx.NetworkXNoPath:
        print(f"No path found between '{start_node}' and '{end_node}'.")
        return None

    print("Path:")
    for node in path:
        print(f" - {node}")

    print("\nJoin conditions:")
    for source, target in zip(path, path[1:]):
        edge_data = graph.get_edge_data(source, target) or {}
        if not edge_data:
            print(f" - {source} -> {target}: [missing edge metadata]")
            continue

        edge_type = edge_data.get("edge_type", "unknown")
        join_on = edge_data.get("join_on", "[unknown join]")
        weight = edge_data.get("weight", "[unknown weight]")
        print(f" - {source} -> {target} [{edge_type}, weight={weight}]: {join_on}")

    return path


if __name__ == "__main__":
    if len(sys.argv) == 1:
        build_hcm_graph()
    elif len(sys.argv) == 4 and sys.argv[1] == "path":
        find_hcm_path(sys.argv[2], sys.argv[3])
    else:
        print("Usage:")
        print("  python build_hcm_graph.py")
        print("  python build_hcm_graph.py path <START_TABLE> <END_TABLE>")
        raise SystemExit(1)
