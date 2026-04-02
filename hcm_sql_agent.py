#!/usr/bin/env python3
"""Foundation LangGraph agent for Oracle HCM table retrieval and join discovery."""

from __future__ import annotations

import json
import os
import pickle
import re
from difflib import get_close_matches
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import TypedDict

import networkx as nx
from langgraph.graph import END
from langgraph.graph import StateGraph
from openai import OpenAI
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from env_config import load_env_file


load_env_file()


PINECONE_API_KEY = os.getenv(
    "PINECONE_API_KEY",
)
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "chatbot")
JSON_PATH = Path(os.getenv("HCM_SCHEMA_JSON_PATH", "hcm_structured.json"))
GRAPH_PATH = Path(os.getenv("HCM_GRAPH_PATH", "hcm_graph.pkl"))
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_TABLES = 5
MAX_TOTAL_TABLES = 7
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
CORE_HR_KEYWORD_BOOSTS = {
    "person": ["PER_ALL_PEOPLE_F", "PER_PERSON_NAMES_F", "PER_ALL_ASSIGNMENTS_M"],
    "employee": ["PER_ALL_PEOPLE_F", "PER_PERSON_NAMES_F", "PER_ALL_ASSIGNMENTS_M"],
    "name": ["PER_ALL_PEOPLE_F", "PER_PERSON_NAMES_F", "PER_ALL_ASSIGNMENTS_M"],
    "salary": ["CMP_SALARY", "CMP_SALARY_V"],
    "pay": ["CMP_SALARY", "CMP_SALARY_V"],
    "department": ["PER_DEPARTMENTS", "HR_ALL_ORGANIZATION_UNITS_F"],
    "organization": ["PER_DEPARTMENTS", "HR_ALL_ORGANIZATION_UNITS_F"],
    "manager": ["PER_DEPARTMENTS", "HR_ALL_ORGANIZATION_UNITS_F"],
}


class AgentState(TypedDict):
    user_question: str
    query_mode: str
    relevant_tables: List[str]
    required_bridge_tables: List[str]
    table_metadata: List[dict[str, Any]]
    join_paths: List[str]
    required_join_sequences: List[str]
    sql_query: str
    response_text: str
    saved_sql_path: str
    iteration_count: int
    corrections: str
    history: List[Dict[str, Any]]
    error_log: str


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def get_pinecone_index():
    client = Pinecone(api_key=PINECONE_API_KEY)
    return client.Index(PINECONE_INDEX_NAME)


@lru_cache(maxsize=1)
def get_openrouter_client() -> OpenAI:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is required for SQL generation.")

    return OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )


@lru_cache(maxsize=1)
def get_hcm_graph() -> nx.Graph:
    with GRAPH_PATH.open("rb") as infile:
        return pickle.load(infile)


@lru_cache(maxsize=1)
def get_table_metadata_lookup() -> dict[str, dict[str, Any]]:
    with JSON_PATH.open("r", encoding="utf-8") as infile:
        schema = json.load(infile)

    metadata_lookup: dict[str, dict[str, Any]] = {}
    for module in schema.get("modules", []):
        module_name = str(module.get("module_name") or module.get("module") or "").strip()
        for object_group, default_category in (("tables", "TABLE"), ("views", "VIEW")):
            for obj in module.get(object_group, []):
                table_name = str(obj.get("name", "")).strip()
                if not table_name:
                    continue

                raw_category = str(obj.get("category", "") or "").strip().upper()
                if raw_category.startswith("TABLE"):
                    category = "TABLE"
                elif raw_category.startswith("VIEW"):
                    category = "VIEW"
                else:
                    category = default_category

                primary_key_columns = obj.get("primary_key", {}).get("columns", []) or []
                column_names = [
                    str(column.get("name", "")).strip().upper()
                    for column in obj.get("columns", [])
                    if column.get("name")
                ]
                metadata_lookup[table_name] = {
                    "table_name": table_name,
                    "module": module_name,
                    "category": category,
                    "primary_key": str(primary_key_columns[0]).strip() if primary_key_columns else "",
                    "columns": column_names,
                    "is_effective_dated": (
                        "EFFECTIVE_START_DATE" in column_names or "EFFECTIVE_DATE" in column_names
                    ),
                }

    return metadata_lookup


def append_error(existing: str, message: str) -> str:
    return f"{existing}\n{message}".strip() if existing else message


def extract_question_ids(question: str) -> list[str]:
    return re.findall(r"\b\d+\b", question)


def get_boosted_tables(question: str) -> list[str]:
    normalized_question = question.lower()
    boosted_tables: list[str] = []
    seen_tables: set[str] = set()

    for keyword, tables in CORE_HR_KEYWORD_BOOSTS.items():
        if not re.search(rf"\b{re.escape(keyword)}\b", normalized_question):
            continue

        for table_name in tables:
            if table_name not in seen_tables:
                boosted_tables.append(table_name)
                seen_tables.add(table_name)

    return boosted_tables


def parse_validator_json(raw_output: str) -> dict[str, Any]:
    fallback = {
        "is_valid": False,
        "score": 0,
        "errors": ["Validator did not return valid JSON."],
        "required_fix": "Return only valid JSON and produce SQL that follows the validator rules exactly.",
    }
    if not raw_output.strip():
        return fallback

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        return fallback

    if not isinstance(parsed, dict):
        return fallback

    is_valid = bool(parsed.get("is_valid", False))
    score = parsed.get("score", 0)
    errors = parsed.get("errors", [])
    required_fix = str(parsed.get("required_fix", "") or "").strip()

    try:
        score = int(score)
    except (TypeError, ValueError):
        score = 0

    if not isinstance(errors, list):
        errors = [str(errors)]
    errors = [str(error) for error in errors]

    if not required_fix and not is_valid:
        required_fix = "Fix the validator errors and return a compliant Oracle SQL query."

    return {
        "is_valid": is_valid,
        "score": max(0, min(10, score)),
        "errors": errors,
        "required_fix": required_fix,
    }


def add_warning_comment(sql_query: str) -> str:
    warning = "-- WARNING: This SQL failed validation. Use with caution."
    if not sql_query:
        return warning
    if sql_query.startswith(warning):
        return sql_query
    return f"{warning}\n{sql_query}"


def slugify_question(question: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", question.lower()).strip("_")
    return slug[:100] or "query"


def normalize_lookup_token(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", value.upper())


def extract_match_table_name(match: Any) -> str | None:
    metadata = getattr(match, "metadata", None)
    if metadata is None and isinstance(match, dict):
        metadata = match.get("metadata")

    if isinstance(metadata, dict):
        table_name = metadata.get("table_name")
        if table_name:
            return str(table_name)

    match_id = getattr(match, "id", None)
    if match_id is None and isinstance(match, dict):
        match_id = match.get("id")
    return str(match_id) if match_id else None


def extract_match_metadata(match: Any) -> dict[str, Any]:
    metadata = getattr(match, "metadata", None)
    if metadata is None and isinstance(match, dict):
        metadata = match.get("metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def iter_matches(result: Any) -> list[Any]:
    matches = getattr(result, "matches", None)
    if matches is None and isinstance(result, dict):
        matches = result.get("matches", [])
    return list(matches or [])


def build_table_metadata(table_name: str, base_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    lookup = dict(get_table_metadata_lookup().get(table_name, {}))
    if base_metadata:
        for key, value in base_metadata.items():
            if value not in (None, ""):
                lookup[key] = value

    return {
        "table_name": table_name,
        "module": lookup.get("module"),
        "category": lookup.get("category"),
        "primary_key": lookup.get("primary_key"),
        "columns": lookup.get("columns", []),
        "is_effective_dated": lookup.get("is_effective_dated"),
    }


@lru_cache(maxsize=1)
def get_module_lookup() -> dict[str, dict[str, list[str]]]:
    with JSON_PATH.open("r", encoding="utf-8") as infile:
        schema = json.load(infile)

    module_lookup: dict[str, dict[str, list[str]]] = {}
    for module in schema.get("modules", []):
        module_name = str(module.get("module_name") or module.get("module") or "").strip()
        if not module_name:
            continue
        module_lookup[module_name] = {
            "tables": [
                str(obj.get("name", "")).strip()
                for obj in module.get("tables", [])
                if obj.get("name")
            ],
            "views": [
                str(obj.get("name", "")).strip()
                for obj in module.get("views", [])
                if obj.get("name")
            ],
        }
    return module_lookup


def infer_query_mode(question: str) -> str:
    normalized_question = question.lower()
    schema_keywords = (
        "what are the tables",
        "list tables",
        "available in",
        "which tables",
        "which views",
        "what are the columns",
        "list columns",
        "primary key",
        "primary keys",
        "pk of",
        "columns present",
        "module",
    )
    if any(keyword in normalized_question for keyword in schema_keywords):
        return "schema"
    return "sql"


def match_module_name(question: str) -> str | None:
    normalized_question = question.lower()
    for module_name in get_module_lookup():
        if module_name.lower() in normalized_question:
            return module_name
    return None


def match_table_name(question: str) -> str | None:
    metadata_lookup = get_table_metadata_lookup()
    normalized_question = normalize_lookup_token(question)
    normalized_table_names = {
        normalize_lookup_token(table_name): table_name
        for table_name in metadata_lookup
    }

    for normalized_name, table_name in normalized_table_names.items():
        if normalized_name and normalized_name in normalized_question:
            return table_name

    question_tokens = [
        normalize_lookup_token(token)
        for token in re.findall(r"[A-Za-z0-9_]+", question)
        if len(token) >= 6
    ]
    for token in question_tokens:
        matches = get_close_matches(token, list(normalized_table_names.keys()), n=1, cutoff=0.72)
        if matches:
            return normalized_table_names[matches[0]]

    return None


def detect_schema_request_kind(question: str) -> str:
    normalized_question = question.lower()
    if "primary key" in normalized_question or re.search(r"\bpk\b", normalized_question):
        return "primary_key"
    if "column" in normalized_question:
        return "columns"
    if "view" in normalized_question:
        return "views"
    return "tables"


def format_schema_response(question: str) -> tuple[str, list[str], list[dict[str, Any]]]:
    module_name = match_module_name(question)
    table_name = match_table_name(question)
    request_kind = detect_schema_request_kind(question)

    if request_kind in {"columns", "primary_key"} and table_name:
        metadata = build_table_metadata(table_name)
        columns = metadata.get("columns", [])
        if request_kind == "columns":
            response = (
                f"Columns available in `{table_name}`:\n\n" +
                "\n".join(f"- `{column}`" for column in columns)
            ) if columns else f"No columns were found for `{table_name}`."
        else:
            primary_key = metadata.get("primary_key")
            response = (
                f"Primary key for `{table_name}`: `{primary_key}`"
                if primary_key
                else f"No primary key metadata was found for `{table_name}`."
            )
        return response, [table_name], [metadata]

    if module_name:
        module_objects = get_module_lookup().get(module_name, {"tables": [], "views": []})
        object_kind = "views" if request_kind == "views" else "tables"
        objects = module_objects.get(object_kind, [])
        response = (
            f"{object_kind.title()} available in module `{module_name}`:\n\n" +
            "\n".join(f"- `{name}`" for name in objects)
        ) if objects else f"No {object_kind} were found for module `{module_name}`."
        metadata = [build_table_metadata(name) for name in objects]
        return response, objects, metadata

    if table_name:
        metadata = build_table_metadata(table_name)
        response = (
            f"Table `{table_name}` was matched.\n\n"
            f"- Primary Key: `{metadata.get('primary_key') or '[none]'}`\n"
            f"- Effective-Dated: `{metadata.get('is_effective_dated')}`\n"
            f"- Module: `{metadata.get('module')}`"
        )
        return response, [table_name], [metadata]

    return (
        "I could not match that schema question to a known module or table. "
        "Try mentioning a module name like `AI` or a table name like `FAI_AGENT_PREFERENCES`.",
        [],
        [],
    )


def retrieve_tables(state: AgentState) -> AgentState:
    try:
        embedding_model = get_embedding_model()
        index = get_pinecone_index()
        boosted_tables = get_boosted_tables(state["user_question"])

        relevant_tables: list[str] = []
        table_metadata: list[dict[str, Any]] = []
        seen_tables: set[str] = set()

        for table_name in boosted_tables:
            if len(relevant_tables) >= MAX_TOTAL_TABLES:
                break
            if table_name not in seen_tables:
                relevant_tables.append(table_name)
                table_metadata.append(build_table_metadata(table_name))
                seen_tables.add(table_name)

        remaining_slots = min(
            max(0, TOP_K_TABLES - len(relevant_tables)),
            max(0, MAX_TOTAL_TABLES - len(relevant_tables)),
        )

        if remaining_slots > 0:
            embedding = embedding_model.encode(
                state["user_question"],
                normalize_embeddings=True,
            )
            result = index.query(
                vector=embedding.tolist(),
                top_k=remaining_slots,
                include_metadata=True,
            )

            for match in iter_matches(result):
                if len(relevant_tables) >= MAX_TOTAL_TABLES:
                    break
                table_name = extract_match_table_name(match)
                if table_name and table_name not in seen_tables:
                    metadata = extract_match_metadata(match)
                    relevant_tables.append(table_name)
                    table_metadata.append(build_table_metadata(table_name, metadata))
                    seen_tables.add(table_name)

        return {
            **state,
            "relevant_tables": relevant_tables,
            "required_bridge_tables": [],
            "table_metadata": table_metadata,
            "required_join_sequences": [],
        }
    except Exception as exc:
        return {
            **state,
            "relevant_tables": [],
            "required_bridge_tables": [],
            "table_metadata": [],
            "required_join_sequences": [],
            "error_log": append_error(state["error_log"], f"Retriever error: {exc}"),
        }


def route_question(state: AgentState) -> AgentState:
    return {
        **state,
        "query_mode": infer_query_mode(state["user_question"]),
    }


def answer_schema_question(state: AgentState) -> AgentState:
    response_text, relevant_tables, table_metadata = format_schema_response(state["user_question"])
    return {
        **state,
        "response_text": response_text,
        "relevant_tables": relevant_tables,
        "table_metadata": table_metadata,
    }


def find_joins(state: AgentState) -> AgentState:
    try:
        graph = get_hcm_graph()
        unique_join_paths: list[str] = []
        seen_join_paths: set[str] = set()
        required_bridge_tables: list[str] = []
        seen_bridge_tables: set[str] = set()
        required_join_sequences: list[str] = []
        seen_join_sequences: set[str] = set()
        join_errors: list[str] = []
        table_metadata_map = {
            str(table_info.get("table_name")): dict(table_info)
            for table_info in state["table_metadata"]
            if table_info.get("table_name")
        }

        for start_table, end_table in combinations(state["relevant_tables"], 2):
            if start_table not in graph or end_table not in graph:
                join_errors.append(f"Graph is missing node(s): {start_table}, {end_table}")
                continue

            try:
                path = nx.shortest_path(graph, source=start_table, target=end_table, weight="weight")
            except nx.NetworkXNoPath:
                join_errors.append(f"No join path found between {start_table} and {end_table}")
                continue

            bridge_tables = path[1:-1]
            for bridge_table in bridge_tables:
                if bridge_table not in seen_bridge_tables:
                    required_bridge_tables.append(bridge_table)
                    seen_bridge_tables.add(bridge_table)
                if bridge_table not in table_metadata_map:
                    table_metadata_map[bridge_table] = build_table_metadata(bridge_table)

            sequence_parts = [path[0]]
            for source, target in zip(path, path[1:]):
                edge_data = graph.get_edge_data(source, target) or {}
                join_on = edge_data.get("join_on")
                join_column = str(edge_data.get("foreign_key_column") or "").strip() or "JOIN_KEY"
                if join_on and join_on not in seen_join_paths:
                    unique_join_paths.append(str(join_on))
                    seen_join_paths.add(str(join_on))
                sequence_parts.append(f"<--({join_column})--> {target}")

            join_sequence = "Required Join Path: " + " ".join(sequence_parts)
            if bridge_tables:
                join_sequence += (
                    "\nMandatory Instruction: To connect the tables above, you MUST include the bridge table"
                    f"{'s' if len(bridge_tables) > 1 else ''} {', '.join(bridge_tables)} in your JOIN even if you are not selecting columns from "
                    f"{'them' if len(bridge_tables) > 1 else 'it'}."
                )
            if join_sequence not in seen_join_sequences:
                required_join_sequences.append(join_sequence)
                seen_join_sequences.add(join_sequence)

        ordered_table_metadata: list[dict[str, Any]] = []
        for table_name in state["relevant_tables"] + required_bridge_tables:
            if table_name in table_metadata_map:
                ordered_table_metadata.append(table_metadata_map[table_name])

        next_error_log = state["error_log"]
        if join_errors:
            next_error_log = append_error(next_error_log, "; ".join(join_errors))

        return {
            **state,
            "required_bridge_tables": required_bridge_tables,
            "table_metadata": ordered_table_metadata,
            "join_paths": unique_join_paths,
            "required_join_sequences": required_join_sequences,
            "error_log": next_error_log,
        }
    except Exception as exc:
        return {
            **state,
            "required_bridge_tables": [],
            "join_paths": [],
            "required_join_sequences": [],
            "error_log": append_error(state["error_log"], f"Pathfinder error: {exc}"),
        }


def build_sql_prompt(state: AgentState) -> str:
    table_lines = []
    for table_info in state["table_metadata"]:
        columns = table_info.get("columns", [])
        column_preview = ", ".join(columns[:20]) if columns else "[unknown]"
        table_lines.append(
            "- "
            f"Table: {table_info.get('table_name')} | "
            f"Effective-Dated: {table_info.get('is_effective_dated')} | "
            f"Primary Key: {table_info.get('primary_key')} | "
            f"Module: {table_info.get('module')} | "
            f"Category: {table_info.get('category')} | "
            f"Columns: {column_preview}"
        )

    join_lines = [f"- {join_path}" for join_path in state["join_paths"]]
    join_guidance = "\n".join(join_lines) if join_lines else (
        "- No join paths were found in the graph. "
        "Use standard Oracle HCM joins very cautiously and only when necessary."
    )
    sequence_guidance = "\n\n".join(state["required_join_sequences"]) if state["required_join_sequences"] else (
        "Required Join Path: [No explicit multi-table route found]"
    )
    retry_guidance = ""
    parsed_corrections = parse_validator_json(state["corrections"]) if state["corrections"] else None
    if parsed_corrections and not parsed_corrections["is_valid"] and parsed_corrections["required_fix"]:
        retry_guidance = (
            "\nYour previous attempt failed. Follow this required fix exactly:\n"
            f"{parsed_corrections['required_fix']}\n"
        )

    return f"""You are a Senior Oracle HCM SQL Developer.

Write a single, valid Oracle SQL query to answer the user's question.
Use the provided join_paths to link tables. Do not hallucinate joins.
CRITICAL: You are ONLY allowed to join tables using the exact pairs listed in join_paths.
If a join is not in that list, you MUST NOT use it.
If you need to connect Table A to Table C, and the path is A -> B -> C,
you must include Table B in your FROM/JOIN clause even if you don't need its columns.
Only use tables from the provided relevant_tables list and required_bridge_tables list.
If multiple tables have a column named ORGANIZATION_ID or any other shared key, always alias your tables
and fully qualify the columns (for example, pa.ORGANIZATION_ID) to avoid ORA-00918.
When filtering by ID (for example, 12345), always look for the primary_key listed in the metadata first.
If that is not appropriate, then use PERSON_ID or ASSIGNMENT_ID.
Apply the user's numeric identifier to PERSON_ID or ASSIGNMENT_ID when those columns are the correct grounded choice.
CRITICAL: For any table where is_effective_dated is True, you MUST add a condition:
TRUNC(SYSDATE) BETWEEN effective_start_date AND effective_end_date
or the appropriate table-specific effective date columns.
Instruction: If is_effective_dated is False, DO NOT add a SYSDATE filter for that table. Doing so will break the query.
Output ONLY the raw SQL. No markdown code blocks, no explanations.
{retry_guidance}

User Question:
{state["user_question"]}

Relevant Tables:
{chr(10).join(table_lines) if table_lines else "- None"}

Required Bridge Tables:
{", ".join(state["required_bridge_tables"]) if state["required_bridge_tables"] else "[none]"}

Join Paths:
{join_guidance}

Required Join Sequences:
{sequence_guidance}
"""


def generate_sql(state: AgentState) -> AgentState:
    try:
        client = get_openrouter_client()
        prompt = build_sql_prompt(state)
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You produce Oracle SQL only.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        sql_query = (response.choices[0].message.content or "").strip()
        return {
            **state,
            "iteration_count": state["iteration_count"] + 1,
            "sql_query": sql_query,
            "response_text": "",
        }
    except Exception as exc:
        return {
            **state,
            "iteration_count": state["iteration_count"] + 1,
            "sql_query": "",
            "error_log": append_error(state["error_log"], f"SQL generator error: {exc}"),
        }


def build_validator_prompt(state: AgentState) -> str:
    table_lines = []
    for table_info in state["table_metadata"]:
        table_lines.append(
            "- "
            f"table_name={table_info.get('table_name')}, "
            f"module={table_info.get('module')}, "
            f"category={table_info.get('category')}, "
            f"primary_key={table_info.get('primary_key')}, "
            f"is_effective_dated={table_info.get('is_effective_dated')}"
        )

    join_lines = [f"- {join_path}" for join_path in state["join_paths"]]
    question_ids = extract_question_ids(state["user_question"])

    return f"""Act as a Senior Oracle SQL Auditor.

You must return ONLY a JSON object with exactly this structure:
{{
  "is_valid": boolean,
  "score": int,
  "errors": ["list of specific technical issues"],
  "required_fix": "A concise instruction for the next iteration"
}}

Do not return markdown. Do not return prose outside the JSON object.

Review the generated SQL for correctness against the user's question and rules below.

First, internally classify the query:
- Type A (Single Record): the question asks for a specific person by name or ID.
- Type B (Aggregate/List): the question asks for a group, department, organization, manager hierarchy, or list of records.

User Question:
{state["user_question"]}

Extracted User IDs:
{question_ids if question_ids else "[]"}

Generated SQL:
{state["sql_query"] or "[empty sql]"}

Relevant Tables:
{chr(10).join(table_lines) if table_lines else "- None"}

Allowed Join Paths:
{chr(10).join(join_lines) if join_lines else "- None"}

Strict Validation Rules:
Rule A (Tables): Only tables found in relevant_tables are allowed.
Bridge tables found in required_bridge_tables are also allowed when needed by the discovered path.
Rule B (Joins): Joins MUST match the provided join_paths.
Rule C (Effective Dating): If is_effective_dated is True in metadata, the SQL must have a date filter.
If is_effective_dated is False, it must not have an effective-date filter for that table.
Rule D (Filters): IF the question is Type A, ensure a WHERE clause filter exists for the specific person using PERSON_ID, ASSIGNMENT_ID, or the clearly grounded person/name predicate.
IF the question is Type B, DO NOT penalize the SQL for lacking a specific PERSON_ID filter.
For Type B department or group queries, expect an appropriate categorical filter such as department name, organization name, or an equivalent grouping filter.

Scoring guidance:
- 10 means fully valid and production-safe.
- 0 means completely unusable.

If valid:
- set is_valid to true
- set required_fix to an empty string

If invalid:
- set is_valid to false
- populate errors with specific technical issues
- set required_fix to one concise instruction the next SQL generation attempt must follow
"""


def validate_sql(state: AgentState) -> AgentState:
    try:
        client = get_openrouter_client()
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict Oracle SQL validator that returns JSON only.",
                },
                {
                    "role": "user",
                    "content": build_validator_prompt(state),
                },
            ],
        )
        validator_output = (response.choices[0].message.content or "").strip()
        try:
            validator_response = parse_validator_json(validator_output)
        except Exception:
            validator_response = {
                "is_valid": False,
                "score": 0,
                "errors": ["Validator returned malformed JSON."],
                "required_fix": "Return valid JSON and fix the SQL according to the validation rules.",
            }

        next_history = list(state["history"])
        next_history.append(
            {
                "iteration": state["iteration_count"],
                "sql": state["sql_query"],
                "validator_response": validator_response,
            }
        )

        next_sql_query = state["sql_query"]
        if not validator_response["is_valid"] and state["iteration_count"] >= 3:
            next_sql_query = add_warning_comment(next_sql_query)

        return {
            **state,
            "sql_query": next_sql_query,
            "corrections": json.dumps(validator_response),
            "history": next_history,
        }
    except Exception as exc:
        fallback = {
            "is_valid": False,
            "score": 0,
            "errors": [f"Validator execution failed: {exc}"],
            "required_fix": "Fix the SQL using only allowed tables, allowed joins, proper PERSON_ID or ASSIGNMENT_ID filters, and correct effective-date handling.",
        }
        next_history = list(state["history"])
        next_history.append(
            {
                "iteration": state["iteration_count"],
                "sql": state["sql_query"],
                "validator_response": fallback,
            }
        )
        next_sql_query = state["sql_query"]
        if state["iteration_count"] >= 3:
            next_sql_query = add_warning_comment(next_sql_query)
        return {
            **state,
            "sql_query": next_sql_query,
            "corrections": json.dumps(fallback),
            "history": next_history,
            "error_log": append_error(state["error_log"], f"Validator error: {exc}"),
        }


def should_continue(state: AgentState) -> str:
    validator_result = parse_validator_json(state["corrections"])
    if validator_result["is_valid"]:
        return "end"
    if state["iteration_count"] >= 3:
        return "end"
    return "generate_sql"


def print_agent_summary(state: AgentState) -> None:
    validator_result = parse_validator_json(state["corrections"])
    final_status = "VALID" if validator_result["is_valid"] else "FAILED"

    print("\n=== Agent Summary ===")
    print(f"User Question: {state['user_question']}")
    print(f"Tables Found: {', '.join(state['relevant_tables']) if state['relevant_tables'] else '[none]'}")
    print(f"Total Iterations: {state['iteration_count']}")
    print(f"Final Status: {final_status}")
    if state["saved_sql_path"]:
        print(f"Saved SQL File: {state['saved_sql_path']}")
    print("\nHistory Trace:")

    if not state["history"]:
        print(" - No history recorded.")
        return

    for entry in state["history"]:
        validator_response = entry.get("validator_response", {})
        errors = validator_response.get("errors", [])
        error_text = "; ".join(str(error) for error in errors) if errors else "[none]"

        print(f"\nIteration {entry.get('iteration')}:")
        print("SQL:")
        print(entry.get("sql") or "[empty sql]")
        print("Validator Errors:")
        print(error_text)


def build_agent():
    workflow = StateGraph(AgentState)
    workflow.add_node("route_question", route_question)
    workflow.add_node("answer_schema_question", answer_schema_question)
    workflow.add_node("retrieve_tables", retrieve_tables)
    workflow.add_node("find_joins", find_joins)
    workflow.add_node("generate_sql", generate_sql)
    workflow.add_node("validate_sql", validate_sql)
    workflow.set_entry_point("route_question")
    workflow.add_conditional_edges(
        "route_question",
        lambda state: "answer_schema_question" if state["query_mode"] == "schema" else "retrieve_tables",
        {
            "answer_schema_question": "answer_schema_question",
            "retrieve_tables": "retrieve_tables",
        },
    )
    workflow.add_edge("answer_schema_question", END)
    workflow.add_edge("retrieve_tables", "find_joins")
    workflow.add_edge("find_joins", "generate_sql")
    workflow.add_edge("generate_sql", "validate_sql")
    workflow.add_conditional_edges(
        "validate_sql",
        should_continue,
        {
            "generate_sql": "generate_sql",
            "end": END,
        },
    )
    return workflow.compile()


if __name__ == "__main__":
    agent = build_agent()
    initial_state: AgentState = {
        "user_question": "What is the salary for person 12345?",
        "relevant_tables": [],
        "required_bridge_tables": [],
        "table_metadata": [],
        "join_paths": [],
        "required_join_sequences": [],
        "sql_query": "",
        "saved_sql_path": "",
        "iteration_count": 0,
        "corrections": "",
        "history": [],
        "error_log": "",
    }
    final_state = agent.invoke(initial_state)

    print("Relevant tables:")
    for table in final_state["relevant_tables"]:
        print(f" - {table}")

    print("\nJoin paths:")
    for join_condition in final_state["join_paths"]:
        print(f" - {join_condition}")

    print("\nSQL query:")
    print(final_state["sql_query"] or "[not generated yet]")

    print("\nValidator output:")
    print(final_state["corrections"] or "[none]")

    print("\nIteration count:")
    print(final_state["iteration_count"])

    print("\nSaved SQL path:")
    print(final_state["saved_sql_path"] or "[not saved]")

    print("\nError log:")
    print(final_state["error_log"] or "[none]")

    print_agent_summary(final_state)
