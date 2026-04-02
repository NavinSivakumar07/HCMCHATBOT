#!/usr/bin/env python3
"""Streamlit dashboard for the Oracle HCM SQL Agent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st

from env_config import load_env_file


load_env_file()

for secret_key in (
    "OPENROUTER_API_KEY",
    "PINECONE_API_KEY",
    "PINECONE_INDEX_NAME",
    "OPENROUTER_MODEL",
    "HCM_SCHEMA_JSON_PATH",
    "HCM_GRAPH_PATH",
):
    if secret_key in st.secrets and secret_key not in os.environ:
        os.environ[secret_key] = str(st.secrets[secret_key])

import hcm_sql_agent as agent


st.set_page_config(
    page_title="Oracle HCM Intelligence Portal",
    page_icon="🏛️",
    layout="wide",
)


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def clear_session() -> None:
    st.session_state.messages = []
    st.rerun()


def build_initial_state(question: str) -> agent.AgentState:
    return {
        "user_question": question,
        "query_mode": "",
        "relevant_tables": [],
        "required_bridge_tables": [],
        "table_metadata": [],
        "join_paths": [],
        "required_join_sequences": [],
        "sql_query": "",
        "response_text": "",
        "saved_sql_path": "",
        "iteration_count": 0,
        "corrections": "",
        "history": [],
        "error_log": "",
    }


def render_result(result: dict[str, Any], message_index: int) -> None:
    if result.get("query_mode") == "schema":
        st.markdown(result.get("response_text") or "No schema answer available.")
        if result.get("relevant_tables"):
            with st.expander("Matched Objects", expanded=True):
                for table in result["relevant_tables"]:
                    st.write(f"- `{table}`")
        if result.get("error_log"):
            with st.expander("Diagnostics"):
                st.text(result["error_log"])
        return

    validator = agent.parse_validator_json(result.get("corrections", ""))
    is_valid = bool(validator.get("is_valid"))
    score = validator.get("score", 0)

    if is_valid:
        st.success(f"Validation passed with score {score}/10.")
    else:
        reason = "; ".join(validator.get("errors", [])) or "The agent hit the retry cap."
        st.warning(f"Validation failed: {reason}")

    with st.expander("Retrieval", expanded=True):
        tables = result.get("relevant_tables", [])
        if tables:
            for table in tables:
                st.write(f"- `{table}`")
        else:
            st.write("No tables were retrieved.")

    with st.expander("Pathfinding", expanded=True):
        sequences = result.get("required_join_sequences", [])
        bridges = result.get("required_bridge_tables", [])
        if bridges:
            st.caption("Bridge tables")
            st.write(", ".join(f"`{table}`" for table in bridges))
        if sequences:
            for sequence in sequences:
                st.text(sequence)
        else:
            st.write("No join sequences were found.")

    with st.expander("Self-Correction", expanded=True):
        st.write(f"Retries attempted: `{result.get('iteration_count', 0)}`")
        history = result.get("history", [])
        if history:
            for item in history:
                validator_response = item.get("validator_response", {})
                errors = validator_response.get("errors", [])
                with st.container():
                    st.markdown(f"**Iteration {item.get('iteration', '?')}**")
                    st.code(item.get("sql", "[empty sql]"), language="sql")
                    if errors:
                        for error in errors:
                            st.write(f"- {error}")
                    else:
                        st.write("- No validator errors.")
        else:
            st.write("No retry history recorded.")

    st.subheader("Final SQL")
    sql_query = result.get("sql_query", "")
    st.code(sql_query or "-- No SQL generated", language="sql")
    st.download_button(
        "Download SQL",
        data=(sql_query or "") + ("\n" if sql_query else ""),
        file_name=Path(result.get("saved_sql_path") or "generated_query.sql").name,
        mime="text/sql",
        key=f"download_sql_{message_index}",
    )

    saved_path = result.get("saved_sql_path")
    if saved_path:
        st.caption(f"Saved to `{saved_path}`")

    if result.get("error_log"):
        with st.expander("Diagnostics"):
            st.text(result["error_log"])


def main() -> None:
    init_session_state()

    st.title("🏛️ Oracle HCM Intelligence Portal")
    st.caption("Agentic SQL Generation with Graph-Based Validation")

    with st.sidebar:
        st.header("Session")

        if st.button("Clear Session", use_container_width=True):
            clear_session()

        st.divider()
        st.subheader("Environment")
        st.write("API keys are loaded from the local `.env` file.")
        st.caption("Expected keys: `OPENROUTER_API_KEY`, `PINECONE_API_KEY`")

        st.divider()
        st.subheader("About")
        st.write(
            "This app uses NetworkX for join-path discovery and LangGraph for "
            "self-correcting SQL generation and validation."
        )

    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                render_result(message["result"], index)

    user_question = st.chat_input("Ask an Oracle HCM question...")
    if not user_question:
        return

    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        try:
            if not os.environ.get("OPENROUTER_API_KEY"):
                raise ValueError("OpenRouter API key is missing. Add it to the local .env file.")

            with st.status("Running Oracle HCM agent...", expanded=True) as status:
                status.write("Compiling LangGraph workflow...")
                workflow = agent.build_agent()

                status.write("Invoking retriever, pathfinder, validator, and SQL generator...")
                result = workflow.invoke(build_initial_state(user_question))

                validator = agent.parse_validator_json(result.get("corrections", ""))
                if validator.get("is_valid"):
                    status.update(label="Agent completed successfully", state="complete", expanded=False)
                else:
                    status.update(label="Agent completed with validation warnings", state="error", expanded=True)

            render_result(result, len(st.session_state.messages))
            st.session_state.messages.append({"role": "assistant", "result": result})
        except Exception as exc:
            st.error(
                "The Oracle HCM agent could not complete your request. "
                "Check that your .env file is configured and the graph is available."
            )
            st.caption(str(exc))


if __name__ == "__main__":
    main()
