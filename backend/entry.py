# -*- coding: utf-8 -*-
"""entry.py  — FastAPI + LangGraph orchestrator."""

from __future__ import annotations

import asyncio
import os
from settings import get_settings

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"]   = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"]    = "main"

from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

from core.socrates_bot import (
    SocratesChatBot,
    ChatState,
    ConversationPhase,
    _extract_selected_chain,
    _format_selected_chain,
)
from neo4jreasoningretriever import Neo4jReasoningRetriever
from core.main_agent import MainAgent


# ======================================================================
# Orchestrator state
# ======================================================================

class OrchestratorState(TypedDict, total=False):
    messages:          Annotated[List[BaseMessage], add_messages]
    phase:             str
    socrates_state:    ChatState
    main_state:        Annotated[List[BaseMessage], add_messages]
    handoff_task:      Optional[str]
    selected_path_id:  Optional[int]
    confirmed_path_id: Optional[int]
    all_graphs:        Optional[Dict[int, Any]]
    graph_context:     Optional[str]


# ======================================================================
# Singletons
# ======================================================================

def _build_retriever() -> Neo4jReasoningRetriever:
    s        = get_settings()
    base_dir = Path(__file__).resolve().parent
    model_path    = str(base_dir / "sbert_onnx" / "sbert_multilingual.onnx")
    tokenizer_dir = str(base_dir / "sbert_onnx")
    retriever = Neo4jReasoningRetriever(
        s.neo4j_uri, s.neo4j_password, model_path, tokenizer_dir)
    try:
        retriever.load_embeddings()
    except Exception:
        pass
    return retriever


shared_retriever    = _build_retriever()
s                   = get_settings()
shared_socrates_bot = SocratesChatBot(
    retriever=shared_retriever,
    model=s.socrates_model,
    api_key=s.google_api_key,
)
shared_main_agent = MainAgent(
    model=s.main_agent_model,
    api_key=s.google_api_key,
)


# ======================================================================
# Pure helpers
# ======================================================================

def _last_human_content(messages: List[BaseMessage]) -> Optional[str]:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return None


def _find_path_by_id(
    graph_data: Optional[Dict[str, Any]],
    path_id: int,
) -> Optional[Dict[str, Any]]:
    if not graph_data:
        return None
    for path in graph_data.get("paths", []):
        if path.get("path_id") == path_id:
            return path
    return None


def _annotate_graph_with_default_path(
    graph_data: Dict[str, Any],
    rc_name:    str,
) -> Dict[str, Any]:
    """在 graph_data 中找出第一條 downstream path 作為預設標記。"""
    downstream_paths = [
        p for p in graph_data.get("paths", [])
        if p.get("direction", "downstream") == "downstream"
    ]

    selected_path_id: Optional[int] = None

    rc_lower = rc_name.lower()
    for path in downstream_paths:
        if (path.get("final_label") or "").lower() == rc_lower:
            selected_path_id = path["path_id"]
            break

    if selected_path_id is None and downstream_paths:
        selected_path_id = downstream_paths[0]["path_id"]

    if selected_path_id is None:
        all_paths = graph_data.get("paths", [])
        if all_paths:
            selected_path_id = all_paths[0]["path_id"]

    return {**graph_data, "selected_path_id": selected_path_id}


def _score_paths_by_intent(
    graph_data:   Dict[str, Any],
    intent_types: List[str],
) -> Dict[str, Any]:
    """對 graph_data 中每條 path 計算 intent_score，並依分數重新排序。

    排序規則：intent_score 降冪；同分時以 node_ids 長度降冪（較長路徑優先）。
    排序完成後重設 path_id（0, 1, 2, ...），selected_path_id = 0。
    """
    if not intent_types or not graph_data:
        return graph_data

    nodes_by_id: Dict[str, Any] = {
        n["id"]: n for n in graph_data.get("nodes", [])
    }

    def _type_overlap(node_analysis_types: List[str]) -> float:
        if not node_analysis_types:
            return 0.0
        matched = len(set(node_analysis_types) & set(intent_types))
        return matched / len(intent_types)

    scored_paths = []
    for path in graph_data.get("paths", []):
        final_id   = path.get("final_concept")
        final_node = nodes_by_id.get(final_id or "", {})
        type_score = _type_overlap(final_node.get("analysis_types", []))
        scored_paths.append({**path, "intent_score": type_score})

    scored_paths.sort(
        key=lambda p: (p["intent_score"], len(p.get("node_ids", []))),
        reverse=True,
    )
    for new_id, path in enumerate(scored_paths):
        path["path_id"] = new_id

    return {**graph_data, "paths": scored_paths, "selected_path_id": 0}


def _build_graph_context(all_graphs: Dict[int, Any]) -> str:
    """將 all_graphs 序列化為 Socrates system prompt 用的文字。

    每個 candidate 只列出：
      - candidate RC 名稱
      - 下游終點（top-3 path 的 final_label，去重）
      - 基礎概念（foundations label，去重）
    """
    lines: List[str] = []

    for idx, g in sorted(all_graphs.items()):
        if not g:
            lines.append(f"【候選 {idx}】（圖譜展開失敗）")
            continue

        concept_name = (g.get("query_concepts") or [{}])[0].get("name", f"選項 {idx}")
        lines.append(f"【候選 {idx}】{concept_name}")

        # 下游終點：paths 已經由 _score_paths_by_intent 或 _trim_to_top_paths
        # 排序並限制為 top-3，直接取 final_label
        proj = Neo4jReasoningRetriever.project_concepts_only(g)
        downstream_finals: List[str] = []
        seen_finals: set = set()
        for path in proj.get("paths", []):
            label = path.get("final_label", "")
            if label and label not in seen_finals:
                seen_finals.add(label)
                downstream_finals.append(label)

        if downstream_finals:
            lines.append(f"  下游終點：{', '.join(downstream_finals)}")
        else:
            lines.append("  下游終點：（無）")

        # 基礎概念：foundations（leaf concepts）
        foundations = g.get("foundations") or []
        foundation_labels = [f["label"] for f in foundations if f.get("label")]
        if foundation_labels:
            lines.append(f"  基礎概念：{', '.join(foundation_labels)}")
        else:
            lines.append("  基礎概念：（無）")

        lines.append("")

    return "\n".join(lines).rstrip()


def _build_handoff_task(
    socrates_state: ChatState,
    selected_path:  Optional[Dict[str, Any]] = None,
    all_graphs:     Optional[Dict[int, Any]] = None,
) -> str:
    query = socrates_state.get("query") or "（未提供）"

    if selected_path and all_graphs:
        model_name  = selected_path.get("final_label") or "（未指定）"

        nodes_by_id: Dict[str, Any] = {}
        for g in all_graphs.values():
            if g:
                for n in g.get("nodes", []):
                    nodes_by_id[n["id"]] = n

        path_labels: List[str] = []
        for nid in (selected_path.get("node_ids") or []):
            node = nodes_by_id.get(nid)
            if node and node.get("node_type") == "concept":
                path_labels.append(node.get("label", nid))
        path_description = " → ".join(path_labels) if path_labels else model_name

        paper_info_lines: List[str] = []
        for nid in (selected_path.get("node_ids") or []):
            node = nodes_by_id.get(nid)
            if node and node.get("node_type") == "paper":
                meta        = node.get("meta") or {}
                line        = f"  論文：{node.get('label', '')}"
                premises    = meta.get("conceptual_premises") or ""
                constraints = meta.get("contextual_model_constraints") or ""
                if premises:
                    line += f"\n    前提假設：{premises}"
                if constraints:
                    line += f"\n    模型約束：{constraints}"
                paper_info_lines.append(line)
        paper_section = "\n".join(paper_info_lines) if paper_info_lines else "（無論文資訊）"

    else:
        candidates   = socrates_state.get("candidates") or []
        model_name   = candidates[0].get("rc_name", "（未指定）") if candidates else "（未指定）"
        path_description = model_name
        paper_section    = "（無路徑資訊）"

    tools_info    = socrates_state.get("tools") or []
    primary_tools = [t["tool_name"] for t in tools_info if not t.get("is_dependency", False)]
    tools_str     = "、".join(primary_tools) if primary_tools else "（無）"

    return (
        f"使用者需求：{query}\n"
        f"選定模型：{model_name}\n"
        f"推理路徑：{path_description}\n"
        f"關聯論文資訊：\n{paper_section}\n"
        f"可用工具：{tools_str}\n"
        f"請根據以上需求、推理路徑與工具，開始執行分析。"
    )


def _find_graph_by_path_id(
    all_graphs: Dict[int, Any],
    path_id: int,
) -> Optional[Dict[str, Any]]:
    for g in all_graphs.values():
        if not g:
            continue
        for path in g.get("paths", []):
            if path.get("path_id") == path_id:
                return g
    return None


# ======================================================================
# Graph expansion helper（並行）
# ======================================================================

async def _expand_all_graphs(
    candidates: List[dict],
) -> Dict[int, Any]:

    TOP_N = 3

    def _trim_to_top_paths(graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """保留 intent_score 最高的 top-3 下游路徑，重設 path_id。"""
        all_paths = graph_data.get("paths") or []
        if not all_paths:
            return graph_data

        sorted_paths = sorted(
            all_paths,
            key=lambda p: (p.get("intent_score", 0.0), len(p.get("node_ids", []))),
            reverse=True,
        )
        top_paths = sorted_paths[:TOP_N]

        for new_id, path in enumerate(top_paths):
            path["path_id"] = new_id

        used_node_ids: set = set()
        used_edge_ids: set = set()
        for path in top_paths:
            used_node_ids.update(path.get("node_ids") or [])
            used_edge_ids.update(path.get("edge_ids") or [])

        trimmed_nodes = [
            n for n in (graph_data.get("nodes") or [])
            if n["id"] in used_node_ids
        ]
        trimmed_edges = [
            e for e in (graph_data.get("edges") or [])
            if e["id"] in used_edge_ids
        ]

        return {
            **graph_data,
            "paths":            top_paths,
            "nodes":            trimmed_nodes,
            "edges":            trimmed_edges,
            "selected_path_id": 0,
        }

    async def _expand_one(cand: dict) -> tuple[int, Any]:
        idx     = cand["index"]
        eid     = cand.get("eid") or ""
        rc_name = cand.get("rc_name") or ""

        if not eid:
            return idx, None

        try:
            # 下游圖譜與上游 foundations 並行跑
            raw_graph, foundations = await asyncio.gather(
                asyncio.to_thread(
                    shared_retriever.retrieve_graph,
                    eid,
                    rc_name,
                ),
                asyncio.to_thread(
                    shared_retriever.retrieve_foundations,
                    eid,
                ),
            )
            raw_graph["foundations"] = foundations
            trimmed = _trim_to_top_paths(raw_graph)
            return idx, trimmed
        except Exception as exc:
            print(f"[_expand_all_graphs] failed for {rc_name}: {exc}")
            return idx, None

    results = await asyncio.gather(*[_expand_one(c) for c in candidates[:3]])
    return dict(results)


# ======================================================================
# LangGraph nodes
# ======================================================================

async def socratic_node(state: OrchestratorState) -> OrchestratorState:
    messages         = state.get("messages") or []
    user_msg         = _last_human_content(messages)
    selected_path_id = state.get("selected_path_id")
    current_phase    = state.get("phase") or "socratic"

    if not user_msg:
        return {}

    old_socrates_state = state.get("socrates_state") or SocratesChatBot.init_state()

    # ── 情況 1：使用者確認 path ───────────────────────────────
    if selected_path_id is not None and current_phase == "concept_resolved":
        new_socrates_state = shared_socrates_bot.confirm_path(old_socrates_state)
        return OrchestratorState(
            messages       = [AIMessage(content="路徑已確認，開始執行分析…")],
            phase          = "path_confirmed",
            socrates_state = new_socrates_state,
        )

    # ── 一般 Socrates 對話 ────────────────────────────────────
    graph_context = (
        state.get("graph_context")
        if current_phase == "concept_resolved"
        else None
    )
    all_graphs_full = state.get("all_graphs") if current_phase == "concept_resolved" else None

    new_socrates_state = shared_socrates_bot.process_message(
        old_socrates_state,
        user_msg,
        graph_context=graph_context,
        all_graphs=all_graphs_full,
    )

    inner_phase = new_socrates_state.get("phase")
    reply_text  = shared_socrates_bot.get_last_reply(new_socrates_state)

    # retrieve_tool 被呼叫 → concept_resolved → 並行展開圖譜
    if inner_phase == ConversationPhase.CONCEPT_RESOLVED.value:
        candidates   = new_socrates_state.get("candidates") or []
        intent_types = new_socrates_state.get("intent_types") or []
        all_graphs   = await _expand_all_graphs(candidates)

        # 依 intent_types 對每個 candidate 的 path 做評分與排序
        if intent_types:
            all_graphs = {
                idx: _score_paths_by_intent(g, intent_types)
                for idx, g in all_graphs.items()
                if g is not None
            }

        new_graph_context = _build_graph_context(all_graphs)

        return OrchestratorState(
            messages       = [AIMessage(content=reply_text)],
            phase          = "concept_resolved",
            socrates_state = new_socrates_state,
            all_graphs     = all_graphs,
            graph_context  = new_graph_context,
        )

    return OrchestratorState(
        messages       = [AIMessage(content=reply_text)],
        phase          = current_phase,
        socrates_state = new_socrates_state,
    )


async def main_agent_node(state: OrchestratorState) -> OrchestratorState:
    socrates_state = state.get("socrates_state")
    if not socrates_state:
        return OrchestratorState(
            messages=[AIMessage(content="目前尚未完成需求釐清，請先重新描述你的需求。")],
            phase="socratic",
        )

    selected_path_id  = state.get("selected_path_id")
    confirmed_path_id = state.get("confirmed_path_id")
    all_graphs        = state.get("all_graphs")

    is_reselect = (
        selected_path_id is not None
        and selected_path_id != confirmed_path_id
    )

    if is_reselect:
        target_graph  = _find_graph_by_path_id(all_graphs or {}, selected_path_id)
        selected_path = _find_path_by_id(target_graph, selected_path_id)
        user_msg      = _build_handoff_task(socrates_state, selected_path, all_graphs)
        history: List[BaseMessage] = []
        path_label    = (selected_path or {}).get("final_label") or f"路徑 {selected_path_id}"
        reselect_notice = AIMessage(
            content=f"已切換至新路徑「{path_label}」，重新開始分析。"
        )
    else:
        user_msg        = _last_human_content(state.get("messages") or [])
        history         = list(state.get("main_state") or [])
        reselect_notice = None

    if not user_msg:
        return {}

    result = await asyncio.to_thread(
        shared_main_agent.chat,
        user_msg,
        socrates_state,
        history,
    )

    output_messages: List[BaseMessage] = []
    if reselect_notice:
        output_messages.append(reselect_notice)
    output_messages.append(AIMessage(content=result["content"]))

    return OrchestratorState(
        messages          = output_messages,
        phase             = "main_agent",
        socrates_state    = socrates_state,
        handoff_task      = None,
        main_state        = result.get("messages") or [],
        confirmed_path_id = selected_path_id if is_reselect else confirmed_path_id,
    )


# ======================================================================
# LangGraph routers
# ======================================================================

def entry_router(state: OrchestratorState) -> str:
    phase = state.get("phase") or "socratic"
    if phase in ("socratic", "concept_resolved"):
        return "socratic"
    if phase in ("path_confirmed", "main_agent"):
        return "main_agent"
    return "socratic"


def socratic_router(state: OrchestratorState) -> str:
    phase = state.get("phase") or "socratic"
    if phase == "path_confirmed":
        return "main_agent"
    return "end"


# ======================================================================
# Build workflow
# ======================================================================

workflow = StateGraph(OrchestratorState)
workflow.add_node("socratic",   socratic_node)
workflow.add_node("main_agent", main_agent_node)

workflow.set_conditional_entry_point(
    entry_router,
    {
        "socratic":   "socratic",
        "main_agent": "main_agent",
    },
)
workflow.add_conditional_edges(
    "socratic", socratic_router,
    {
        "main_agent": "main_agent",
        "end":        END,
    },
)
workflow.add_edge("main_agent", END)

memory           = MemorySaver()
orchestrator_app = workflow.compile(checkpointer=memory)


# ======================================================================
# FastAPI
# ======================================================================

class ChatRequest(BaseModel):
    session_id:       str
    message:          str
    phase:            Optional[str] = None
    selected_path_id: Optional[int] = None


class ChatResponse(BaseModel):
    reply:          str
    phase:          str
    socrates_phase: Optional[str]            = None
    all_graphs:     Optional[Dict[int, Any]] = None


api = FastAPI()

cors_origins = [o.strip() for o in s.cors_allow_origins.split(",") if o.strip()]
if cors_origins:
    api.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@api.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    config        = {"configurable": {"thread_id": request.session_id}}
    initial_phase = request.phase or "socratic"

    invoke_input: dict = {
        "messages": [HumanMessage(content=request.message)],
        "phase":    initial_phase,
    }
    if request.selected_path_id is not None:
        invoke_input["selected_path_id"] = request.selected_path_id

    result = await orchestrator_app.ainvoke(invoke_input, config=config)

    messages   = result.get("messages") or []
    last_msg   = messages[-1] if messages else AIMessage(content="")
    reply_text = getattr(last_msg, "content", str(last_msg))

    socrates_state = result.get("socrates_state")
    socrates_phase = (
        socrates_state.get("phase")
        if socrates_state and isinstance(socrates_state, dict)
        else None
    )

    all_graphs_out: Optional[Dict[int, Any]] = None
    if (
        result.get("phase") == "concept_resolved"
        and request.selected_path_id is None
    ):
        raw_all_graphs = result.get("all_graphs") or {}
        all_graphs_out = {
            idx: (
                Neo4jReasoningRetriever.project_concepts_only(g)
                if g else None
            )
            for idx, g in raw_all_graphs.items()
        }

    return ChatResponse(
        reply          = reply_text,
        phase          = result.get("phase", initial_phase),
        socrates_phase = socrates_phase,
        all_graphs     = all_graphs_out,
    )
