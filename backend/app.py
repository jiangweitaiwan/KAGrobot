# -*- coding: utf-8 -*-
"""api/app.py — FastAPI application factory with lifespan。

所有 singleton 的初始化與關閉集中在 lifespan context manager，
掛載於 app.state，由 route handler 透過 request.app.state 取用。

app.state 欄位清單：
    orchestrator     — compiled LangGraph app
    layout_builder   — GlobalLayoutBuilder
    layout_cache     — Optional[dict]，全域佈局快取
    settings         — Settings
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from neo4j import GraphDatabase

from settings import get_settings
from retrieval import (
    SbertOnnxEncoder,
    CandidateRetriever,
    ConceptDetailRetriever,
    DownstreamBFS,
    UpstreamBFS,
    PathSelector,
    GlobalLayoutBuilder,
)
from graph_service import GraphExpander, PathScorer, GraphContextBuilder
from core.socrates_bot import SocratesChatBot
from core.main_agent   import MainAgent
from core.skill_loader import SkillLoader
from orchestrator      import OrchestratorNodes, build_orchestrator
from api.routes        import chat_router, graph_router


# ======================================================================
# Lifespan（singleton 初始化 / 關閉）
# ======================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    s        = get_settings()
    base_dir = Path(__file__).resolve().parent

    # ── Neo4j driver ──────────────────────────────────────────
    driver = GraphDatabase.driver(
        s.neo4j_uri, auth=("neo4j", s.neo4j_password)
    )

    # ── SBERT encoder ─────────────────────────────────────────
    model_path, tokenizer_dir = s.resolved_sbert_paths(base_dir=base_dir)
    encoder = SbertOnnxEncoder(model_path, tokenizer_dir)

    # ── Retrieval 層 ──────────────────────────────────────────
    candidate_retriever = CandidateRetriever(driver, encoder)
    candidate_retriever.load_embeddings()

    concept_retriever = ConceptDetailRetriever(driver)

    path_selector  = PathSelector()
    downstream_bfs = DownstreamBFS(driver)
    upstream_bfs   = UpstreamBFS(driver, path_selector)

    layout_builder = GlobalLayoutBuilder(driver, encoder, s)

    # ── Graph service 層 ──────────────────────────────────────
    scorer          = PathScorer()
    graph_expander  = GraphExpander(downstream_bfs, upstream_bfs, scorer)
    context_builder = GraphContextBuilder()

    # ── Core agents ───────────────────────────────────────────
    socrates_bot = SocratesChatBot(
        candidate_retriever=candidate_retriever,
        concept_retriever=concept_retriever,
        model=s.socrates_model,
    )
    main_agent   = MainAgent(model=s.main_agent_model)
    skill_loader = SkillLoader(
        bucket_name=s.skills_bucket_name,
        gcs_prefix="skills",
    )

    # ── Orchestrator ──────────────────────────────────────────
    nodes        = OrchestratorNodes(
        socrates_bot    = socrates_bot,
        main_agent      = main_agent,
        graph_expander  = graph_expander,
        context_builder = context_builder,
        skill_loader    = skill_loader,
    )
    orchestrator = build_orchestrator(nodes)

    # ── 掛上 app.state ────────────────────────────────────────
    app.state.orchestrator   = orchestrator
    app.state.layout_builder = layout_builder
    app.state.layout_cache   = None          # 首次 GET /graph/global-layout 時填入
    app.state.settings       = s

    yield  # ── 服務運行中 ──

    # ── Shutdown ──────────────────────────────────────────────
    driver.close()


# ======================================================================
# FastAPI app factory
# ======================================================================

def create_app() -> FastAPI:
    """建立並回傳 FastAPI app（供 entry.py 與測試使用）。"""
    s = get_settings()

    app = FastAPI(lifespan=lifespan)

    # CORS
    origins = [o.strip() for o in s.cors_allow_origins.split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # LangSmith tracing
    if s.langchain_tracing_v2:
        os.environ["LANGCHAIN_TRACING_V2"] = s.langchain_tracing_v2
    if s.langchain_endpoint:
        os.environ["LANGCHAIN_ENDPOINT"]   = s.langchain_endpoint
    if s.langchain_project:
        os.environ["LANGCHAIN_PROJECT"]    = s.langchain_project

    # Routes
    app.include_router(chat_router)
    app.include_router(graph_router)

    return app
