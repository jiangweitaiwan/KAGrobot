# -*- coding: utf-8 -*-
"""app.py の変更差分メモ（PR1）
以下の箇所だけ変更する。ファイル全体は原本を維持。
"""

# ─────────────────────────────────────────────────────────────
# 変更 1: lifespan 内で app.state.driver と app.state.encoder を追加
# ─────────────────────────────────────────────────────────────
#
# 既存コード（変更なし）：
#   driver = GraphDatabase.driver(...)
#   encoder = SbertOnnxEncoder(model_path, tokenizer_dir)
#
# 追加するコード（yield の直前、app.state.orchestrator の後に追記）：
#
#   app.state.driver  = driver   # ← concept/detail endpoint が直接使う
#   app.state.encoder = encoder  # ← concept/suggest の SBERT prefilter が使う
#
# ─────────────────────────────────────────────────────────────
# 変更 2: create_app() で新しいルーターを include する
# ─────────────────────────────────────────────────────────────
#
# 既存コード：
#   from api.routes import chat_router, graph_router
#   app.include_router(chat_router)
#   app.include_router(graph_router)
#
# 変更後：
#   from api.routes import chat_router, graph_router, concept_router, user_brain_router
#   app.include_router(chat_router)
#   app.include_router(graph_router)
#   app.include_router(concept_router)    # ← 追加
#   app.include_router(user_brain_router) # ← 追加
#
# ─────────────────────────────────────────────────────────────
# 変更 3: requirements.txt に google-cloud-firestore を追加
# ─────────────────────────────────────────────────────────────
#
# google-cloud-firestore>=2.16.0

# ─────────────────────────────────────────────────────────────
# 下に「変更後の lifespan と create_app の完全版」を示す。
# 原本の他の部分はそのままコピーすること。
# ─────────────────────────────────────────────────────────────

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
from api.routes        import chat_router, graph_router, concept_router, user_brain_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    s        = get_settings()
    base_dir = Path(__file__).resolve().parent

    driver = GraphDatabase.driver(
        s.neo4j_uri, auth=("neo4j", s.neo4j_password)
    )

    model_path, tokenizer_dir = s.resolved_sbert_paths(base_dir=base_dir)
    encoder = SbertOnnxEncoder(model_path, tokenizer_dir)

    candidate_retriever = CandidateRetriever(driver, encoder)
    candidate_retriever.load_embeddings()

    concept_retriever = ConceptDetailRetriever(driver)
    path_selector     = PathSelector()
    downstream_bfs    = DownstreamBFS(driver)
    upstream_bfs      = UpstreamBFS(driver, path_selector)
    layout_builder    = GlobalLayoutBuilder(driver, encoder, s)

    scorer          = PathScorer()
    graph_expander  = GraphExpander(downstream_bfs, upstream_bfs, scorer)
    context_builder = GraphContextBuilder()

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

    nodes        = OrchestratorNodes(
        socrates_bot    = socrates_bot,
        main_agent      = main_agent,
        graph_expander  = graph_expander,
        context_builder = context_builder,
        skill_loader    = skill_loader,
    )
    orchestrator = build_orchestrator(nodes)

    # ── app.state ─────────────────────────────────────────────
    app.state.orchestrator   = orchestrator
    app.state.layout_builder = layout_builder
    app.state.layout_cache   = None
    app.state.settings       = s
    # ↓ PR1 で追加
    app.state.driver         = driver
    app.state.encoder        = encoder

    yield

    driver.close()


def create_app() -> FastAPI:
    s   = get_settings()
    app = FastAPI(lifespan=lifespan)

    origins = [o.strip() for o in s.cors_allow_origins.split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    if s.langchain_tracing_v2:
        os.environ["LANGCHAIN_TRACING_V2"] = s.langchain_tracing_v2
    if s.langchain_endpoint:
        os.environ["LANGCHAIN_ENDPOINT"]   = s.langchain_endpoint
    if s.langchain_project:
        os.environ["LANGCHAIN_PROJECT"]    = s.langchain_project

    # ↓ PR1 で追加: concept_router, user_brain_router
    app.include_router(chat_router)
    app.include_router(graph_router)
    app.include_router(concept_router)
    app.include_router(user_brain_router)

    return app
