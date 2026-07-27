# -*- coding: utf-8 -*-
"""app.py — FastAPI application factory（更新版）

與原版的差異：
  1. 新增 AdjacencyCache、BFSCoordinator 的初始化與注入
  2. GraphExpander 改接 BFSCoordinator（不再直接注入 downstream/upstream bfs）
  3. OrchestratorNodes 仍注入 graph_expander（介面不變）
  4. app.state 新增 coordinator
  5. 移除 app.state.expand_cache / app.state.rcpath_cache
     （已統一由 BFSCoordinator 管理）
  6. DownstreamBFS / UpstreamBFS 仍保留在 app.state，
     供非 coordinator 路徑使用（未來可逐步遷移）
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
    PathSelector,
    GlobalLayoutBuilder,
)
from graph_service.bfs_cache       import AdjacencyCache
from graph_service.bfs_coordinator import BFSCoordinator
from graph_service.foundation_recommender import FoundationRecommender
from core.socrates_bot             import SocratesChatBot
from core.main_agent               import MainAgent
from core.skill_loader             import SkillLoader
from orchestrator                  import OrchestratorNodes, build_orchestrator
from api.routes                    import (
    chat_router,
    graph_router,
    concept_router,
    user_brain_router,
    brain_router,
)

from core.skill_sandbox import SkillSandboxProvisioner

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

    layout_builder = GlobalLayoutBuilder(driver, encoder, s)

    # ── 新增：鄰接層 cache + BFS Coordinator ─────────────────
    adjacency_cache = AdjacencyCache(
        driver=driver,
        maxsize=500,
        ttl_s=7 * 24 * 3600,   # 7 天
    )
    coordinator = BFSCoordinator(
        adjacency_cache=adjacency_cache,
        path_selector=path_selector,
    )



    socrates_bot = SocratesChatBot(
        candidate_retriever=candidate_retriever,
        concept_retriever=concept_retriever,
        model=s.socrates_model,
    )

    sandbox_provisioner = SkillSandboxProvisioner(
        api_key=s.e2b_api_key,
        template=s.e2b_skill_template,
    )
    main_agent = MainAgent(
        model=s.main_agent_model,
        sandbox_provisioner=sandbox_provisioner,
        skill_outputs_bucket=s.skill_outputs_bucket_name,   # ← 新增
    )

    skill_loader = SkillLoader(
        bucket_name=s.skills_bucket_name,
        gcs_prefix="skills",
    )
    
    nodes = OrchestratorNodes(
        socrates_bot        = socrates_bot,
        main_agent          = main_agent,
        skill_loader         = skill_loader,
        sandbox_provisioner  = sandbox_provisioner,
        coordinator          = coordinator,   # ← 新增
    )
    orchestrator = build_orchestrator(nodes)
    foundation_recommender = FoundationRecommender(
        coordinator=coordinator,
        encoder=encoder,
    )

    # ── app.state ─────────────────────────────────────────────
    app.state.orchestrator    = orchestrator
    app.state.layout_builder  = layout_builder
    app.state.layout_cache    = None
    app.state.settings        = s
    app.state.driver          = driver
    app.state.encoder         = encoder
    app.state.coordinator     = coordinator     # 新增：統一 BFS 排程器
    app.state.candidate_retriever   = candidate_retriever      # ← 新增
    app.state.foundation_recommender = foundation_recommender  # ← 新增

    # detail_cache 維持不變（Concept 詳情查詢，與 BFS 無關）
    app.state.detail_cache = {}
    app.state.bertopic_layout = None   # GET /graph/bertopic-layout 的結果快取
    # 注意：expand_cache / rcpath_cache 已移入 BFSCoordinator，
    # 此處不再設置，routes_brain.py 也已更新為使用 coordinator

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

    app.include_router(chat_router)
    app.include_router(graph_router)
    app.include_router(concept_router)
    app.include_router(user_brain_router)
    app.include_router(brain_router)

    return app
