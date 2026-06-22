# -*- coding: utf-8 -*-
"""app.py — FastAPI application factory."""

from __future__ import annotations

import os
import logging
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
    GlobalLayoutBuilder,
    GlobalPredecessorGraph,
    UpstreamBFSFromCache,
    DownstreamBFSFromCache,
    PathSelector,
)
from graph_service    import GraphExpander, PathScorer, GraphContextBuilder
from core.socrates_bot import SocratesChatBot
from core.main_agent   import MainAgent
from core.skill_loader import SkillLoader
from orchestrator      import OrchestratorNodes, build_orchestrator
from api.routes        import (
    chat_router,
    graph_router,
    concept_router,
    user_brain_router,
    brain_router,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    s        = get_settings()
    base_dir = Path(__file__).resolve().parent

    logger.info("[App] Starting up...")
    
    driver = GraphDatabase.driver(
        s.neo4j_uri, auth=("neo4j", s.neo4j_password)
    )

    model_path, tokenizer_dir = s.resolved_sbert_paths(base_dir=base_dir)
    encoder = SbertOnnxEncoder(model_path, tokenizer_dir)

    candidate_retriever = CandidateRetriever(driver, encoder)
    candidate_retriever.load_embeddings()

    concept_retriever = ConceptDetailRetriever(driver)
    path_selector     = PathSelector()
    layout_builder    = GlobalLayoutBuilder(driver, encoder, s)

    scorer          = PathScorer()
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

    # ✨ 初始化全局前驱图缓存（Lazy Loading）
    logger.info("[App] Initializing GlobalPredecessorGraph...")
    pred_graph = GlobalPredecessorGraph(
        driver=driver,
        ttl_hours=int(os.environ.get("PRED_GRAPH_TTL_HOURS", "24")),
        max_lazy_batch=int(os.environ.get("PRED_GRAPH_MAX_BATCH", "100")),
    )
    
    try:
        pred_graph.initialize()
        logger.info("[App] GlobalPredecessorGraph initialized successfully")
    except Exception as e:
        logger.error(f"[App] Failed to initialize PredecessorGraph: {e}")
        # 继续启动，但 cached BFS 可能不可用
    
    # ✨ 创建基于缓存的 BFS
    upstream_bfs_cached = UpstreamBFSFromCache(pred_graph, path_selector)
    downstream_bfs_cached = DownstreamBFSFromCache(pred_graph)
    
    # GraphExpander 使用缓存版本
    graph_expander = GraphExpander(
        downstream_bfs=downstream_bfs_cached,
        upstream_bfs=upstream_bfs_cached,
        scorer=scorer,
    )

    nodes = OrchestratorNodes(
        socrates_bot    = socrates_bot,
        main_agent      = main_agent,
        graph_expander  = graph_expander,
        context_builder = context_builder,
        skill_loader    = skill_loader,
    )
    orchestrator = build_orchestrator(nodes)

    # ── app.state ─────────────────────────────────────────────
    app.state.orchestrator           = orchestrator
    app.state.layout_builder         = layout_builder
    app.state.layout_cache           = None
    app.state.settings               = s
    app.state.driver                 = driver
    app.state.encoder                = encoder
    app.state.pred_graph             = pred_graph
    app.state.upstream_bfs_cached    = upstream_bfs_cached
    app.state.downstream_bfs_cached  = downstream_bfs_cached

    # ── 内存缓存（process 生命周期）──
    app.state.expand_cache = {}
    app.state.rcpath_cache = {}
    app.state.detail_cache = {}

    logger.info("[App] Startup complete")
    
    yield

    logger.info("[App] Shutting down...")
    driver.close()
    logger.info("[App] Shutdown complete")


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