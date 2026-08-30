# -*- coding: utf-8 -*-
"""app.py — FastAPI application factory（新增 random walk mode 的組裝）

與前一版的差異：
  1. 新增 StartConceptPool、make_frontier_fetcher、WalkTurnRunner 的組裝
     （見 graph_service/walk_start_pool.py、graph_service/walk_frontier.py、
     core/walk_bot.py）。
  2. app.state 新增 start_concept_pool / walk_runner / walk_sessions
     三個欄位，供 api/routes/routes_walk.py 使用。
  3. 掛載 walk_router。
  4. walk_runner 的 LLM 呼叫沿用既有 core/google_llm.py +
     core/llm_retry.py 的組合方式，與 socrates_bot / main_agent 一致，
     只是 bind_tools 的對象換成 walk_bot.py 動態產生的 advance_path schema。
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import SystemMessage
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
from graph_service.walk_start_pool import StartConceptPool
from graph_service.walk_frontier    import make_frontier_fetcher
from core.socrates_bot             import SocratesChatBot
from core.main_agent               import MainAgent
from core.skill_loader             import SkillLoader
from core.google_llm               import create_chat_google_generative_ai
from core.llm_retry                import call_with_retry
from core.walk_bot                 import WalkTurnRunner
from orchestrator                  import OrchestratorNodes, build_orchestrator
from api.routes                    import (
    chat_router,
    graph_router,
    concept_router,
    user_brain_router,
    brain_router,
    users_router,
    walk_router,
)

from core.skill_sandbox import SkillSandboxProvisioner
from retrieval.rc_category_loader import RCCategoryLoader

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

    # ── 鄰接層 cache + BFS Coordinator ────────────────────────
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
        skill_outputs_bucket=s.skill_outputs_bucket_name,
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
        coordinator          = coordinator,
    )
    orchestrator = build_orchestrator(nodes)
    foundation_recommender = FoundationRecommender(
        coordinator=coordinator,
        encoder=encoder,
    )
    rc_category_loader = RCCategoryLoader(bucket_name=s.gcs_bucketforresult)

    # ── Random Walk Mode 組裝（新增）───────────────────────────
    #
    # StartConceptPool：起點檢索池（見 graph_service/walk_start_pool.py），
    # 與 CandidateRetriever 各自獨立索引，職責不同（leaf foundation
    # concept vs. RC），不共用同一份 embedding index。
    start_concept_pool = StartConceptPool(driver, encoder)

    # frontier_fetcher：接 AdjacencyCache 取一跳下游鄰居 + sbert prefilter
    # （見 graph_service/walk_frontier.py）。與 dialogue mode 共用同一個
    # adjacency_cache 實例，同一個節點不論在哪個 mode 展開過都能互相命中。
    walk_frontier_fetcher = make_frontier_fetcher(adjacency_cache, driver, encoder)

    # walk LLM：獨立一個 LLM 實例（而非重用 socrates_bot 內部的），
    # 因為 walk_bot.py 每輪要動態 bind_tools 不同的 advance_path schema
    # （target_eid 的 Literal 隨當輪 frontier 變動），與 socrates_bot
    # 固定 bind_tools 一次的用法不同，分開持有比較乾淨。
    walk_llm = create_chat_google_generative_ai(
        model=s.socrates_model,
        temperature=0.2,
    )

    def _walk_llm_caller(system_prompt: str, tool_schema):
        return call_with_retry(
            lambda: walk_llm.bind_tools([tool_schema]).invoke(
                [SystemMessage(content=system_prompt)]
            )
        )

    walk_runner = WalkTurnRunner(
        frontier_fetcher=walk_frontier_fetcher,
        llm_caller=_walk_llm_caller,
    )

    # ── app.state ─────────────────────────────────────────────
    app.state.orchestrator    = orchestrator
    app.state.layout_builder  = layout_builder
    app.state.layout_cache    = None
    app.state.settings        = s
    app.state.driver          = driver
    app.state.encoder         = encoder
    app.state.coordinator     = coordinator
    app.state.candidate_retriever   = candidate_retriever
    app.state.foundation_recommender = foundation_recommender
    app.state.skill_loader          = skill_loader
    app.state.detail_cache = {}
    app.state.bertopic_layout = None

    app.state.rc_category_loader = rc_category_loader
    app.state.rc_category_map    = None 

    # Random Walk Mode 的新增狀態：
    #   walk_sessions 是 process 內記憶體 dict（key: "{uid}:{walk_id}"），
    #   重啟即遺失——見 api/routes/routes_walk.py 檔頭關於這個取捨的說明。
    app.state.start_concept_pool = start_concept_pool
    app.state.walk_runner        = walk_runner
    app.state.walk_sessions      = {}


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
    app.include_router(users_router)
    app.include_router(walk_router)

    return app