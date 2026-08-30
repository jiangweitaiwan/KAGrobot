# -*- coding: utf-8 -*-
"""API route routers."""

from api.routes.chat         import router as chat_router
from api.routes.routes_graph import router as graph_router
from api.routes.concept      import router as concept_router
from api.routes.user_brain   import router as user_brain_router
from api.routes.routes_brain import router as brain_router
from api.routes.users        import router as users_router
from api.routes.routes_walk  import router as walk_router

__all__ = [
    "chat_router",
    "graph_router",
    "concept_router",
    "user_brain_router",
    "brain_router",
    "users_router",
    "walk_router",
]
