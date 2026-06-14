# -*- coding: utf-8 -*-
"""API route routers."""

from api.routes.chat import router as chat_router
from api.routes.routes_graph import router as graph_router

__all__ = ["chat_router", "graph_router"]
