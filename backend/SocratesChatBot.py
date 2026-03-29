"""
Compatibility wrapper.

`entry.py` imports `SocratesChatBot` / `ChatState` / `ConversationPhase` /
`Neo4jReasoningRetriever` from this module, but the actual implementation
currently lives in `mainagent可部署.py`.
"""

from mainagent可部署 import (  # type: ignore
    SocratesChatBot,
    ChatState,
    ConversationPhase,
    Neo4jReasoningRetriever,
)

__all__ = [
    "SocratesChatBot",
    "ChatState",
    "ConversationPhase",
    "Neo4jReasoningRetriever",
]

