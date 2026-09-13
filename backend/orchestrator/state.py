# -*- coding: utf-8 -*-
"""orchestrator/state.py — LangGraph Orchestrator 狀態定義。

【新增 — Random Walk Mode 整合】
walk_bot.py 原本是獨立於 orchestrator 之外的一套狀態機（WalkSession /
WalkScheduler 存在 app.state.walk_sessions 這個 process 內記憶體 dict）。
現在把它併進同一個 OrchestratorState，讓 random walk 跟 dialogue mode
共用同一個 checkpointer、同一個 thread_id 慣例（"{uid}:{session_id}"）、
同一套「main_agent 只認 selected_rc_eid + phase」的交接機制。

新增欄位：
  mode                   : "dialogue"（預設）或 "random_walk"，在對話一
                            開始就鎖定，決定 entry_router 走哪一條分支。
  walk_session           : WalkSession.to_dict() 的輸出（純 dict），
                            不是 WalkSession 實例本身——見 graph_service/
                            walk_state.py::WalkSession.to_dict() 檔頭
                            說明：目前版本 LangGraph 的 checkpointer
                            序列化層（msgpack）不支援自訂 dataclass，
                            未來版本會直接擋下來，因此一律以 dict 形式
                            存放，node 內部要用時才呼叫
                            WalkSession.from_dict() 還原。
  walk_pointer           : WalkScheduler 的 round-robin 指標，見
                            walk_state.py 的 WalkScheduler 變更說明——
                            這是「跨 node 呼叫存續」的狀態，必須明確存在
                            state 裡，不能再依賴 WalkScheduler 實例本身
                            活著。
  walk_pending_question  : walk_decide 節點判斷「這一步該問使用者」時，
                            把 {"path_id": int, "question": str} 放這裡，
                            交給 walk_ask 節點呼叫 interrupt()。刻意拆成
                            兩個節點（walk_decide 負責呼叫 LLM、walk_ask
                            只負責呼叫 interrupt()）：LangGraph 的
                            interrupt() 在 resume 時會把整個 node 從頭
                            重新執行一次，若把「呼叫 LLM」跟「呼叫
                            interrupt()」放在同一個 node，resume 時會連帶
                            重新呼叫一次 LLM（多花一次 API 成本，且理論上
                            可能問出不同的問題）。拆開後，resume 只會重跑
                            walk_ask 這個幾乎無副作用的小節點。
                            resume 完成後這個欄位會被清空（None）。

【新增 — HTML Widget】
  pending_widget : main_agent 節點單輪展示用的 widget 資訊，見
                   core/main_agent.py 檔頭「HTML Widget」段落與
                   orchestrator/nodes.py::OrchestratorNodes.main_agent()。
                   刻意不需要跨輪延續：使用者提交/取消是全新一輪 /chat
                   呼叫，history（main_state）本身已含完整的
                   AIMessage(tool_calls) + ToolMessage，這個欄位純粹是
                   單輪透傳給 api/routes/chat.py 組成
                   ChatResponse.pending_widget，下一輪呼叫時上一輪的值
                   不會、也不需要被讀取。
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class OrchestratorState(TypedDict, total=False):
    messages:          Annotated[List[BaseMessage], add_messages]
    phase:             str
    socrates_state:    Any
    main_state:        Annotated[List[BaseMessage], add_messages]
    handoff_task:      Optional[str]
    selected_rc_eid:   Optional[str]
    selected_rc_label: Optional[str]
    confirmed_rc_eid:  Optional[str]
    skill_sandbox_id:  Optional[str]
    attachments:       List[Dict[str, Any]]
    uid:               Optional[str]
    # 新增：讓 walk_seed 等節點能拿到「這是哪個 chat session」，
    # 純粹用於 WalkSession.session_id 顯示與 debug log，不影響
    # LangGraph thread_id 本身（thread_id 是 config 層級的東西，
    # 與這個欄位無關，見 api/routes/chat.py 的 invoke_input 組裝）。
    session_id:        Optional[str]

    walk_enabled:           bool
    walk_session:           Optional[Dict[str, Any]]
    walk_pointer:            int
    walk_pending_question:  Optional[Dict[str, Any]]
    thinking_process:       str

    # 新增 — HTML Widget（見檔頭說明）
    pending_widget:         Optional[Dict[str, Any]]
