# -*- coding: utf-8 -*-
"""core/main_agent.py — 主分析 Agent（無狀態設計）。

MainAgent 每次 chat() 呼叫：
  1. 動態組裝 system prompt（BASE + 需求 + skill context + 工具清單）
  2. 動態組裝工具清單（skill 存在時掛泛用 run_skill_code，否則 fallback 到 socrates_state）
  3. 建立單次 LangGraph 執行圖並完成推理

Skill 注入：
  chat(..., skill=RCSkill) 時，skill.prompt_context 取代基於 socrates_state
  的 fallback 組裝邏輯；若 skill.has_functions 為真，掛載單一泛用工具
  run_skill_code，讓 LLM 自行撰寫 Python 程式碼在該 skill 專屬的 e2b sandbox
  中 import 並呼叫 skill 內的函式，不再逐一把函式包成獨立 StructuredTool。

參數解析（resolve_params / retrieve_param）：
  比照 dsge-model skill 的 references/params_manifest.yaml 協定，
  凡是有 skill_dir 的情境，都會額外掛載 resolve_params 工具（後端規則
  引擎版本，見 tools/params_resolver.py）：LLM 傳入已從對話中確認的
  參數，工具依 manifest 規則（user_override > tool_retrieval >
  calibrated_default）解析，回傳 resolved / provenance / unresolved。
  這個工具與 skill 是否真的提供 params_manifest.yaml 無關，永遠掛載，
  沒有 manifest 時呼叫會直接回報「此 Skill 未提供 manifest」，不影響
  其餘流程。經濟數據查詢由 resolve_params 內部透過 dispatch_retrieve_param
  直接呼叫，不透過額外的 LLM tool call。

Sandbox 生命週期：
  - Sandbox 採 lazy 建立：只有在 LLM 第一次真的呼叫沙盒工具時才建立，
    由 SkillSandboxProvisioner.create() 完成（見 core/skill_sandbox.py）。
  - 續命併入第一次工具操作的 connect(timeout=...)；單輪 chat() 內快取
    Sandbox 連線，避免反覆 reconnect。sandbox 明確不可用時才清空 id 並重建。
  - sandbox_id 本身不由 MainAgent 持有狀態，而是每次 chat() 回傳最新值，
    由呼叫端（OrchestratorNodes）寫回 OrchestratorState.skill_sandbox_id 逐 session 保存。

Attachment 收集（新增）：
  比照 sandbox_id 的「mutable dict closure」pattern，chat() 開頭建立
  attachment_state = {"items": []}，傳給 _build_tools() 掛載
  get_sandbox_file 時一併注入。單輪對話中每次成功呼叫 get_sandbox_file
  都會往裡面 append 一筆結構化紀錄（ref/kind/url/filename/created_at）。
  graph 執行完畢後，從 attachment_state["items"] 讀出完整清單，放進
  chat() 的回傳值 "attachments"，交由呼叫端（OrchestratorNodes →
  ChatResponse）往前端透傳。這樣圖片/檔案的顯示不再依賴 LLM 有沒有把
  URL 正確寫進回覆文字，而是結構化資料，前端可以穩定渲染；LLM 只需要
  在文字中插入 [[attachment:{ref}]] 佔位符標記位置（見
  tools/skill_code.py 的 get_sandbox_file 工具說明）。

  【更新】uid 命名空間：chat(..., uid=...) 由呼叫端（OrchestratorNodes，
  最終來自 api/routes/chat.py 已驗證的 Firebase uid）傳入，往下交給
  tools/skill_code.py 的 get_sandbox_file 工具，讓輸出檔案的 GCS 物件
  路徑改為 "{uid}/{sandbox_id}/{filename}"，避免不同使用者的輸出檔案
  只靠 sandbox_id 分隔（sandbox_id 本身不對外洩漏，但仍以 uid 多加一層
  隔離，方便未來要依使用者盤點/清理輸出檔案時有明確前綴可用）。
  uid 為 None 時（理論上不會發生，因為 /chat 已強制登入）以 "anon"
  代替，不讓工具直接壞掉。

References 注入（lazy loading）：
  skill.prompt_context 只含 references 的檔名索引（見 skill_loader.py）。
  若 skill.has_references 為真，會額外掛上 read_skill_reference 工具，
  並在 system prompt 固定加上使用提示，讓 LLM 依需要自行呼叫讀取全文。

【新增 — HTML Widget】
  skill.has_html 為真時，額外掛上 show_interactive_html 工具（見
  tools/skill_widget.py），讓 LLM 可以顯示互動式表單取代純文字追問參數。

  這個工具的呼叫本身是完全同步、正常的一輪 tool round-trip（不留
  懸空的 tool_call），差異只在於：_build_graph 新增了 route_after_tools
  路由——若這輪剛執行完的 tool 裡包含 show_interactive_html，直接讓
  graph 提早結束（END），不像其他工具一樣自動繞回 agent 節點。這是因為
  「已經顯示表單、正在等待使用者操作」不需要（也不應該）讓 LLM 在
  同一輪立刻再說一次話。

  chat() 開頭比照 attachment_state 建立 turn_widget_state = {"item": None}，
  工具呼叫成功時寫入；chat() 結束後從這裡讀出，放進回傳值的
  "pending_widget"，由呼叫端（OrchestratorNodes → ChatResponse）透傳給前端。

  使用者在前端提交/取消後，走的是全新一輪 /chat 呼叫（見
  tools/skill_widget.py 檔頭說明），對 MainAgent 而言與一般使用者輸入
  完全無異——history 已經包含「顯示過 widget」的 AIMessage + ToolMessage，
  這裡不需要任何額外的 resume 邏輯。

所有歷史由呼叫端（OrchestratorNodes）管理，MainAgent 本身不持有狀態。
"""

from __future__ import annotations

import textwrap
from typing import List, Optional, Sequence, TypedDict, TYPE_CHECKING

from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage,
)
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import Annotated

from core.chat_state import ChatState, ConversationPhase
from core.google_llm import create_chat_google_generative_ai
from core.llm_retry  import call_with_retry

from core.skill_sandbox import SkillSandboxProvisioner
from tools.params_resolver import make_resolve_params_tool
from tools.knowledge import (
    make_trace_rc_logic_tool,
    make_read_skill_reference_tool,
)
from core.skill_sandbox import SandboxComputer
from tools.skill_code import (
    make_get_sandbox_file_tool,
    make_run_bash_tool,
    make_run_skill_code_tool,
    make_write_file_tool,
)
from tools.skill_widget import make_show_interactive_html_tool

if TYPE_CHECKING:
    from core.skill_loader import RCSkill


# ═══════════════════════════════════════════════════════════════
# LangGraph Agent State（內部用）
# ═══════════════════════════════════════════════════════════════

class _AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ═══════════════════════════════════════════════════════════════
# 常數
# ═══════════════════════════════════════════════════════════════

_BASE_SYSTEM_PROMPT = textwrap.dedent("""\
    你是一位專業的金融與經濟分析助手。
    你的核心任務是：理解使用者的分析需求，運用可用工具，提供清晰、有深度的分析結果。

    ## 使用原則

    - **主動判斷**：根據需求與可用工具，主動決定呼叫哪些工具、以何種順序組合
    - **結果解讀**：收到工具結果後，以繁體中文清楚解讀，結合理論背景提供深度分析
    - **參數確認**：若執行工具所需的參數不完整，先向使用者詢問，再執行
    - **回答結構**：先給結論，再解釋原因；專業但易懂，避免過度學術術語
    - **不確定性**：若分析結果存在前提假設或限制，主動向使用者說明
    - **附件標記**：若你呼叫 get_sandbox_file 取得圖表或檔案，請在回覆文字中
      用 [[attachment:{ref}]] 佔位符標記該檔案對應的位置，不要自己貼網址。
    - **互動表單提交**：若使用者訊息以 [widget_submission: widget_id=...] 開頭，
      代表這是你先前呼叫 show_interactive_html 顯示的表單，使用者剛提交的結果
      （後面接著一段 JSON 資料）；若帶有 cancelled=true，代表使用者取消了這次
      輸入，此時不要假裝已經拿到資料，應詢問使用者接下來想怎麼做。
""")


# ═══════════════════════════════════════════════════════════════
# MainAgent
# ═══════════════════════════════════════════════════════════════

class MainAgent:
    """無狀態設計的主分析 Agent。

    Parameters
    ----------
    model               : LLM 模型名稱
    api_key             : Google API key（選填）
    sandbox_provisioner : e2b sandbox 生命週期管理器（選填，
                          未提供時掛載 skill 的 run_skill_code 工具會直接失敗，
                          僅適用於完全不需要程式碼執行能力的部署情境）
    skill_outputs_bucket : GCS bucket，供 get_sandbox_file 與
                          show_interactive_html 共用（各自以不同物件路徑
                          前綴區隔："{uid}/{sandbox_id}/..." vs
                          "{uid}/widgets/..."）。
    widget_token_secret / widget_token_ttl_s : 見 core/widget_token.py，
                          未設定 secret 時 show_interactive_html 工具
                          仍會掛載，但實際呼叫會回傳簽發失敗的錯誤訊息
                          （不在啟動階段就 fail-fast，維持與其餘工具一致
                          的「掛載永遠成功、失敗留給執行期」設計）。
    """

    def __init__(
        self,
        *,
        model:                str = "gemini-2.5-flash",
        api_key:              str = "",
        sandbox_provisioner:  Optional[SkillSandboxProvisioner] = None,
        skill_outputs_bucket: str = "",
        widget_token_secret:  str = "",
        widget_token_ttl_s:   int = 3600,
        frontend_origin:      str = "",
    ) -> None:
        self._model_name = model
        self._base_llm   = create_chat_google_generative_ai(
            model=model,
            temperature=0,
            api_key=api_key,
        )
        self._sandbox_provisioner  = sandbox_provisioner
        self._skill_outputs_bucket = skill_outputs_bucket
        self._widget_token_secret  = widget_token_secret
        self._widget_token_ttl_s   = widget_token_ttl_s
        self._frontend_origin      = frontend_origin

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        user_input:     str,
        socrates_state: ChatState,
        history:        List[BaseMessage],
        skill:          Optional["RCSkill"] = None,
        sandbox_id:     Optional[str]       = None,
        uid:            Optional[str]       = None,
    ) -> dict:
        """單次處理對話請求（完全無狀態）。

        Parameters
        ----------
        user_input     : 使用者當次輸入（可能是一般問句，也可能是
                         widget 提交/取消訊息，見模組檔頭說明——兩者
                         對 MainAgent 而言沒有特殊分支，都當作一般
                         HumanMessage 處理）
        socrates_state : 包含 query / tools / parameters 等意圖收斂結果
        history        : 目前為止的對話歷史（由外部傳入）
        skill          : 選取 path 後對應的 RC Skill（可為 None）
        sandbox_id     : 此 session 既有的 e2b sandbox id（可為 None，
                         代表尚未建立，或上一輪重新選路後已捨棄）
        uid            : 已驗證的使用者 id（見 api/routes/chat.py），
                         供 get_sandbox_file / show_interactive_html 工具
                         替輸出檔案 / widget token 加上使用者命名空間；
                         可為 None（理論上不會，因為 /chat 已強制登入，
                         此處保留 None 只是為了向後相容直接呼叫
                         MainAgent 的既有測試/腳本）

        Returns
        -------
        {
            "content":         str,
            "tool_calls_made": List[str],
            "messages":        List[BaseMessage],
            "skill_used":      Optional[str],
            "sandbox_id":      Optional[str],  # 本輪結束後最新的 sandbox id，
                                                # 呼叫端需寫回 OrchestratorState 保存
            "attachments":     List[dict],     # 本輪 get_sandbox_file 產出的
                                                # 結構化附件清單
            "pending_widget":  Optional[dict], # 本輪 show_interactive_html
                                                # 產出的待互動 widget（若有）；
                                                # 呼叫端需透傳給前端組成
                                                # ChatResponse.pending_widget，
                                                # 不需要寫回 OrchestratorState
                                                # （單輪展示用，不用跨輪保存）
        }
        """
        if socrates_state.get("phase") != ConversationPhase.PATH_CONFIRMED.value:
            raise ValueError(
                f"MainAgent 需要 phase='path_confirmed'。"
                f"目前為 {socrates_state.get('phase')}"
            )

        # 可變容器：沙盒工具第一次執行時會寫入／更新 sandbox_id，
        # chat() 結束後從這裡讀出回傳給呼叫端保存。
        # 續命與探活改由 SandboxComputer 在實際操作時 connect(timeout=...) 完成，
        # 不再於 chat() 開頭額外打一輪 keep_alive。
        sandbox_state: dict = {"id": sandbox_id}

        # 可變容器：get_sandbox_file 每次成功呼叫會往這裡 append 一筆
        # 結構化 attachment 紀錄（見 tools/skill_code.py）。
        attachment_state: dict = {"items": []}

        # 可變容器：show_interactive_html 成功呼叫時寫入一筆待互動的
        # widget 紀錄（單輪展示用，見 tools/skill_widget.py）。
        turn_widget_state: dict = {"item": None}

        system_prompt = self._build_system_prompt(socrates_state, skill)
        tools         = self._build_tools(
            socrates_state, skill, sandbox_state, attachment_state,
            turn_widget_state, uid=uid,
        )
        has_tools     = bool(tools)

        llm_bound = self._base_llm.bind_tools(tools) if has_tools else self._base_llm

        graph    = self._build_graph(llm_bound, tools, system_prompt, has_tools)
        messages = list(history) + [HumanMessage(content=user_input)]
        result   = graph.invoke({"messages": messages})

        updated_messages: List[BaseMessage] = list(result["messages"])

        try:
            content = SocratesChatBot_extract_raw_content(updated_messages)
        except ValueError:
            # show_interactive_html 會讓 graph 在 ToolMessage 後直接 END，
            # 若 LLM 這輪只呼叫工具、完全沒附文字說明（常見於純 function
            # calling 回應），就不會有任何非空的 AIMessage 可供萃取。
            # 這種情況給一個保底文字，避免整輪對話因此報錯中斷；
            # 有 pending_widget 時給出更貼切的說明，其餘情況維持原本的
            # 通用保底訊息。
            if turn_widget_state.get("item"):
                content = "請在下方表單輸入參數。"
            else:
                content = "（本輪沒有可顯示的文字回覆。）"

        tool_calls_made: List[str] = []
        for msg in updated_messages[len(history):]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_calls_made.extend(tc["name"] for tc in msg.tool_calls)

        return {
            "content":         content,
            "tool_calls_made": tool_calls_made,
            "messages":        updated_messages,
            "skill_used":      skill.rc_name if skill else None,
            "sandbox_id":      sandbox_state["id"],
            "attachments":     attachment_state["items"],
            "pending_widget":  turn_widget_state["item"],
        }

    def describe(
        self,
        socrates_state: ChatState,
        skill: Optional["RCSkill"] = None,
    ) -> str:
        """回傳 Agent 摘要資訊（供 debug / logging）。"""
        query         = socrates_state.get("query")
        sandbox_state = {"id": None}
        attachment_state: dict = {"items": []}
        turn_widget_state: dict = {"item": None}
        tools         = self._build_tools(
            socrates_state, skill, sandbox_state, attachment_state,
            turn_widget_state, uid=None,
        )
        tool_names    = [t.name for t in tools]
        skill_info    = f"  RC Skill: {skill.rc_name}" if skill else "  RC Skill: （無）"
        return "\n".join([
            "MainAgent 摘要 (Stateless)",
            f"  當前需求: {query}",
            skill_info,
            f"  工具清單: {', '.join(tool_names)}",
        ])

    # ------------------------------------------------------------------
    # LangGraph 建構（每次 chat() 動態建立）
    # ------------------------------------------------------------------

    def _build_graph(
        self,
        llm_bound,
        tools:         List,
        system_prompt: str,
        has_tools:     bool,
    ):
        def call_model(state: _AgentState) -> _AgentState:
            normalized = [
                m for m in state["messages"]
                if not isinstance(m, SystemMessage)
            ]
            response = call_with_retry(
                lambda: llm_bound.invoke(
                    [SystemMessage(content=system_prompt)] + normalized
                )
            )
            return {"messages": [response]}

        def route_after_agent(state: _AgentState) -> str:
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return END

        def route_after_tools(state: _AgentState) -> str:
            """ToolNode 剛執行完這一輪所有 tool_call 對應的 ToolMessage。

            若其中包含 show_interactive_html，代表已經把互動表單顯示給
            使用者、正在等待對方操作，這輪對話直接提早結束（END），
            不像其他工具那樣自動繞回 agent 節點讓 LLM 立刻再說一次話。

            實作方式：從訊息串尾端往回收集「連續的 ToolMessage」——
            ToolNode 每次執行完，會把這一輪 AIMessage.tool_calls 對應的
            所有 ToolMessage 依序 append 在訊息串最後面，中間不會夾雜
            其他型別的訊息，因此遇到第一個非 ToolMessage 就停止即可
            準確框出「這一輪」的 tool 執行結果，不會誤吃到更早之前
            某一輪已經處理完的 ToolMessage。
            """
            this_round: List[ToolMessage] = []
            for m in reversed(state["messages"]):
                if isinstance(m, ToolMessage):
                    this_round.append(m)
                else:
                    break
            if any(m.name == "show_interactive_html" for m in this_round):
                return END
            return "agent"

        builder = StateGraph(_AgentState)
        builder.add_node("agent", call_model)

        if has_tools:
            builder.add_node("tools", ToolNode(tools))
            builder.set_entry_point("agent")
            builder.add_conditional_edges(
                "agent", route_after_agent, {"tools": "tools", END: END}
            )
            builder.add_conditional_edges(
                "tools", route_after_tools, {"agent": "agent", END: END}
            )
        else:
            builder.set_entry_point("agent")
            builder.add_edge("agent", END)

        return builder.compile()

    # ------------------------------------------------------------------
    # System prompt 組裝（私有）
    # ------------------------------------------------------------------

    def _build_system_prompt(
        self,
        socrates_state: ChatState,
        skill:          Optional["RCSkill"],
    ) -> str:
        summary = socrates_state.get("context_summary") or "（未提供）"

        need_section = textwrap.dedent(f"""\
            ══════════════════════════════
            使用者需求
            ══════════════════════════════
            {summary}
        """)

        skill_section = ""
        if skill and skill.prompt_context.strip():
            skill_section = textwrap.dedent(f"""\

            ══════════════════════════════
            分析方法：{skill.rc_name}
            ══════════════════════════════
            {skill.prompt_context.strip()}
            """)

        tool_section = self._build_tool_usage_section(skill)

        return (
            _BASE_SYSTEM_PROMPT
            + "\n"
            + need_section
            + skill_section
            + "\n"
            + tool_section
        )

    def _build_tool_usage_section(self, skill: Optional["RCSkill"]) -> str:
        """組裝『可用工具』區塊，唯一負責解釋通用工具怎麼用、彼此怎麼分工。
        Skill 帶了哪些資源（code_dirs / references / html widgets）已由
        SkillLoader 列在上方「分析方法」區塊，這裡只講存取方式。
        """
        header = "══════════════════════════════\n可用工具\n══════════════════════════════"
        parts: List[str] = [header]

        if skill and skill.has_functions and self._sandbox_provisioner is not None:
            parts.append(self._sandbox_tools_description())
        elif skill and skill.has_functions:
            parts.append(
                "（此 Skill 含程式碼，但目前未設定沙盒環境，無法執行——"
                "僅能依「分析方法」說明自行推理，不要嘗試呼叫沙盒工具。）"
            )
        elif skill and skill.has_references:
            parts.append(self._read_skill_reference_description())
        else:
            parts.append("（此對話目前無額外分析工具。）")

        if skill and skill.has_html:
            parts.append("\n" + self._html_widget_description())

        parts.append(
            "\n你也隨時可呼叫 **trace_rc_logic** 工具查詢 concept 的理論背景。"
        )

        if skill and skill.skill_dir is not None:
            parts.append(self._resolve_params_description())

        return "\n".join(parts)

    def _sandbox_tools_description(self) -> str:
        desc = (
            "此 Skill 已掛載沙盒環境（含上方「附帶的資源」列出的程式碼與參考文件），"
            "以下工具共用同一個沙盒：\n"
            "- **run_skill_code**：執行 Python，適合數值運算、模擬、繪圖，"
            "跨多次呼叫保留變數狀態。資源清單中的程式碼目錄已加入 Python path，"
            "可直接 import 對應模組；references 檔案可用 "
            "`open('references/檔名').read()` 直接讀取。\n"
            "- **run_bash**：執行 shell 指令，適合安裝套件、瀏覽/搜尋檔案。"
            "每次呼叫是獨立 process，不與 run_skill_code 共享變數，"
            "不適合用來做數值運算。\n"
            "- **write_file**：建立/覆寫沙盒內檔案。內容含特殊字元、引號、"
            "或大量非 ASCII 文字時，優先用此工具，不要在 run_bash 裡用 "
            "heredoc（cat > file << EOF）寫檔，容易轉義出錯。"
        )
        if self._skill_outputs_bucket:
            desc += (
                "\n- **get_sandbox_file**：把 run_skill_code 產出的檔案"
                "（例如圖表，先存到 outputs/ 目錄）取出並登記為附件。"
                "呼叫成功後會拿到一個 ref 編號，請在回覆文字中用 "
                "[[attachment:{ref}]] 佔位符標記位置，不要自己貼網址，"
                "前端會依 ref 自動換成縮圖／檔案卡片。"
            )
        return desc

    @staticmethod
    def _read_skill_reference_description() -> str:
        return (
            "此 Skill 提供補充參考文件（見上方「附帶的資源」），"
            "可呼叫 **read_skill_reference** 工具，傳入確切檔名讀取完整內容。"
            "若當前需求不需要更多細節，則不必主動呼叫。"
        )

    @staticmethod
    def _html_widget_description() -> str:
        return (
            "此 Skill 提供互動式表單（見上方「附帶的資源」），可呼叫 "
            "**show_interactive_html** 工具顯示給使用者，讓對方直接輸入參數，"
            "比純文字逐一追問更直覺（例如需要多個數值/滑桿式參數時優先考慮）。"
            "呼叫後這輪對話會結束，不要在同一則回覆裡假裝已經拿到使用者輸入的值；"
            "使用者提交後你會在下一輪收到結果。"
        )

    @staticmethod
    def _resolve_params_description() -> str:
        return (
            "\n若此 Skill 有提供 params_manifest.yaml，"
            "執行模型前應先呼叫 **resolve_params** 工具，"
            "傳入你已從對話中確認的參數，取得 resolved／provenance／unresolved。"
            "若 unresolved 非空，先跟使用者確認清單裡的問題，"
            "答案齊全後再重新呼叫一次確認 unresolved 為空，"
            "才可以呼叫 run_skill_code 執行模型。"
            "回覆時若 provenance 顯示 tool_retrieval 失敗改用 "
            "calibrated_default（見 _fallback_log），"
            "或某個數值超出 valid_range 而附上 warning，"
            "務必向使用者說明數據來源、時效性與可靠性風險，"
            "不要讓使用者誤以為是即時或無風險的數據。"
            "若此 Skill 未提供 manifest，呼叫 resolve_params 會直接說明，"
            "不影響你照舊自行判斷參數並執行 run_skill_code。"
        )





    # ------------------------------------------------------------------
    # Tool 清單組裝（私有）
    # ------------------------------------------------------------------

    def _build_tools(
        self,
        socrates_state:     ChatState,
        skill:              Optional["RCSkill"],
        sandbox_state:      dict,
        attachment_state:   dict,
        turn_widget_state:  dict,
        uid:                Optional[str] = None,
    ) -> List[StructuredTool]:
        langchain_tools: List[StructuredTool] = []   # ← 修正：搬到最上面，兩個分支都能安全 append

        if skill and skill.has_functions and skill.skill_dir is not None:
            if self._sandbox_provisioner is not None:
                remote_root = f"/home/user/skills/{self._skill_slug(skill.rc_name)}"
                computer = SandboxComputer(
                    provisioner=self._sandbox_provisioner,
                    sandbox_state=sandbox_state,
                    skill_dir=skill.skill_dir,
                    remote_root=remote_root,
                    code_dirs=skill.code_dirs,
                )
                langchain_tools.append(
                    make_run_skill_code_tool(computer, skill.rc_name, skill.code_dirs)
                )
                langchain_tools.append(make_run_bash_tool(computer, skill.rc_name))
                langchain_tools.append(make_write_file_tool(computer, skill.rc_name))

                if self._skill_outputs_bucket:
                    langchain_tools.append(
                        make_get_sandbox_file_tool(
                            computer,
                            skill.rc_name,
                            self._skill_outputs_bucket,
                            attachment_state,
                            uid=uid,
                        )
                    )
                else:
                    print(f"[MainAgent] skill_outputs_bucket 未設定，"
                          f"跳過 get_sandbox_file 工具（skill='{skill.rc_name}'）。")
                # references 已隨 skill_dir 一起上傳進沙盒，run_bash/run_skill_code
                # 內即可直接 cat/open 讀取，不需要額外的 host-side 工具。
            else:
                print(f"[MainAgent] skill '{skill.rc_name}' has_functions=True 但未設定 "
                      f"sandbox_provisioner，跳過沙盒工具掛載（僅適用於完全不需要"
                      f"程式碼執行能力的部署情境）。")
        else:
            # 沒有 script → 沒有 sandbox，references 改由 host-side 工具直接讀本機檔案
            if skill and skill.has_references and skill.skill_dir is not None:
                allowed_files = [r.name for r in skill.reference_paths]
                langchain_tools.append(
                    make_read_skill_reference_tool(skill.skill_dir, allowed_files)
                )

        # show_interactive_html：獨立於 has_functions/sandbox 之外，
        # 只要 skill 有宣告 html widgets 就掛載（不需要 e2b sandbox，
        # 即時試算走獨立的 subprocess，見 core/skill_compute.py）。
        if skill and skill.has_html and skill.skill_dir is not None:
            if self._skill_outputs_bucket:
                langchain_tools.append(
                    make_show_interactive_html_tool(
                        skill_dir=skill.skill_dir,
                        skill_name=skill.rc_name,
                        widgets=skill.html_widgets,
                        bucket_name=self._skill_outputs_bucket,
                        uid=uid or "anon",
                        widget_token_secret=self._widget_token_secret,
                        widget_token_ttl_s=self._widget_token_ttl_s,
                        turn_widget_state=turn_widget_state,
                        frontend_origin=self._frontend_origin,
                    )
                )
            else:
                print(f"[MainAgent] skill_outputs_bucket 未設定，"
                      f"跳過 show_interactive_html 工具（skill='{skill.rc_name}'）。")

        # resolve_params：只要有 skill_dir 就掛載
        if skill and skill.skill_dir is not None:
            langchain_tools.append(make_resolve_params_tool(skill.skill_dir))


        langchain_tools.append(make_trace_rc_logic_tool())
        return langchain_tools








    @staticmethod
    def _skill_slug(rc_name: str) -> str:
        import re
        s = rc_name.lower()
        s = re.sub(r"[^a-z0-9\s_]", "", s)
        s = re.sub(r"\s+", "_", s.strip())
        return re.sub(r"_+", "_", s) or "skill"




# ═══════════════════════════════════════════════════════════════
# Module-level 相容函數
# （供 orchestrator/nodes.py 呼叫 _extract_raw_content 使用）
# ═══════════════════════════════════════════════════════════════

def SocratesChatBot_extract_raw_content(msgs: List[BaseMessage]) -> str:
    """從 agent_messages 中提取最後一則 AI 回覆的原始文字。

    命名刻意不以下底線開頭，表示這是 module-level 公開函數，
    而非 SocratesChatBot 的私有方法。
    實作邏輯移至此處，避免 main_agent 直接呼叫 socrates_bot 的私有方法。
    """
    for msg in reversed(msgs):
        if not (isinstance(msg, AIMessage) and msg.content):
            continue
        if isinstance(msg.content, list):
            if not msg.content:
                raise ValueError("AIMessage content list is empty.")
            item = msg.content[0]
            raw  = item.get("text", "") if isinstance(item, dict) else str(item)
        else:
            raw = str(msg.content)
        raw = raw.strip()
        if not raw:
            raise ValueError("AIMessage content is blank.")
        return raw
    raise ValueError("No AIMessage with content found in messages.")
