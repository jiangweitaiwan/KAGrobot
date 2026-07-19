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
  其餘流程。
  retrieve_param 工具（單一經濟數據序列查詢，FRED/WorldBank/OECD/BEA）
  則無條件掛載，供 LLM 在一般對話中查詢時效性數據，不限於任何 Skill
  的 manifest。

Sandbox 生命週期：
  - Sandbox 採 lazy 建立：只有在 LLM 第一次真的呼叫 run_skill_code 時才建立，
    由 SkillSandboxProvisioner.create() 完成（見 core/skill_sandbox.py）。
  - 每次 chat() 呼叫開頭，若傳入既有 sandbox_id，會先嘗試 keep_alive() 續命；
    續命失敗（sandbox 已過期）視同不存在，交由下一次 run_skill_code 呼叫時重建。
  - sandbox_id 本身不由 MainAgent 持有狀態，而是每次 chat() 回傳最新值，
    由呼叫端（OrchestratorNodes）寫回 OrchestratorState.skill_sandbox_id 逐 session 保存。

References 注入（lazy loading）：
  skill.prompt_context 只含 references 的檔名索引（見 skill_loader.py）。
  若 skill.has_references 為真，會額外掛上 read_skill_reference 工具，
  並在 system prompt 固定加上使用提示，讓 LLM 依需要自行呼叫讀取全文。

所有歷史由呼叫端（OrchestratorNodes）管理，MainAgent 本身不持有狀態。
"""

from __future__ import annotations

import textwrap
from typing import List, Optional, Sequence, TypedDict, TYPE_CHECKING

from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage,
)
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import Annotated

from core.chat_state import ChatState, ConversationPhase
from core.google_llm import create_chat_google_generative_ai
from core.llm_retry  import call_with_retry
from core.registry   import AGENTS_REGISTRY, TOOLS_REGISTRY, collect_tools_with_deps
from core.skill_sandbox import SkillSandboxProvisioner
from tools.base      import _TOOL_IMPLEMENTATIONS
from tools.data_retriever import retrieve_param
from tools.params_resolver import make_resolve_params_tool
from tools.knowledge import (
    make_knowledge_qa_tool,
    make_trace_rc_logic_tool,
    make_read_skill_reference_tool,
)
from tools.skill_code import make_run_skill_code_tool

import tools  # noqa: F401  確保所有工具副作用式註冊已執行

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
    """

    def __init__(
        self,
        *,
        model:               str = "gemini-2.5-flash",
        api_key:             str = "",
        sandbox_provisioner: Optional[SkillSandboxProvisioner] = None,
    ) -> None:
        self._model_name = model
        self._base_llm   = create_chat_google_generative_ai(
            model=model,
            temperature=0,
            api_key=api_key,
        )
        self._sandbox_provisioner = sandbox_provisioner

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
    ) -> dict:
        """單次處理對話請求（完全無狀態）。

        Parameters
        ----------
        user_input     : 使用者當次輸入
        socrates_state : 包含 query / tools / parameters 等意圖收斂結果
        history        : 目前為止的對話歷史（由外部傳入）
        skill          : 選取 path 後對應的 RC Skill（可為 None）
        sandbox_id     : 此 session 既有的 e2b sandbox id（可為 None，
                         代表尚未建立，或上一輪重新選路後已捨棄）

        Returns
        -------
        {
            "content":         str,
            "tool_calls_made": List[str],
            "messages":        List[BaseMessage],
            "skill_used":      Optional[str],
            "sandbox_id":      Optional[str],  # 本輪結束後最新的 sandbox id，
                                                # 呼叫端需寫回 OrchestratorState 保存
        }
        """
        if socrates_state.get("phase") != ConversationPhase.PATH_CONFIRMED.value:
            raise ValueError(
                f"MainAgent 需要 phase='path_confirmed'。"
                f"目前為 {socrates_state.get('phase')}"
            )

        # 每次呼叫都幫既有 sandbox 續命；續命失敗（sandbox 已過期）
        # 視同不存在，交由 run_skill_code 第一次被呼叫時 lazy 重建。
        if sandbox_id and self._sandbox_provisioner:
            if not self._sandbox_provisioner.keep_alive(sandbox_id):
                print(f"[MainAgent] sandbox '{sandbox_id}' keep_alive failed, "
                      f"will lazily recreate on next run_skill_code call.")
                sandbox_id = None

        # 可變容器：run_skill_code 內部第一次執行時會寫入新建立的 sandbox_id，
        # chat() 結束後從這裡讀出回傳給呼叫端保存。
        sandbox_state: dict = {"id": sandbox_id}

        system_prompt = self._build_system_prompt(socrates_state, skill)
        tools         = self._build_tools(socrates_state, skill, sandbox_state)
        has_tools     = bool(tools)

        llm_bound = self._base_llm.bind_tools(tools) if has_tools else self._base_llm

        graph    = self._build_graph(llm_bound, tools, system_prompt, has_tools)
        messages = list(history) + [HumanMessage(content=user_input)]
        result   = graph.invoke({"messages": messages})

        updated_messages: List[BaseMessage] = list(result["messages"])
        content          = SocratesChatBot_extract_raw_content(updated_messages)

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
        }

    def describe(
        self,
        socrates_state: ChatState,
        skill: Optional["RCSkill"] = None,
    ) -> str:
        """回傳 Agent 摘要資訊（供 debug / logging）。"""
        query         = socrates_state.get("query")
        sandbox_state = {"id": None}
        tools         = self._build_tools(socrates_state, skill, sandbox_state)
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

        def should_continue(state: _AgentState) -> str:
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return END

        builder = StateGraph(_AgentState)
        builder.add_node("agent", call_model)

        if has_tools:
            builder.add_node("tools", ToolNode(tools))
            builder.set_entry_point("agent")
            builder.add_conditional_edges(
                "agent", should_continue, {"tools": "tools", END: END}
            )
            builder.add_edge("tools", "agent")
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
        query = socrates_state.get("query") or "（未提供）"

        need_section = textwrap.dedent(f"""\
            ══════════════════════════════
            使用者需求
            ══════════════════════════════
            {query}
        """)

        skill_section = ""
        if skill and skill.prompt_context.strip():
            skill_section = textwrap.dedent(f"""\

            ══════════════════════════════
            分析方法：{skill.rc_name}
            ══════════════════════════════
            {skill.prompt_context.strip()}
            """)

        if skill and skill.has_functions:
            tool_lines  = self._tool_lines_from_skill(skill)
            tool_header = "此 Skill 提供的分析函式（透過 run_skill_code 呼叫）"
        else:
            tool_lines  = self._fallback_tool_lines(socrates_state)
            tool_header = "可用工具"

        tool_section = (
            f"══════════════════════════════\n"
            f"{tool_header}\n"
            f"══════════════════════════════\n"
            + ("\n".join(tool_lines) if tool_lines else "（無額外工具）")
        )

        trace_hint = (
            "\n此外，你始終可使用 **trace_rc_logic** 工具查詢 concept 中的理論背景。"
        )
        if skill and skill.has_functions:
            trace_hint += (
                "\n若需要執行上方列出的函式、進行數值運算、模擬或繪圖，"
                "請呼叫 **run_skill_code** 工具，直接撰寫可執行的 Python 程式碼"
                "（可直接 import 上方函式，不需自行處理路徑）。"
            )

        reference_hint = self._build_reference_hint(skill)
        resolve_hint    = self._build_resolve_params_hint(skill)

        return (
            _BASE_SYSTEM_PROMPT
            + "\n"
            + need_section
            + skill_section
            + "\n"
            + tool_section
            + trace_hint
            + reference_hint
            + resolve_hint
        )

    @staticmethod
    def _build_reference_hint(skill: Optional["RCSkill"]) -> str:
        if not skill or not skill.has_references:
            return ""

        file_list = "、".join(f"`{r.name}`" for r in skill.reference_paths)

        if skill.has_functions:
            # references 已隨 skill_dir 一起上傳進沙盒，不需要額外的 host-side 工具
            return (
                f"\n若上方「分析方法」說明中提及需要查閱的補充文件"
                f"（目前此 Skill 提供：{file_list}），"
                f"這些檔案已放在沙盒內的 references/ 目錄，"
                f"可在 **run_skill_code** 中直接讀取，"
                f"例如 `print(open('references/檔名').read())`。"
                f"若當前需求不需要更多細節，則不必主動讀取。"
            )

        return (
            f"\n若上方「分析方法」說明中提及需要查閱的補充文件"
            f"（目前此 Skill 提供：{file_list}），"
            f"可呼叫 **read_skill_reference** 工具，傳入確切檔名讀取完整內容。"
            f"若當前需求不需要更多細節，則不必主動呼叫。"
        )

    @staticmethod
    def _build_resolve_params_hint(skill: Optional["RCSkill"]) -> str:
        """resolve_params / retrieve_param 的使用規則說明。

        resolve_params 只在有 skill_dir 時掛載（見 _build_tools），
        故此提示也只在該情境下加入；retrieve_param 無條件掛載，
        提示文字放在最後一句、不受 skill 是否存在影響。
        """
        resolve_part = ""
        if skill and skill.skill_dir is not None:
            resolve_part = (
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

        retrieve_part = (
            "\n你也可以隨時呼叫 **retrieve_param** 查詢單一經濟數據序列"
            "（FRED / World Bank / OECD / BEA，不限於此 Skill 的 manifest），"
            "例如使用者臨時想知道其他國家或指標的現況。"
        )

        return resolve_part + retrieve_part

    @staticmethod
    def _tool_lines_from_skill(skill: "RCSkill") -> List[str]:
        if skill.function_specs:
            return [
                f"{i}. **{spec.function}**（{spec.script}） — {spec.description}"
                for i, spec in enumerate(skill.function_specs, 1)
            ]
        # frontmatter 未宣告個別函式時，實際函式清單已在
        # skill.prompt_context（SKILL.md body + sandbox 使用提示）中說明，
        # 這裡只需提示程式碼實際位置即可。
        dirs = [d for d in skill.code_dirs if d != "."]
        dir_hint = "、".join(f"`{d}/`" for d in dirs) if dirs else "skill 根目錄"
        return [
            f"（此 Skill 未在 frontmatter 逐一宣告函式，"
            f"完整程式碼已掛載於 {dir_hint}，"
            f"請參考上方「分析方法」說明中的使用範例與檔案結構。）"
        ]





    # ------------------------------------------------------------------
    # Tool 清單組裝（私有）
    # ------------------------------------------------------------------

    def _build_tools(
        self,
        socrates_state: ChatState,
        skill:          Optional["RCSkill"],
        sandbox_state:  dict,
    ) -> List[StructuredTool]:
        if skill and skill.has_functions and skill.skill_dir is not None:
            remote_root = f"/home/user/skills/{self._skill_slug(skill.rc_name)}"
            langchain_tools: List[StructuredTool] = [
                make_run_skill_code_tool(
                    provisioner=self._sandbox_provisioner,
                    sandbox_state=sandbox_state,
                    skill_dir=skill.skill_dir,
                    remote_root=remote_root,
                    code_dirs=skill.code_dirs,   # ← 新增：告知 sandbox 要掛哪些目錄到 sys.path
                )
            ]            
        else:

            # 沒有 script → 沒有 sandbox，references 改由 host-side 工具直接讀本機檔案
            if skill and skill.has_references and skill.skill_dir is not None:
                allowed_files = [r.name for r in skill.reference_paths]
                langchain_tools.append(
                    make_read_skill_reference_tool(skill.skill_dir, allowed_files)
                )

        # resolve_params：只要有 skill_dir 就掛載（不判斷 manifest 存不存在，
        # 有沒有 manifest 由工具被呼叫「當下」自己去讀檔案決定，見
        # tools/params_resolver.py 的說明）。
        if skill and skill.skill_dir is not None:
            langchain_tools.append(make_resolve_params_tool(skill.skill_dir))

        # retrieve_param：與 skill/manifest 完全無關的一般查詢工具，
        # 無條件掛載（跟 trace_rc_logic 同一層級）。
        langchain_tools.append(retrieve_param)

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