"""
Agent 运行时：把 AgentWorker.run() 拆成三块
=============================================
MessageBuilder  —— system + history → 干净的 messages（tool_calls 规范化、内容截断）
ToolExecutor    —— QThreadPool + QRunnable 跑工具，支持超时（默认 30s），超时返回结构化错误
StopCondition   —— max_steps / 最后一轮是否还有 tool_calls / 连续无进展，三重判断
AgentLoop       —— 把上面三块串起来；触顶时给 LLM 发一条"用尽预算"的 user 消息收尾

所有交给 LLM 的 dict 都先过 jsonable()，杜绝 numpy / Path 打爆 json.dumps。
本模块不强依赖 Qt：没有 PySide6 时自动退回 threading。
"""

import re
import json
import time
import threading
import traceback
from dataclasses import dataclass, field
from typing import Optional, Callable

from agent.util import jsonable, dumps_safe

MAX_TOOL_CHARS = 2400      # 单条 tool 结果喂回 LLM 的字符上限
MAX_HISTORY_TURNS = 24     # 只回传最近 N 条历史，防止上下文膨胀
MAX_PLAN_STRAY = 2         # 连续 N 步没推进计划 → 强制把计划下一步塞给模型

# ==================== token 预算 ====================
# 粗估而不是真 tokenizer：tiktoken 对 DeepSeek / Qwen 会算错，不如用"宁可估多"的启发式。
# 观测日志 context_utilization.util 长期偏低 = 估保守了，把下面两个分母调大即可。
CHARS_PER_TOKEN_ASCII = 4        # 英文 / 代码约 4 字符 1 token
CHARS_PER_TOKEN_NON_ASCII = 1.2  # 中文偏保守：1 字按 0.83 token（实际区间 0.6~1.5）
MSG_OVERHEAD_TOKENS = 4          # role + 消息包装结构的固定开销

# history 之外还要留给：工具 schema（~1.5K）+ 长期记忆（~0.5K）+ 当前轮 tool 结果（~3K × N 步）
RESERVE_FOR_SCHEMA_AND_MEMORY = 2048

# 不强依赖 openai 包：拿不到常量就用同值兜底，保证本模块可单独导入
try:
    from agent.llm_client import DEFAULT_OUTPUT_RESERVE, DEFAULT_SAFETY_MARGIN
except Exception:  # pragma: no cover
    DEFAULT_OUTPUT_RESERVE = 4096
    DEFAULT_SAFETY_MARGIN = 1024


def estimate_tokens(text: str) -> int:
    """粗估 token 数：ASCII 按 4 字符/token，非 ASCII 按 1.2 字符/token。"""
    if not text:
        return 0
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    non_ascii = len(text) - ascii_chars
    return (ascii_chars // CHARS_PER_TOKEN_ASCII) + int(non_ascii / CHARS_PER_TOKEN_NON_ASCII)


def msg_tokens(msg: dict) -> int:
    """一条 message 的 token 估算：content + tool_calls(name/arguments) + 角色开销"""
    if not isinstance(msg, dict):
        return 0
    n = estimate_tokens(msg.get("content") or "")
    for tc in msg.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        func = tc.get("function") or {}
        if not isinstance(func, dict):
            continue
        n += estimate_tokens(func.get("name", ""))
        n += estimate_tokens(func.get("arguments", ""))
    return n + MSG_OVERHEAD_TOKENS


def compute_history_budget(system_prompt: str, context_window: int,
                           output_reserve: int = DEFAULT_OUTPUT_RESERVE) -> int:
    """history 能分到多少 token。

    context_window 为 0（未知模型）→ 返回 0，即不按 token 截断，退回 turn 截断。
    短 context 模型（8K 的 qwen）算出来会是 0 —— 这是刻意行为：只喂 system + 当前轮，
    让模型单轮收工，而不是硬塞一段必然会被截断的 history。
    """
    if context_window <= 0:
        return 0
    total = context_window - output_reserve - DEFAULT_SAFETY_MARGIN
    total -= estimate_tokens(system_prompt or "")
    total -= RESERVE_FOR_SCHEMA_AND_MEMORY
    return max(0, total)


# ==================== 消息构造 ====================

class MessageBuilder:
    """负责把 system + 会话历史拼成 OpenAI 格式的 messages

    两阶段构建：先 sanitize 规范化，再按 token 预算从尾向头塞。
    max_history_tokens=0 表示不按 token 截断（退化成原来的 turn 截断）。
    """

    def __init__(self, system_prompt: str, max_history_tokens: int = 0,
                 context_window: int = 0,
                 output_reserve: int = DEFAULT_OUTPUT_RESERVE,
                 max_history_turns: int = MAX_HISTORY_TURNS):
        self.system_prompt = system_prompt or ""
        self.max_history_turns = max_history_turns
        self.max_history_tokens = max(0, int(max_history_tokens or 0))
        self.context_window = int(context_window or 0)
        self.output_reserve = int(output_reserve or DEFAULT_OUTPUT_RESERVE)
        self.last_stats: dict = {}

    def build(self, history: list, extra_system: str = "",
              budget_note: str = "") -> list:
        system = self.system_prompt
        if extra_system:
            system = f"{system}\n\n{extra_system}"
        system_msg = {"role": "system", "content": system}

        clean = [self.sanitize(m) for m in (history or [])]
        clean = [m for m in clean if m]

        stats = {
            "system_tokens": msg_tokens(system_msg),
            "history_before": len(clean),
            "tokens_before": sum(msg_tokens(m) for m in clean),
            "budget": self.max_history_tokens,
        }

        # 先按 turns 粗截一轮，避免超长 history 让后面逐条估算白跑
        if self.max_history_turns > 0 and len(clean) > self.max_history_turns:
            clean = clean[-self.max_history_turns:]

        if self.max_history_tokens > 0:
            clean = self._trim_to_budget(clean, self.max_history_tokens)

        stats["history_after"] = len(clean)
        stats["tokens_after"] = sum(msg_tokens(m) for m in clean)
        stats["dropped"] = stats["history_before"] - len(clean)
        self.last_stats = stats

        messages = [system_msg] + clean
        if budget_note:
            messages.append({"role": "user", "content": budget_note})
        return messages

    # ---------------- 预算截断 ----------------

    def _trim_to_budget(self, messages: list, budget: int) -> list:
        """从尾向头凑，塞满 budget 为止。

        保留下来的一定是**连续的**最新后缀：某条塞不下就停，不再往前捞更旧的。
        否则旧消息会被排到新消息后面，时间顺序错乱比少喂几条历史更伤。
        """
        kept: list = []
        used = 0
        for msg in reversed(messages):
            cost = msg_tokens(msg)
            if used + cost <= budget:
                kept.append(msg)
                used += cost
                continue
            # 塞不下：tool 结果可以压缩保留（头尾各留一段），比整条丢掉安全
            if msg.get("role") == "tool":
                shrunk = self._shrink_tool(msg, budget - used)
                if shrunk is not None:
                    kept.append(shrunk)
                    used += msg_tokens(shrunk)
                    continue
            break
        kept.reverse()
        return self._repair_tool_pairs(kept)

    @staticmethod
    def _shrink_tool(msg: dict, room_tokens: int) -> Optional[dict]:
        """把超长 tool 结果压成"头 + 尾"。

        保留首尾比只留开头有用：工具结果通常开头是状态、结尾是结论/路径。
        """
        content = msg.get("content") or ""
        if len(content) <= 240 or room_tokens <= 0:
            return None
        note = f"\n...(已截断，原文 {len(content)} 字，保留首尾)"
        # token 预算折算成字符数时按中文系数算，宁可少留也不要超
        avail = int(room_tokens * CHARS_PER_TOKEN_NON_ASCII) - len(note)
        if avail < 160:
            return None
        head = avail // 2
        tail = avail - head
        out = dict(msg)
        out["content"] = content[:head] + note + content[-tail:]
        return out

    @staticmethod
    def _repair_tool_pairs(messages: list) -> list:
        """tool result 与 assistant(tool_call) 必须成对。

        截断后任一侧落单都会让 LLM 报 tool_call_id mismatch，所以分两步收敛：
          1. 丢掉没有对应 assistant 的孤儿 tool result；
          2. 丢掉返回结果不齐全的 assistant（连它的 tool result 一起落单）；
          3. 第 2 步会产生新的孤儿，再清一遍。
        """
        def _call_ids(msgs: list) -> set:
            ids = set()
            for m in msgs:
                if m.get("role") == "assistant":
                    for tc in m.get("tool_calls") or []:
                        tid = (tc or {}).get("id")
                        if tid:
                            ids.add(tid)
            return ids

        def _drop_orphan_tools(msgs: list) -> list:
            wanted = _call_ids(msgs)
            return [m for m in msgs
                    if not (m.get("role") == "tool"
                            and m.get("tool_call_id") not in wanted)]

        kept = _drop_orphan_tools(messages)
        present = {m.get("tool_call_id") for m in kept if m.get("role") == "tool"}
        kept = [
            m for m in kept
            if not (
                m.get("role") == "assistant" and m.get("tool_calls")
                and any((tc or {}).get("id") not in present
                        for tc in m["tool_calls"] if (tc or {}).get("id"))
            )
        ]
        return _drop_orphan_tools(kept)

    @staticmethod
    def sanitize(msg: dict) -> Optional[dict]:
        """只保留 LLM 认识的字段，去掉 None / numpy 引用"""
        if not isinstance(msg, dict):
            return None
        role = msg.get("role")
        if role not in ("user", "assistant", "tool", "system"):
            return None
        content = msg.get("content") or ""
        if not isinstance(content, str):
            content = dumps_safe(content)
        out = {"role": role, "content": content}

        if role == "assistant":
            tool_calls = MessageBuilder.normalize_tool_calls(msg.get("tool_calls"))
            if tool_calls:
                out["tool_calls"] = tool_calls
            if not out["content"] and not tool_calls:
                return None
        if role == "tool":
            call_id = msg.get("tool_call_id") or msg.get("tool_call") or ""
            if not call_id:
                return None
            out["tool_call_id"] = str(call_id)
            if len(out["content"]) > MAX_TOOL_CHARS:
                out["content"] = out["content"][:MAX_TOOL_CHARS] + "\n...(结果已截断)"
        if role == "user" and not out["content"]:
            return None
        return out

    @staticmethod
    def normalize_tool_calls(tool_calls) -> list:
        """pydantic 对象 / dict 混着来也能处理；arguments 统一为 JSON 字符串"""
        if not tool_calls:
            return []
        out = []
        for tc in tool_calls:
            if hasattr(tc, "model_dump"):
                try:
                    tc = tc.model_dump()
                except Exception:
                    continue
            if not isinstance(tc, dict):
                continue
            func = tc.get("function", {}) or {}
            if hasattr(func, "model_dump"):
                func = func.model_dump()
            name = func.get("name", "")
            args = func.get("arguments", "")
            if not isinstance(args, str):
                args = dumps_safe(jsonable(args))
            out.append({
                "id": str(tc.get("id", f"call_{len(out)}")),
                "type": "function",
                "function": {"name": name, "arguments": args},
            })
        return out

    @staticmethod
    def assistant_entry(content: str, tool_calls: list) -> dict:
        """把 LLM 返回的 assistant 消息写回 messages（工具调用用原始 tool_calls 结构）"""
        entry = {"role": "assistant", "content": content or ""}
        if tool_calls:
            entry["tool_calls"] = tool_calls
        return entry

    @staticmethod
    def tool_entry(call_id: str, result: dict) -> dict:
        text = dumps_safe(result)
        if len(text) > MAX_TOOL_CHARS:
            result = dict(result or {})
            result["_note"] = "结果过长已截断，完整内容请查看 UI 或报告文件"
            text = dumps_safe(result)[:MAX_TOOL_CHARS] + "\n...(结果已截断)"
        return {"role": "tool", "tool_call_id": str(call_id), "content": text}


# ==================== 带超时的工具执行 ====================

class _Runnable:
    """统一的执行体：Qt 环境用 QRunnable，否则用 threading.Thread"""

    def __init__(self, func):
        self.func = func
        self.result = None
        self.error = None
        self.done = threading.Event()

    def _run(self):
        try:
            self.result = self.func()
        except Exception as e:
            self.error = e
        finally:
            self.done.set()


try:  # pragma: no cover - 取决于运行环境
    from PySide6.QtCore import QRunnable, QThreadPool

    class _QtRunnable(QRunnable, _Runnable):
        def __init__(self, func):
            QRunnable.__init__(self)
            _Runnable.__init__(self, func)

        def run(self):
            self._run()

except Exception:  # pragma: no cover
    _QtRunnable = None
    QThreadPool = None


class ToolExecutor:
    """在线程池里跑工具，超时返回结构化错误而不是炸掉 Agent 线程"""

    def __init__(self, default_timeout: int = 30, use_qt: bool = True):
        self.default_timeout = default_timeout
        self.use_qt = use_qt and (_QtRunnable is not None)

    def execute(self, tool, args: dict, timeout: Optional[int] = None,
                on_elapsed: Optional[Callable] = None) -> dict:
        timeout = timeout or getattr(tool, "timeout", self.default_timeout) or self.default_timeout
        started = time.time()
        runnable_holder = {}

        def _job():
            return tool.execute(args or {})

        if self.use_qt:
            try:
                runnable = _QtRunnable(_job)
                pool = QThreadPool.globalInstance()
                if pool.maxThreadCount() < 4:
                    pool.setMaxThreadCount(4)
                pool.start(runnable)
                holder = runnable
            except Exception:
                holder = None
        else:
            holder = None

        if holder is None:
            holder = _Runnable(_job)
            threading.Thread(target=holder._run, daemon=True).start()

        finished = holder.done.wait(timeout=timeout)
        elapsed = time.time() - started
        if on_elapsed:
            try:
                on_elapsed(elapsed)
            except Exception:
                pass

        if not finished:
            return {
                "success": False,
                "error": f"工具 {getattr(tool, 'name', '?')} 执行超时（>{timeout}s）",
                "error_type": "timeout",
                "tool": getattr(tool, "name", "?"),
                "timeout_s": timeout,
                "hint": "可以改用更轻量的参数重试，或换一条路径（例如直接给 FC/BOLD 文件）。",
            }
        if holder.error is not None:
            return {
                "success": False,
                "error": f"{type(holder.error).__name__}: {holder.error}",
                "error_type": type(holder.error).__name__,
                "tool": getattr(tool, "name", "?"),
            }
        result = holder.result
        if not isinstance(result, dict):
            result = {"success": True, "result": jsonable(result)}
        result.setdefault("tool", getattr(tool, "name", "?"))
        result["elapsed_s"] = round(elapsed, 3)
        return result


# ==================== 停止条件 ====================

@dataclass
class LoopState:
    steps: int = 0
    tool_calls_made: int = 0
    last_had_tool_calls: bool = False
    no_progress_streak: int = 0
    recent_signatures: list = field(default_factory=list)
    budget_exhausted: bool = False
    # 连续失败追踪：LLM 常靠"换个参数再试一次"硬闯，签名去重抓不到
    last_tool_name: str = ""
    fail_streak: int = 0
    # Plan-as-tool-chain：计划当骨架，而不是摆设
    plan_tools: list = field(default_factory=list)       # 计划全量（已与用户确认）
    remaining_steps: list = field(default_factory=list)  # 还没走到的计划步骤
    plan_stray: int = 0                                  # 连续多少步没推进计划
    plan_invalid: list = field(default_factory=list)     # 计划里不存在的工具（已剔除）


class StopCondition:
    """三重判断：步数上限 / 最后一轮是否还在调工具 / 连续无进展（含连续失败）"""

    def __init__(self, max_steps: int = 8, max_no_progress: int = 2,
                 max_fail_streak: int = 3):
        self.max_steps = max_steps
        self.max_no_progress = max_no_progress
        self.max_fail_streak = max_fail_streak

    def signature(self, name: str, args: dict) -> str:
        try:
            return f"{name}:{dumps_safe(args, sort_keys=True)[:200]}"
        except Exception:
            return f"{name}:?"

    def note_call(self, state: LoopState, name: str, args: dict, success: bool = True):
        state.tool_calls_made += 1
        sig = self.signature(name, args)
        if sig in state.recent_signatures:
            state.no_progress_streak += 1
        else:
            state.no_progress_streak = 0
        state.recent_signatures.append(sig)
        state.recent_signatures = state.recent_signatures[-6:]

        # 同一工具连续失败 → 极可能是"猜路径"式死循环，必须打断
        if not success:
            state.fail_streak = state.fail_streak + 1 if name == state.last_tool_name else 1
        else:
            state.fail_streak = 0
        state.last_tool_name = name

    def should_stop(self, state: LoopState, last_had_tool_calls: bool) -> tuple:
        if state.steps >= self.max_steps:
            return True, "max_steps"
        if state.fail_streak >= self.max_fail_streak:
            return True, "fail_streak"
        if state.no_progress_streak >= self.max_no_progress:
            return True, "no_progress"
        if not last_had_tool_calls and state.tool_calls_made > 0:
            return True, "answered"
        if not last_had_tool_calls and state.steps >= 1:
            return True, "answered"
        return False, ""


# ==================== 流式输出分段 ====================

_SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?；;\n])")


def stream_chunks(text: str, min_len: int = 12, max_len: int = 60):
    """按句子 / 标点切段，避免 10 字符一次的固定节奏。"""
    if not text:
        return
    buf = ""
    for piece in _SENT_SPLIT_RE.split(text):
        if piece is None:
            continue
        buf += piece
        if len(buf) >= min_len or piece.endswith("\n"):
            yield buf
            buf = ""
    if buf:
        yield buf


# ==================== Planner ====================

PLAN_PROMPT = (
    "你是任务规划器。根据用户诉求和可用工具，输出严格的 JSON 计划，不要输出多余文字：\n"
    '{"steps": [{"tool": "工具名", "reason": "为什么需要这一步"}], '
    '"needs_data": true/false, "risk_note": "需要注意的点（可为空）"}\n'
    "可用工具清单：\n{tool_list}\n"
    "规则：只选确实需要的工具；能直接给 FC/BOLD 路径时不要规划特征提取；"
    "不确定的信息用 null 而不是猜测。"
)


@dataclass
class Plan:
    steps: list = field(default_factory=list)
    needs_data: bool = False
    risk_note: str = ""
    raw: str = ""

    def to_text(self) -> str:
        if not self.steps:
            return "无需工具，直接回答"
        lines = []
        for i, s in enumerate(self.steps, 1):
            lines.append(f"{i}. {s.get('tool', '?')} —— {s.get('reason', '')}")
        if self.risk_note:
            lines.append(f"注意：{self.risk_note}")
        return "\n".join(lines)

    @property
    def tools(self) -> list:
        return [str(s.get("tool", "")).strip()
                for s in (self.steps or []) if str(s.get("tool", "")).strip()]


def plan_alignment_score(plan_tools: list, called_tools: list):
    """用户预期一致性：用户看到的计划 vs 实际执行的工具序列。

    follow   = 计划里承诺的工具，实际做了多少（说到做到没）
    precise  = 实际做的工具，有多少在计划内（有没有偷偷多干）
    取两者 F1：只兑现一半、或计划外乱加工具，都会被扣分。

    计划为空时返回 None——没有计划就没有"预期一致性"可言，不该污染均值。
    """
    plan_tools = [t for t in (plan_tools or []) if t]
    if not plan_tools:
        return None
    called_tools = [t for t in (called_tools or []) if t]
    planned = [t for t in plan_tools if t in called_tools]
    follow = len(planned) / len(plan_tools)
    precise = len(planned) / len(called_tools) if called_tools else 0.0
    if follow + precise <= 0:
        return 0.0
    return round(2 * follow * precise / (follow + precise), 4)


def make_plan(llm, user_msg: str, tool_list: str) -> Plan:
    """先让 LLM 出计划（JSON），UI 可先展示给用户确认"""
    try:
        resp = llm.chat(
            [
                {"role": "system", "content": PLAN_PROMPT.format(tool_list=tool_list)},
                {"role": "user", "content": user_msg},
            ],
            tools=None,
        )
        content = resp.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        match = re.search(r"\{[\s\S]*\}", content)
        if not match:
            return Plan(raw=content)
        data = json.loads(match.group(0))
        return Plan(
            steps=data.get("steps", []) or [],
            needs_data=bool(data.get("needs_data", False)),
            risk_note=str(data.get("risk_note", "") or ""),
            raw=content,
        )
    except Exception:
        return Plan()


# ==================== Agent 主循环 ====================

class AgentLoop:
    """一次完整对话：LLM ⇄ 工具，直到 StopCondition 触顶"""

    def __init__(self, llm, tools: list, tool_map: dict,
                 system_prompt: str = "", max_steps: int = 8,
                 tool_timeout: int = 30, session_id: str = "",
                 callbacks: Optional[dict] = None):
        self.llm = llm
        self.tools = tools
        self.tool_map = tool_map
        self.schemas = [t.schema() for t in tools if getattr(t, "visible", True)]
        # 从 llm 读上下文预算；拿不到（自研客户端 / mock）就为 0，退回 turn 截断
        self.context_window = int(getattr(llm, "context_window", 0) or 0)
        self.output_reserve = int(
            getattr(llm, "max_output_reserve", 0) or DEFAULT_OUTPUT_RESERVE)
        self.history_budget = compute_history_budget(
            system_prompt, self.context_window, self.output_reserve)
        self.builder = MessageBuilder(
            system_prompt,
            max_history_tokens=self.history_budget,
            context_window=self.context_window,
            output_reserve=self.output_reserve,
        )
        self.stopper = StopCondition(max_steps=max_steps)
        self.executor = ToolExecutor(default_timeout=tool_timeout)
        self.session_id = session_id
        self.cb = callbacks or {}

    # ---------------- 回调 ----------------

    def _emit(self, key, *args):
        fn = self.cb.get(key)
        if fn:
            try:
                fn(*args)
            except Exception:
                pass

    # ---------------- 计划骨架（Plan-as-tool-chain） ----------------

    def _plan_hint(self, state: LoopState) -> str:
        """下一步该干什么；计划走完就提示收尾。

        只点下一个工具，不罗列剩余清单——一次甩出一串工具名会诱导模型跳步乱选。
        """
        if not state.plan_tools:
            return ""
        if not state.remaining_steps:
            return PLAN_DONE_HINT
        return PLAN_NEXT_HINT.format(tool=state.remaining_steps[0])

    def _plan_request_messages(self, messages: list, state: LoopState) -> list:
        """拼出本次请求真正发出去的 messages。

        hint 合并进**最后一条 user 消息**，而不是新挂一条 user：
        - 独立挂一条会抢走"最近一条指令"的位置，模型容易只顾计划、弱化用户原始诉求；
        - 合并进去则用户原话仍在最前面，计划提醒跟在后面当补充。

        另外 hint 只写临时副本，**不写回持久历史**——否则工具结果会插在 hint 后面，
        下一步再也摘不掉，历史里会堆一串过期的"下一步应调用 X"。
        顺带还有个好处：持久 messages 保持稳定，prompt 前缀缓存能持续命中。
        """
        hint = self._plan_hint(state)
        if not hint:
            return messages
        out = list(messages)
        for i in range(len(out) - 1, -1, -1):
            if out[i].get("role") == "user":
                merged = dict(out[i])
                merged["content"] = f"{out[i].get('content') or ''}\n\n{hint}"
                out[i] = merged
                return out
        return out + [{"role": "user", "content": hint}]

    def _next_tool_choice(self, state: LoopState) -> tuple:
        """决定这一步的 tool_choice，返回 (tool_choice, 是否强制)。

        冷启动和"连续偏离"时把计划下一步直接指定给模型（保骨架）；
        其余时间交回 auto（保灵活：模型可以先插一个它认为必要的前置步骤）。
        """
        if not state.remaining_steps:
            return "auto", False
        nxt = state.remaining_steps[0]
        if nxt not in self.tool_map:
            return "auto", False      # 计划里写了个不存在的工具，别硬来
        if state.tool_calls_made == 0 or state.plan_stray >= MAX_PLAN_STRAY:
            return {"type": "function", "function": {"name": nxt}}, True
        return "auto", False

    @staticmethod
    def _advance_plan(state: LoopState, called_names: list):
        """按本步实际调用推进计划。

        命中的全部出队（一次并行调用可能一次吃掉计划里好几步）；
        一步都没推进才计一次偏离，偏离累积到 MAX_PLAN_STRAY 就强制纠偏。
        """
        if not state.remaining_steps:
            return
        before = len(state.remaining_steps)
        state.remaining_steps = [t for t in state.remaining_steps
                                 if t not in (called_names or [])]
        state.plan_stray = 0 if len(state.remaining_steps) < before else state.plan_stray + 1

    # ---------------- 主流程 ----------------

    def run(self, history: list, extra_system: str = "",
            user_override: str = "", plan=None) -> dict:
        messages = self.builder.build(history, extra_system=extra_system)
        state = LoopState()
        tool_steps = []

        # Plan-as-tool-chain：把已确认的计划挂成骨架，执行中仍允许模型插步
        if plan is not None:
            raw = [str(t).strip() for t in (
                getattr(plan, "tools", None)
                or [s.get("tool", "") for s in (getattr(plan, "steps", None) or [])])
                if str(t).strip()]
            # 计划里可能写了不存在的工具（模型幻觉）：既执行不了、也算不进兑现率，
            # 从骨架里剔除，单独留在 invalid 里供调用方诊断
            state.plan_tools = [t for t in raw if t in self.tool_map]
            state.plan_invalid = [t for t in raw if t not in self.tool_map]
            state.remaining_steps = list(state.plan_tools)
        final_text = ""
        stop_reason = "answered"
        usage_total = {}

        try:
            from agent import logging_utils as _log
        except Exception:
            _log = None

        while state.steps < self.stopper.max_steps:
            state.steps += 1
            state.last_had_tool_calls = False

            # 计划引导临时挂在尾部，不写回 messages（详见 _plan_request_messages）
            req_messages = self._plan_request_messages(messages, state)
            tool_choice, forced = self._next_tool_choice(state)

            try:
                resp = self.llm.chat(jsonable(req_messages), tools=self.schemas,
                                      tool_choice=tool_choice)
            except Exception as e:
                # 部分 provider（本地 / 自研）不支持"指定函数"，退回 auto 再试一次
                if not forced:
                    return {
                        "success": False,
                        "text": "",
                        "error": f"调用大模型失败: {type(e).__name__}: {e}",
                        "tool_steps": tool_steps,
                    }
                try:
                    resp = self.llm.chat(jsonable(messages), tools=self.schemas,
                                          tool_choice="auto")
                except Exception as e2:
                    return {
                        "success": False,
                        "text": "",
                        "error": f"调用大模型失败: {type(e2).__name__}: {e2}",
                        "tool_steps": tool_steps,
                    }

            usage = (resp.get("usage") or {}) if isinstance(resp, dict) else {}
            if usage:
                usage_total = usage
                if _log:
                    _log.token_usage(self.session_id, usage, state.steps)
                # 真实占用率 vs 估算预算：自适应截断调得准不准，全看这条
                prompt_tokens = usage.get("prompt_tokens", 0) or 0
                if _log and self.context_window > 0 and prompt_tokens > 0:
                    _log.context_utilization(
                        self.session_id,
                        prompt_tokens / self.context_window,
                        prompt_tokens,
                        self.context_window,
                        state.steps,
                    )

            choice = (resp.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            if hasattr(msg, "model_dump"):
                msg = msg.model_dump()

            raw_tool_calls = msg.get("tool_calls") or []
            tool_calls = MessageBuilder.normalize_tool_calls(raw_tool_calls)

            if _log:
                _log.log_messages(self.session_id, jsonable(messages), state.steps)

            if tool_calls:
                state.last_had_tool_calls = True
                messages.append(MessageBuilder.assistant_entry(msg.get("content") or "", tool_calls))

                for tc in tool_calls:
                    name = tc["function"]["name"]
                    raw_args = tc["function"]["arguments"]
                    try:
                        args = json.loads(raw_args) if raw_args else {}
                    except json.JSONDecodeError:
                        args = {}
                    if not isinstance(args, dict):
                        args = {}

                    self._emit("tool_start", name, args)
                    if _log:
                        _log.tool_start(self.session_id, name, args)

                    tool = self.tool_map.get(name)
                    if tool is None:
                        result = {"success": False, "error": f"未知工具: {name}",
                                  "error_type": "unknown_tool"}
                    else:
                        result = self.executor.execute(tool, args)

                    ok = bool(result.get("success"))
                    self.stopper.note_call(state, name, args, success=ok)
                    summary = tool.summarize(result) if tool else str(result)[:60]
                    self._emit("tool_done", name, result, summary)
                    if _log:
                        _log.tool_done(self.session_id, name, ok,
                                       result.get("elapsed_s", 0), summary)

                    tool_steps.append({
                        "name": name,
                        "args": jsonable(args),
                        "success": ok,
                        "result_summary": summary,
                    })
                    messages.append(MessageBuilder.tool_entry(tc["id"], result))

                # 计划推进：本步实际调用了哪些工具
                self._advance_plan(state, [tc["function"]["name"] for tc in tool_calls])

                # 工具跑完后判断是否该收尾
                should, reason = self.stopper.should_stop(state, True)
                if should and reason == "fail_streak":
                    messages.append({"role": "user",
                                     "content": FAIL_STREAK_NOTE.format(
                                         tool=state.last_tool_name, n=state.fail_streak)})
                    state.fail_streak = 0      # 给它一次改过的机会，再犯就撞 max_steps
                    continue
                if should and reason == "no_progress":
                    messages.append({"role": "user", "content": NO_PROGRESS_NOTE})
                    state.no_progress_streak = 0
                    continue
                if should and reason == "max_steps":
                    state.budget_exhausted = True
                    messages.append({"role": "user", "content": BUDGET_NOTE})
                    continue
                continue

            # 没有工具调用 → 这是最终答复
            content = msg.get("content") or ""
            final_text = content
            stop_reason = "answered"
            break

        else:
            # 步数用尽仍未给出答复：再要一次总结
            state.budget_exhausted = True
            messages.append({"role": "user", "content": BUDGET_NOTE})
            try:
                resp = self.llm.chat(jsonable(messages), tools=None)
                final_text = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
                stop_reason = "max_steps"
            except Exception as e:
                final_text = f"⚠ 已用尽步骤预算，且最后一次调用失败：{e}"
                stop_reason = "error"

        called_all = [s["name"] for s in tool_steps]
        alignment = plan_alignment_score(state.plan_tools, called_all)
        if _log and state.plan_tools:
            _log.plan_alignment(self.session_id, state.plan_tools, called_all, alignment)

        return {
            "success": True,
            "text": final_text,
            "tool_steps": tool_steps,
            "steps": state.steps,
            "stop_reason": stop_reason,
            "usage": usage_total,
            "budget_exhausted": state.budget_exhausted,
            "plan": {
                "tools": state.plan_tools,
                "remaining": state.remaining_steps,
                "invalid": state.plan_invalid,
                "called": called_all,
                "stray": state.plan_stray,
                "alignment": alignment,
            },
        }


# Plan-as-tool-chain 的引导语。刻意挂在 messages 末尾而不是塞进 system：
# 改 system 会让整段 prompt 前缀缓存失效，改尾部不影响前面已缓存的部分。
PLAN_HINT_PREFIX = "【执行计划】"

# 措辞刻意避开"下一步 / 流程 / 步骤"这类词：它们与工具自身的语义关键词重叠，
# 会把模型（乃至关键词型调用方）带偏到"讲流程"的工具上去。
PLAN_NEXT_HINT = (
    "【执行计划】按已与用户确认过的安排，接下来请调用：{tool}。\n"
    "若你判断需要先补一次别的前置调用（例如先把数据文件找出来），可以先调；"
    "但不要长时间偏离——连续几轮没推进这份安排，我会直接指定 {tool}。\n"
    "信息已经足够时直接给出最终答复，不必把整份安排跑完。"
)

PLAN_DONE_HINT = (
    "【执行计划】这份安排里的工具已全部调用过。信息足够就直接给出最终答复；"
    "只有确实缺关键信息时才追加调用，并说明为什么。"
)

BUDGET_NOTE = (
    "【系统提示】你已经用尽了本轮的工具调用预算。"
    "请立即停止调用任何工具，综合目前已经获得的全部结果，"
    "直接给出完整、可执行的最终答复（若信息不足，明确说明缺什么、下一步让用户做什么）。"
)

NO_PROGRESS_NOTE = (
    "【系统提示】你刚才重复调用了相同或等价的工具，没有产生新信息。"
    "请换一种思路，或直接基于已有结果给出结论，不要再重复调用。"
)

FAIL_STREAK_NOTE = (
    "【系统提示】工具 {tool} 已连续失败 {n} 次。大概率是你在猜参数（尤其是猜文件路径）。\n"
    "请立刻停止重试该工具，改为：\n"
    "1) 用 find_data_files 按样本编号搜索真实路径（只知道样本 ID 时必须先搜）；\n"
    "2) 或换一条完全不同的路线（例如跳过预处理，直接对用户给的文件做 check_ready）；\n"
    "3) 若确实缺少必要信息，直接告诉用户缺什么、请他补充，而不是继续猜。"
)

__all__ = ["MessageBuilder", "ToolExecutor", "StopCondition", "LoopState",
           "AgentLoop", "AgentRunResult", "Plan", "make_plan", "stream_chunks",
           "BUDGET_NOTE", "NO_PROGRESS_NOTE", "FAIL_STREAK_NOTE"]


# 兼容别名
AgentRunResult = dict
