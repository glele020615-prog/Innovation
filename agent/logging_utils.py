"""
结构化日志：会话 messages（脱敏）+ 事件流（tool_call / error / token）
======================================================================
产物：~/.智影agent/logs/YYYY-MM-DD/
        session-<id>.jsonl   每轮 messages 快照（已脱敏）
        events.jsonl         结构化事件，可 replay
"""

import os
import json
import time
import threading

from agent.security import CONFIG_DIR
from agent.redaction import redact
from agent.util import today, now_stamp

LOG_ROOT = os.path.join(CONFIG_DIR, "logs")
_lock = threading.Lock()


def _dir() -> str:
    d = os.path.join(LOG_ROOT, today())
    os.makedirs(d, exist_ok=True)
    return d


def _append(path: str, payload: dict):
    try:
        line = json.dumps(redact(payload), ensure_ascii=False)
        with _lock, open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def log_event(session_id: str, event: dict):
    payload = {
        "ts": now_stamp(),
        "session_id": session_id,
    }
    payload.update(event)
    _append(os.path.join(_dir(), "events.jsonl"), payload)


def log_messages(session_id: str, messages: list, round_no: int = 0):
    _append(
        os.path.join(_dir(), "session-<id>.jsonl".replace("<id>", str(session_id))),
        {
            "ts": now_stamp(),
            "session_id": session_id,
            "round": round_no,
            "messages": messages,
        },
    )


def tool_start(session_id: str, name: str, args: dict):
    log_event(session_id, {"event": "tool_call_start", "tool": name, "args": args})


def tool_done(session_id: str, name: str, ok: bool, elapsed: float, summary: str = ""):
    log_event(session_id, {
        "event": "tool_call_done", "tool": name, "success": ok,
        "elapsed_s": round(elapsed, 3), "summary": summary,
    })


def error_event(session_id: str, where: str, message: str):
    log_event(session_id, {"event": "error", "where": where, "message": message})


def token_usage(session_id: str, usage: dict, round_no: int = 0):
    log_event(session_id, {"event": "token_usage", "round": round_no, "usage": usage})


def plan_alignment(session_id: str, plan_tools: list, called_tools: list,
                   score=None):
    """计划 vs 实际执行：用户预期一致性的原始素材。

    score 为 None 表示本轮没有计划（用户没点「先出计划」），不参与统计。
    """
    log_event(session_id, {
        "event": "plan_alignment",
        "plan_tools": list(plan_tools or []),
        "called_tools": list(called_tools or []),
        "score": score,
    })


def context_utilization(session_id: str, util: float, prompt_tokens: int,
                        context_window: int, round_no: int = 0):
    """上下文占用率：判断自适应截断调得准不准，全靠这条。

    util 长期 > 0.8 还能跑 → 估算偏保守，可以调小 safety_margin；
    util 很低就触发截断 → 截太狠了，可以调大 history 预算。
    """
    log_event(session_id, {
        "event": "context_utilization",
        "round": round_no,
        "util": round(float(util), 4),
        "prompt_tokens": prompt_tokens,
        "context_window": context_window,
    })
