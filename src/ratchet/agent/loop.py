import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Callable

from ratchet.agent.config import load_config
from ratchet.agent.context import AgentContext, QuestionHandler
from ratchet.agent.events import TurnEvent, TurnResult
from ratchet.agent.plan import (
    PLAN_EXECUTION_NOTE,
    PLAN_MODE_NOTE,
    PLAN_MODE_TOOLS,
    blocked_in_plan_mode,
    get_plan,
    plan_mode_schemas,
)
from ratchet.agent.responses import REPAIR_NOTE
from ratchet.agent.subagent import PARALLEL_SPAWN_TOOL, max_parallel
from ratchet.agent.tools import SYSTEM_PROMPT, TOOL_SCHEMAS, execute_tool_result

DEFAULT_MAX_STEPS = 12


def _arguments(call: dict) -> dict | None:
    try:
        return json.loads(call["function"]["arguments"] or "{}")
    except json.JSONDecodeError:
        return None


def _execute(name: str, arguments: dict | None, ctx: AgentContext) -> tuple[str, int]:
    if arguments is None:
        return f"[Error] invalid JSON arguments for {name}; resend the call with valid JSON", 1
    if ctx.plan_mode and name not in PLAN_MODE_TOOLS:
        return blocked_in_plan_mode(name), 1
    if ctx.allowed_tools is not None and name not in ctx.allowed_tools:
        return f"[Error] {name} is blocked; it is outside this agent's tool permissions", 1
    return execute_tool_result(name, arguments, ctx)


def _parallel_group(tool_calls: list[dict]) -> bool:
    if len(tool_calls) < 2:
        return False
    return all(call["function"]["name"] == PARALLEL_SPAWN_TOOL for call in tool_calls)


def _run_lanes(
    tool_calls: list[dict],
    ctx: AgentContext,
    on_event: Callable[[TurnEvent], None] | None,
    step: int,
    first_index: int,
    agent: str,
) -> list[tuple[str, str]]:
    lanes = [
        (call, _arguments(call), first_index + offset, offset + 1)
        for offset, call in enumerate(tool_calls)
    ]

    def event(phase: str, lane_parts: tuple, **extra) -> TurnEvent:
        call, arguments, index, lane = lane_parts
        return TurnEvent(
            phase=phase,
            step=step,
            index=index,
            name=call["function"]["name"],
            arguments=arguments or {},
            depth=ctx.depth,
            agent=agent,
            lane=lane,
            **extra,
        )

    if on_event:
        for parts in lanes:
            on_event(event("tool_start", parts))

    def work(parts: tuple) -> tuple[list[TurnEvent], str, int, float]:
        call, arguments, _index, lane = parts
        buffer: list[TurnEvent] = []
        lane_ctx = replace(ctx, on_event=lambda e: buffer.append(replace(e, lane=lane)))
        started = time.monotonic()
        output, exit_code = _execute(call["function"]["name"], arguments, lane_ctx)
        return buffer, output, exit_code, time.monotonic() - started

    outputs: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=min(max_parallel(), len(lanes))) as pool:
        futures = [pool.submit(work, parts) for parts in lanes]
        for parts, future in zip(lanes, futures):
            buffer, output, exit_code, elapsed = future.result()
            if on_event:
                for nested in buffer:
                    on_event(nested)
                on_event(
                    event(
                        "tool_done",
                        parts,
                        output=output,
                        exit_code=exit_code,
                        elapsed=elapsed,
                    )
                )
            outputs.append((parts[0]["id"], output))
    return outputs


def run_turn(
    call_llm_fn: Callable,
    user_text: str,
    sandbox_root: Path,
    override_config: dict | None = None,
    on_event: Callable[[TurnEvent], None] | None = None,
    messages: list[dict] | None = None,
    *,
    tools_schema: list[dict] | None = None,
    system_prompt: str | None = None,
    max_steps: int | None = None,
    depth: int = 0,
    agent: str = "",
    plan_mode: bool = False,
    ask_user: QuestionHandler | None = None,
    plan_root: Path | None = None,
) -> TurnResult:
    if messages is None:
        messages = [{"role": "system", "content": system_prompt or SYSTEM_PROMPT}]
    if max_steps is None:
        max_steps = load_config().get("agent", {}).get("max_steps", DEFAULT_MAX_STEPS)
    if tools_schema is None:
        tools_schema = TOOL_SCHEMAS
    execution_note = False
    if plan_mode:
        allowed_names = {schema["function"]["name"] for schema in tools_schema}
        tools_schema = [
            schema for schema in plan_mode_schemas()
            if schema["function"]["name"] in allowed_names
        ]
        user_text = f"{PLAN_MODE_NOTE}\n\n{user_text}"
    elif depth == 0:
        tool_names = {schema["function"]["name"] for schema in tools_schema}
        if {"get_plan", "update_plan_step"} <= tool_names:
            execution_note = get_plan(plan_root or sandbox_root)["exit_code"] == 0
    messages.append({"role": "user", "content": user_text})

    ctx = AgentContext(
        sandbox_root=sandbox_root,
        call_llm_fn=call_llm_fn,
        override_config=override_config,
        on_event=on_event,
        depth=depth,
        plan_mode=plan_mode,
        allowed_tools=frozenset(schema["function"]["name"] for schema in tools_schema),
        ask_user=ask_user,
        plan_root=plan_root,
    )
    started = time.monotonic()
    tools_used: list[str] = []
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    index = 0
    step = 0
    repaired_response = False

    for step in range(1, max_steps + 1):
        if on_event:
            on_event(TurnEvent(phase="thinking", step=step, depth=depth, agent=agent))
        request_messages = messages
        if execution_note:
            request_messages = [{"role": "system", "content": PLAN_EXECUTION_NOTE}, *messages]
        if repaired_response:
            request_messages = [{"role": "system", "content": REPAIR_NOTE}, *request_messages]
        result = call_llm_fn(request_messages, override_config, tools=tools_schema)

        if on_event and result.get("fallback_from"):
            on_event(TurnEvent(
                phase="fallback", step=step, depth=depth, agent=agent,
                name=result["model"], arguments={"from": result["fallback_from"]},
            ))

        usage = result.get("usage")
        if usage:
            prompt_tokens = (prompt_tokens or 0) + usage.get("prompt_tokens", 0)
            completion_tokens = (completion_tokens or 0) + usage.get("completion_tokens", 0)
            if on_event:
                on_event(TurnEvent(
                    phase="usage", step=step, depth=depth, agent=agent,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    context_window=result.get("context_window"),
                ))

        if result["status"] == "invalid_response" and not repaired_response:
            repaired_response = True
            continue

        if result["status"] != "success":
            return TurnResult(
                text=result["content"],
                status="error",
                steps=step,
                elapsed=time.monotonic() - started,
                tools_used=tools_used,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        tool_calls = result.get("tool_calls") or []
        if not tool_calls:
            messages.append({"role": "assistant", "content": result["content"]})
            return TurnResult(
                text=result["content"],
                status="ok",
                steps=step,
                elapsed=time.monotonic() - started,
                tools_used=tools_used,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        messages.append(
            {"role": "assistant", "content": result.get("content") or "", "tool_calls": tool_calls}
        )
        if _parallel_group(tool_calls):
            tools_used += [call["function"]["name"] for call in tool_calls]
            for call_id, output in _run_lanes(tool_calls, ctx, on_event, step, index + 1, agent):
                messages.append({"role": "tool", "tool_call_id": call_id, "content": output})
            index += len(tool_calls)
            continue

        for call in tool_calls:
            tool_name = call["function"]["name"]
            arguments = _arguments(call)
            index += 1
            tools_used.append(tool_name)
            if on_event:
                on_event(
                    TurnEvent(
                        phase="tool_start",
                        step=step,
                        index=index,
                        name=tool_name,
                        arguments=arguments or {},
                        depth=depth,
                        agent=agent,
                    )
                )
            call_started = time.monotonic()
            output, exit_code = _execute(tool_name, arguments, ctx)
            if on_event:
                on_event(
                    TurnEvent(
                        phase="tool_done",
                        step=step,
                        index=index,
                        name=tool_name,
                        arguments=arguments or {},
                        output=output,
                        exit_code=exit_code,
                        elapsed=time.monotonic() - call_started,
                        depth=depth,
                        agent=agent,
                    )
                )
            messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": output}
            )

    return TurnResult(
        text="[Error] tool call loop exceeded max iterations",
        status="max_steps",
        steps=step,
        elapsed=time.monotonic() - started,
        tools_used=tools_used,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def run_agent_turn(
    call_llm_fn: Callable,
    user_text: str,
    sandbox_root: Path,
    override_config: dict | None = None,
    on_event: Callable[[TurnEvent], None] | None = None,
    messages: list[dict] | None = None,
    plan_mode: bool = False,
    *,
    ask_user: QuestionHandler | None = None,
    plan_root: Path | None = None,
) -> str:
    return run_turn(
        call_llm_fn,
        user_text,
        sandbox_root,
        override_config,
        on_event,
        messages,
        plan_mode=plan_mode,
        ask_user=ask_user,
        plan_root=plan_root,
    ).text
