import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Callable

from ratchet.agent.config import load_config
from ratchet.agent.context import AgentContext
from ratchet.agent.events import TurnEvent, TurnResult
from ratchet.agent.subagent import PARALLEL_SPAWN_TOOL, max_parallel
from ratchet.agent.tools import SYSTEM_PROMPT, TOOL_SCHEMAS, execute_tool_result

DEFAULT_MAX_STEPS = 12


def _arguments(call: dict) -> dict:
    return json.loads(call["function"]["arguments"] or "{}")


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
            arguments=arguments,
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
        output, exit_code = execute_tool_result(call["function"]["name"], arguments, lane_ctx)
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
) -> TurnResult:
    if messages is None:
        messages = [{"role": "system", "content": system_prompt or SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": user_text})
    if max_steps is None:
        max_steps = load_config().get("agent", {}).get("max_steps", DEFAULT_MAX_STEPS)
    if tools_schema is None:
        tools_schema = TOOL_SCHEMAS

    ctx = AgentContext(
        sandbox_root=sandbox_root,
        call_llm_fn=call_llm_fn,
        override_config=override_config,
        on_event=on_event,
        depth=depth,
    )
    started = time.monotonic()
    tools_used: list[str] = []
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    index = 0
    step = 0

    for step in range(1, max_steps + 1):
        if on_event:
            on_event(TurnEvent(phase="thinking", step=step, depth=depth, agent=agent))
        result = call_llm_fn(messages, override_config, tools=tools_schema)

        usage = result.get("usage")
        if usage:
            prompt_tokens = (prompt_tokens or 0) + usage.get("prompt_tokens", 0)
            completion_tokens = (completion_tokens or 0) + usage.get("completion_tokens", 0)

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
                        arguments=arguments,
                        depth=depth,
                        agent=agent,
                    )
                )
            call_started = time.monotonic()
            output, exit_code = execute_tool_result(tool_name, arguments, ctx)
            if on_event:
                on_event(
                    TurnEvent(
                        phase="tool_done",
                        step=step,
                        index=index,
                        name=tool_name,
                        arguments=arguments,
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
) -> str:
    return run_turn(
        call_llm_fn, user_text, sandbox_root, override_config, on_event, messages
    ).text
