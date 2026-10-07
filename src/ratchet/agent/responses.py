import json
import re
from uuid import uuid4

RAW_START = re.compile(
    r"^(?:analysis(?=[A-Z]|\s+[A-Z])|assistant(?:analysis|commentary|final)|"
    r"<\|(?:im_start|start|meta_sep|channel)\|>)"
)
CHANNEL_HEADER = re.compile(
    r"(?:(?:<\|(?:im_start|start)\|>)?assistant"
    r"(?:\s+to=(?:functions\.)?(?P<before>[A-Za-z_][\w-]*))?"
    r"(?:<\|(?:meta_sep|channel)\|>)?|<\|(?:meta_sep|channel)\|>)"
    r"(?P<channel>analysis|commentary|final)"
    r"(?:\s+to=(?:functions\.)?(?P<after>[A-Za-z_][\w-]*))?"
    r"\s*(?:<\|(?:meta_sep|constrain)\|>)?"
    r"(?:(?:json|code)(?=\s*(?:<\|(?:im_sep|message)\|>|[\{\[])))?\s*"
    r"(?:<\|(?:im_sep|message)\|>)?\s*"
)
CHANNEL_END = re.compile(r"(?:\s*<\|(?:im_end|end|fim_suffix|ghissue|call|return)\|>)*\s*\Z")
INVALID_RESPONSE = (
    "[Error] Model returned unreadable channel/tool output instead of a valid response."
)
REPAIR_NOTE = (
    "Your previous response used malformed channel/tool markup. "
    "Continue the task using structured tool_calls with valid JSON arguments, "
    "or return a user-facing final answer. Do not print analysis, channel headers, "
    "or textual tool-call syntax in message content."
)


def _raw_response(content: str, tools: list[dict] | None) -> dict:
    offered = {tool["function"]["name"]: tool["function"] for tool in tools or []}
    calls = []
    offset = 0
    while header := CHANNEL_HEADER.search(content, offset):
        name = header["before"] or header["after"]
        offset = header.end()
        if name:
            if name not in offered or header["channel"] == "final":
                raise ValueError("Tool is not offered.")
            arguments, end = json.JSONDecoder().raw_decode(content, offset)
            required = offered[name].get("parameters", {}).get("required", [])
            if not isinstance(arguments, dict) or not set(required) <= arguments.keys():
                raise ValueError("Invalid tool arguments.")
            next_header = CHANNEL_HEADER.search(content, end)
            tail = content[end:next_header.start() if next_header else len(content)]
            if not CHANNEL_END.fullmatch(tail):
                raise ValueError("Unexpected text after tool arguments.")
            calls.append({
                "id": f"call_{uuid4().hex}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            })
            offset = end
        elif header["channel"] == "final":
            if not calls:
                final = CHANNEL_END.sub("", content[offset:]).strip()
                if final:
                    return {"status": "success", "content": final}
            break
    if not calls:
        raise ValueError("No final answer or tool call.")
    return {"status": "success", "content": "", "tool_calls": calls}


def normalize_message(message: dict, tools: list[dict] | None) -> dict:
    content = message.get("content") or ""
    raw = bool(RAW_START.match(content.lstrip()))
    native = message.get("tool_calls")
    if native:
        return {"status": "success", "content": "" if raw else content, "tool_calls": native}
    if raw:
        try:
            return _raw_response(content.lstrip(), tools)
        except ValueError:
            return {"status": "invalid_response", "content": INVALID_RESPONSE}
    if not content and (message.get("reasoning") or message.get("reasoning_content")):
        return {"status": "invalid_response", "content": INVALID_RESPONSE}
    return {"status": "success", "content": content}
