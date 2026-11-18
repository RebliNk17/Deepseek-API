"""Translate between OpenAI's chat-completions shapes and our DeepSeek client.

DeepSeek's protocol has no system/role channel — just a single `prompt` string.
So we flatten the OpenAI `messages` array into one prompt, and wrap DeepSeek's
text output back into OpenAI response/stream objects.

Because DeepSeek's web API doesn't expose a native tool-call channel, we
implement a shim when `tools` is supplied: incoming `tools` become a
prompt-level instruction asking the model to emit JSON like {"tool_calls":[...]};
we then parse that JSON out of the model's text and translate it into OpenAI's
`tool_calls` shape in the response.
"""

from __future__ import annotations

import ast
import base64
import json
import mimetypes
import re
import time
import urllib.parse
import uuid
from typing import Any, Iterable, List, Optional, Tuple, Union

import httpx

from deepseek.client import RESPONSE, THINK

from .schemas import ChatMessage

_ROLE_LABELS = {
    "system": "System",
    "developer": "Developer",
    "user": "User",
    "assistant": "Assistant",
    "tool": "Tool",
    "function": "Function",
}

_MIME_TO_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/svg+xml": ".svg",
    "application/pdf": ".pdf",
}

_BACKSLASH = chr(92)
_DOUBLE_QUOTE = chr(34)
_SINGLE_QUOTE = chr(39)


def _text_of(content: Any) -> str:
    """Extract plain text from a message's content (string or list-of-parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                if p.get("type") == "text":
                    parts.append(p.get("text", ""))
        return "\n".join(parts)
    return str(content)


def extract_images(messages: List[ChatMessage]) -> List[Tuple[bytes, str, str]]:
    """Extract images/files from OpenAI multimodal message content parts.
    Returns a list of `(file_bytes, filename, mime_type)` tuples.
    Supports `data:<mime>;base64,...` data URLs and `http(s)://` URLs.
    """
    attachments: List[Tuple[bytes, str, str]] = []
    idx = 0
    for m in messages:
        if not isinstance(m.content, list):
            continue
        for p in m.content:
            if not isinstance(p, dict):
                continue
            ptype = p.get("type")
            url = None
            if ptype == "image_url":
                img_obj = p.get("image_url")
                url = img_obj.get("url") if isinstance(img_obj, dict) else img_obj
            elif ptype in ("input_image", "image"):
                url = p.get("image_url") or p.get("url")
                if isinstance(url, dict):
                    url = url.get("url")
            if not url or not isinstance(url, str):
                continue
            idx += 1
            if url.startswith("data:"):
                header, _, b64_data = url.partition(",")
                mime_type = "image/png"
                if ";" in header:
                    mime_type = header[5:].split(";")[0] or "image/png"
                elif len(header) > 5:
                    mime_type = header[5:]
                ext = _MIME_TO_EXT.get(mime_type) or mimetypes.guess_extension(mime_type) or ".png"
                raw = base64.b64decode(b64_data)
                attachments.append((raw, f"image_{idx}{ext}", mime_type))
            elif url.startswith(("http://", "https://")):
                r = httpx.get(url, timeout=30.0, follow_redirects=True)
                r.raise_for_status()
                mime_type = (r.headers.get("content-type") or "image/png").split(";")[0].strip()
                parsed_path = urllib.parse.urlparse(url).path
                ext = _MIME_TO_EXT.get(mime_type) or mimetypes.guess_extension(mime_type) or ".png"
                name = parsed_path.rsplit("/", 1)[-1] if "/" in parsed_path else ""
                if not name or "." not in name:
                    name = f"image_{idx}{ext}"
                attachments.append((r.content, name, mime_type))
    return attachments


def _format_tools(
    tools: List[dict],
    tool_choice: Optional[Union[str, dict]] = None,
) -> str:
    """Turn OpenAI tool definitions into a prompt-level instruction.

    DeepSeek's web API has no native tool schema, so we describe the functions in
    prose and ask the model to emit a strict JSON block. The sample is written
    with *double quotes* on purpose: the model tends to mirror the example's
    style, and a single-quoted sample makes it emit non-JSON output.
    """
    if not tools or tool_choice == "none":
        return ""

    tool_descs = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        func = tool.get("function", {})
        name = func.get("name", "")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        tool_descs.append(
            f"- {name}: {desc}\n"
            f"  Parameters: {json.dumps(params, ensure_ascii=False, indent=2)}"
        )

    if not tool_descs:
        return ""

    instruction = (
        "You have access to the following functions. To call a function, "
        "respond with ONLY a single valid JSON object (double quotes, no "
        "markdown, no extra text) in exactly this format:\n"
        '{"tool_calls": [{"name": "function_name", "arguments": {"arg": "value"}}]}\n'
        "If you do not need to call a function, respond normally in plain text.\n\n"
        "Available functions:\n" + "\n".join(tool_descs)
    )

    if tool_choice == "required":
        instruction += "\n\nYou must call one of the functions."
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        forced = tool_choice.get("function", {}).get("name", "")
        if forced:
            instruction += f"\n\nYou must call the function {forced}."

    return instruction


def _extract_json_object(text: str) -> Optional[str]:
    """Return the first balanced {...} block that contains a tool_calls key.

    Finds the tool_calls key, walks back to the enclosing open brace, then
    brace-matches — string-aware, so braces inside string literals are ignored.
    """
    idx = text.find("tool_calls")
    if idx == -1:
        return None
    start = text.rfind("{", 0, idx)
    if start == -1:
        return None

    depth = 0
    in_str = False
    quote = ""
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == _BACKSLASH:
                esc = True
            elif ch == quote:
                in_str = False
        else:
            if ch == _DOUBLE_QUOTE or ch == _SINGLE_QUOTE:
                in_str = True
                quote = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def _loads_tolerant(blob: str):
    """Parse JSON, falling back to Python-literal syntax (single quotes)."""
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(blob)
    except Exception:
        return None


def _parse_dsml_tool_calls(text: str) -> Tuple[Optional[List[dict]], str]:
    """Parse DeepSeek's native DSML tool-calling format (`<｜DSML｜...>` or `< | DSML | | ...>`)
    or Anthropic/XML `<function_calls>/<invoke>` blocks.

    Returns `(tool_calls, cleaned_text_before_calls)`.
    """
    # Match `<｜DSML｜function_calls>` / `< | | DSML | | calls>` / `<function_calls>`
    dsml_Token = r"(?:<(?:[\s｜|]*DSML[\s｜|]*)?|\s*<\s*)"
    end_dsml_Token = r"(?:</(?:[\s｜|]*DSML[\s｜|]*)?|\s*</\s*)"

    invoke_pattern = re.compile(
        dsml_Token + r'invoke\s+name="([^"]+)"\s*>(.*?)' + end_dsml_Token + r"invoke\s*>",
        re.DOTALL,
    )
    param_pattern = re.compile(
        dsml_Token
        + r'parameter\s+name="([^"]+)"(?:\s+string="(true|false)")?\s*>(.*?)'
        + end_dsml_Token
        + r"parameter\s*>",
        re.DOTALL,
    )

    invokes = list(invoke_pattern.finditer(text))
    if not invokes:
        return None, text

    result = []
    for inv in invokes:
        func_name = inv.group(1).strip()
        body = inv.group(2)
        args: dict = {}
        for pm in param_pattern.finditer(body):
            p_name = pm.group(1).strip()
            is_str = pm.group(2)
            p_val_raw = pm.group(3)
            if is_str == "false":
                try:
                    args[p_name] = json.loads(p_val_raw)
                except Exception:
                    args[p_name] = p_val_raw
            else:
                args[p_name] = p_val_raw
        result.append(
            {
                "id": "call_" + uuid.uuid4().hex[:8],
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }
        )

    # Strip the `<...calls>...</...calls>` wrapper (or from the first invoke onwards)
    calls_start = re.search(
        dsml_Token + r"(?:function_)?calls\s*>",
        text,
    )
    cut_idx = calls_start.start() if calls_start else invokes[0].start()
    cleaned = text[:cut_idx].strip()
    return (result if result else None), cleaned


def parse_tool_calls_with_text(text: str) -> Tuple[Optional[List[dict]], Optional[str]]:
    """Extract tool calls and any accompanying assistant text from raw output.

    Supports both:
    1. Our prompted JSON format: `{"tool_calls": [{"name": ..., "arguments": ...}]}`
    2. DeepSeek's native DSML format: `<｜DSML｜function_calls><｜DSML｜invoke name="...">...`
       (including space-separated `< | DSML | | calls>` variants).

    Returns `(tool_calls, remaining_text)`.
    """
    if not text:
        return None, text

    # 1. Check for JSON `{"tool_calls": [...]}`
    blob = _extract_json_object(text)
    if blob is not None:
        data = _loads_tolerant(blob)
        if isinstance(data, dict) and isinstance(data.get("tool_calls"), list):
            result = []
            for tc in data["tool_calls"]:
                if not isinstance(tc, dict):
                    continue
                name = tc.get("name")
                args = tc.get("arguments", {})
                if not name:
                    continue
                args_str = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
                result.append(
                    {
                        "id": "call_" + uuid.uuid4().hex[:8],
                        "type": "function",
                        "function": {"name": name, "arguments": args_str},
                    }
                )
            if result:
                idx = text.find(blob)
                prefix = text[:idx].strip() if idx > 0 else ""
                # Strip markdown code fences around the JSON blob if any
                prefix = re.sub(r"```(?:json)?\s*$", "", prefix).strip()
                return result, (prefix or None)

    # 2. Check for DeepSeek native DSML `<｜DSML｜...>` / `< | DSML | | ...>` calls
    dsml_calls, cleaned = _parse_dsml_tool_calls(text)
    if dsml_calls:
        return dsml_calls, (cleaned or None)

    return None, text


def parse_tool_calls(text: str) -> Optional[List[dict]]:
    """Extract tool calls from the model's raw text output."""
    calls, _ = parse_tool_calls_with_text(text)
    return calls


def messages_to_prompt(
    messages: List[ChatMessage],
    tools: Optional[List[dict]] = None,
    tool_choice: Optional[Union[str, dict]] = None,
) -> str:
    """Flatten a chat history into a single prompt DeepSeek can answer.

    A lone user message with no tools is sent verbatim. Multi-turn /
    system-prompted conversations are serialised with role labels and a
    trailing 'Assistant:' cue so the model continues in the right voice.

    `tools` (when present and `tool_choice != "none"`) is rendered as an
    instruction block at the top. Tool-result and prior assistant-with-tool-calls
    messages are rendered so the model sees the call/answer pair.
    """
    tool_instruction = _format_tools(tools, tool_choice) if tools else ""

    if len(messages) == 1 and messages[0].role == "user" and not tool_instruction:
        return _text_of(messages[0].content)

    lines = []
    if tool_instruction:
        lines.append(f"System: {tool_instruction}")

    for m in messages:
        role_lower = m.role.lower()
        if role_lower == "tool":
            lines.append(f"Tool result: {_text_of(m.content)}")
        elif role_lower == "assistant" and m.tool_calls:
            calls = []
            for tc in m.tool_calls:
                func = tc.get("function", {})
                calls.append(
                    {
                        "name": func.get("name"),
                        "arguments": func.get("arguments"),
                    }
                )
            lines.append(
                f"Assistant: {json.dumps({'tool_calls': calls}, ensure_ascii=False)}"
            )
        else:
            label = _ROLE_LABELS.get(role_lower, m.role.capitalize())
            lines.append(f"{label}: {_text_of(m.content)}")

    lines.append("Assistant:")
    return "\n\n".join(lines)


def _now() -> int:
    return int(time.time())


def _id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


def _est_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) — DeepSeek's web API gives us no count."""
    return max(1, len(text) // 4)


def completion_response(
    model: str,
    content: str,
    prompt: str,
    conversation_id: Optional[str] = None,
    reasoning: str = "",
    tool_calls: Optional[List[dict]] = None,
) -> dict:
    """A full (non-streaming) OpenAI chat.completion object.

    When `tool_calls` is present the message is shaped to OpenAI's spec:
    `content` becomes `null` and `finish_reason` is `"tool_calls"`.

    `conversation_id` is an extra top-level field (outside OpenAI's schema) you
    send back to resume the conversation. DeepThink text, if any, goes in
    `message.reasoning_content` (DeepSeek's official API uses the same field).
    """
    pt, ct = _est_tokens(prompt), _est_tokens(reasoning + content)
    message: dict = {
        "role": "assistant",
        "content": content,
        "refusal": None,
    }
    if reasoning:
        message["reasoning_content"] = reasoning

    finish_reason = "stop"
    if tool_calls:
        message["tool_calls"] = tool_calls
        message["content"] = content or None
        finish_reason = "tool_calls"

    return {
        "id": _id(),
        "object": "chat.completion",
        "created": _now(),
        "model": model,
        "system_fingerprint": "fp_deepseek_web",
        "conversation_id": conversation_id,
        "choices": [
            {
                "index": 0,
                "message": message,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
        },
    }


def stream_chunks(
    model: str,
    stream,
    *,
    prompt: str = "",
    has_tools: bool = False,
    include_usage: bool = False,
) -> Iterable[str]:
    """Yield OpenAI SSE lines (`data: {...}\\n\\n`) for a streamed completion.

    `stream` is the client's stream object. Reply text goes out as `content`
    deltas and DeepThink text as `reasoning_content` deltas; after it's consumed
    we read its `.conversation_id` and attach it to the final chunk.

    When `has_tools` is False (the normal case), RESPONSE chunks stream
    immediately as they arrive. When `has_tools` is True, RESPONSE chunks are
    buffered (while THINK chunks still stream live) so we can detect whether
    the reply is a JSON tool-call block and emit proper `delta.tool_calls`
    with `index` fields and a single `finish_reason="tool_calls"` frame.
    """
    cid, created = _id(), _now()
    total_text: list[str] = []

    def frame(delta: dict, finish=None, extra: Optional[dict] = None) -> str:
        obj = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "system_fingerprint": "fp_deepseek_web",
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "logprobs": None,
                    "finish_reason": finish,
                }
            ],
        }
        if extra:
            obj.update(extra)
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    def usage_frame() -> str:
        full_content = "".join(total_text)
        pt, ct = _est_tokens(prompt), _est_tokens(full_content)
        usage_obj = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "system_fingerprint": "fp_deepseek_web",
            "choices": [],
            "usage": {
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": pt + ct,
            },
        }
        return f"data: {json.dumps(usage_obj, ensure_ascii=False)}\n\n"

    # First frame announces the assistant role.
    yield frame({"role": "assistant", "content": ""})

    if not has_tools:
        for kind, d in stream.events():
            if not d:
                continue
            total_text.append(d)
            if kind == RESPONSE:
                yield frame({"content": d})
            elif kind == THINK:
                yield frame({"reasoning_content": d})
        conversation_id = getattr(stream, "conversation_id", None)
        yield frame({}, finish="stop", extra={"conversation_id": conversation_id})
        if include_usage:
            yield usage_frame()
        yield "data: [DONE]\n\n"
        return

    # Tools enabled: stream THINK chunks live, buffer RESPONSE chunks to inspect for tool_calls.
    response_chunks: list[str] = []
    for kind, d in stream.events():
        if not d:
            continue
        total_text.append(d)
        if kind == THINK:
            yield frame({"reasoning_content": d})
        elif kind == RESPONSE:
            response_chunks.append(d)

    conversation_id = getattr(stream, "conversation_id", None)
    full_response = "".join(response_chunks)
    tool_calls, pre_text = parse_tool_calls_with_text(full_response)

    if tool_calls:
        if pre_text:
            yield frame({"content": pre_text})
        indexed_calls = [
            {"index": idx, **tc}
            for idx, tc in enumerate(tool_calls)
        ]
        yield frame({"tool_calls": indexed_calls})
        yield frame({}, finish="tool_calls", extra={"conversation_id": conversation_id})
    else:
        for d in response_chunks:
            yield frame({"content": d})
        yield frame({}, finish="stop", extra={"conversation_id": conversation_id})

    if include_usage:
        yield usage_frame()
    yield "data: [DONE]\n\n"


def stream_error(message: str, err_type: str = "server_error") -> Iterable[str]:
    """SSE lines reporting a failure after the stream has started."""
    err = {"error": {"message": message, "type": err_type}}
    yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"
