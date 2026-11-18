"""Translate between OpenAI's chat-completions shapes and our DeepSeek client.

DeepSeek's protocol has no system/role channel — just a single `prompt` string.
So we flatten the OpenAI `messages` array into one prompt, and wrap DeepSeek's
text output back into OpenAI response/stream objects.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import time
import urllib.parse
import uuid
from typing import Iterable, List, Tuple

import httpx

from deepseek.client import RESPONSE, THINK

from .schemas import ChatMessage

_ROLE_LABELS = {"system": "System", "user": "User", "assistant": "Assistant"}

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


def _text_of(content) -> str:
    """Extract plain text from a message's content (string or list-of-parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if isinstance(p, dict) and p.get("type") == "text":
            parts.append(p.get("text", ""))
    return "\n".join(parts)


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


def messages_to_prompt(messages: List[ChatMessage]) -> str:
    """Flatten a chat history into a single prompt DeepSeek can answer.

    A lone user message is sent verbatim. Multi-turn / system-prompted
    conversations are serialised with role labels and a trailing 'Assistant:'
    cue so the model continues in the right voice.
    """
    if len(messages) == 1 and messages[0].role == "user":
        return _text_of(messages[0].content)

    lines = []
    for m in messages:
        label = _ROLE_LABELS.get(m.role, m.role.capitalize())
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


def completion_response(model: str, content: str, prompt: str,
                        conversation_id: str = None, reasoning: str = "") -> dict:
    """A full (non-streaming) OpenAI chat.completion object.

    `conversation_id` is an extra top-level field (outside OpenAI's schema) you
    send back to resume the conversation. DeepThink text, if any, goes in
    `message.reasoning_content` (DeepSeek's official API uses the same field).
    """
    pt, ct = _est_tokens(prompt), _est_tokens(reasoning + content)
    message = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    return {
        "id": _id(),
        "object": "chat.completion",
        "created": _now(),
        "model": model,
        "conversation_id": conversation_id,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
        },
    }


def stream_chunks(model: str, stream) -> Iterable[str]:
    """Yield OpenAI SSE lines (`data: {...}\\n\\n`) for a streamed completion.

    `stream` is the client's stream object. Reply text goes out as `content`
    deltas and DeepThink text as `reasoning_content` deltas; after it's consumed
    we read its `.conversation_id` and attach it to the final chunk.
    """
    cid, created = _id(), _now()

    def frame(delta: dict, finish=None, extra: dict = None) -> str:
        obj = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if extra:
            obj.update(extra)
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    # First frame announces the assistant role.
    yield frame({"role": "assistant", "content": ""})
    for kind, d in stream.events():
        if not d:
            continue
        if kind == RESPONSE:
            yield frame({"content": d})
        elif kind == THINK:
            yield frame({"reasoning_content": d})
    conversation_id = getattr(stream, "conversation_id", None)
    yield frame({}, finish="stop", extra={"conversation_id": conversation_id})
    yield "data: [DONE]\n\n"


def stream_error(message: str, err_type: str = "server_error") -> Iterable[str]:
    """SSE lines reporting a failure after the stream has started."""
    err = {"error": {"message": message, "type": err_type}}
    yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"
