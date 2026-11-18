"""Pydantic models for the OpenAI-compatible request/response shapes we support."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict

from .config import DEFAULT_MODEL


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="allow")
    include_usage: Optional[bool] = False


class ResponseFormat(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Optional[str] = "text"  # "text" or "json_object"


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    # content is a plain string, or a list of parts (OpenAI vision-style).
    content: Union[str, List[Any], None] = None
    name: Optional[str] = None
    # For assistant messages that contain tool calls (from an OpenAI client).
    tool_calls: Optional[List[Dict[str, Any]]] = None
    # For tool-result messages (the id of the call being answered).
    tool_call_id: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = DEFAULT_MODEL
    messages: List[ChatMessage]
    stream: bool = False
    # Pass a conversation_id from a previous response to resume that thread.
    conversation_id: Optional[str] = None
    # Tools to enable for this request, independent of the model. OpenAI clients
    # pass these via extra_body: `thinking` (DeepThink), `search` (web).
    thinking: bool = False
    search: bool = False

    # OpenAI-style function/tool definitions. When provided, the model is
    # instructed to emit a JSON block {"tool_calls":[...]} in its response, which
    # we translate back into OpenAI's tool_calls shape.
    tools: Optional[List[Dict[str, Any]]] = None
    # How to handle tool_choice, forwarded as a prompt-level instruction:
    # "none" -> never call a function, "required" -> must call one, or a dict
    # {"type": "function", "function": {"name": ...}} to force a specific one.
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

    # Standard OpenAI Chat Completion parameters (accepted for compatibility).
    stream_options: Optional[StreamOptions] = None
    response_format: Optional[Union[ResponseFormat, Dict[str, Any]]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    seed: Optional[int] = None
