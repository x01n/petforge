from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from core.contracts.chat import ChatRequest, ToolDefinition

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]")
_MAX_PROVIDER_NAME = 64


def _encode_character(match: re.Match[str]) -> str:
    return f"_x{ord(match.group(0)):02x}_"


@dataclass(frozen=True)
class ProviderToolNames:
    to_provider: Mapping[str, str]
    to_core: Mapping[str, str]

    def provider(self, identity: str) -> str:
        return str(self.to_provider.get(identity, identity))

    def core(self, provider_name: str) -> str:
        value = str(provider_name or "").strip()
        return str(self.to_core.get(value, value if ":" in value else f"remote:{value}"))


def build_tool_name_map(tools: Sequence[ToolDefinition]) -> ProviderToolNames:

    forward: dict[str, str] = {}
    reverse: dict[str, str] = {}
    for tool in tools:
        identity = str(tool.identity).strip()
        encoded = _SAFE_NAME.sub(_encode_character, identity)
        if not encoded or not encoded[0].isalpha():
            encoded = f"tool_{encoded}"
        # 预留短哈希以处理截断或罕见编码碰撞，核心身份仍由映射恢复。
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
        base_limit = _MAX_PROVIDER_NAME - len(digest) - 2
        provider_name = f"{encoded[:base_limit]}__{digest}"
        # 同一请求中即使截断后相同，也必须分配不同名称。
        if provider_name in reverse and reverse[provider_name] != identity:
            provider_name = f"t_{digest}_{len(reverse)}"
        forward[identity] = provider_name
        reverse[provider_name] = identity
    return ProviderToolNames(forward, reverse)


def _json_arguments(arguments: Mapping[str, Any]) -> str:
    return json.dumps(dict(arguments), ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _content_parts(content: object) -> list[Mapping[str, object]]:
    if isinstance(content, Mapping):
        return [dict(content)]
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        return [dict(item) for item in content if isinstance(item, Mapping)]
    return [{"type": "text", "text": str(content or "")}]


def _image_part(part: Mapping[str, object], *, provider: str) -> Mapping[str, object] | None:
    """转换核心内联图片；核心契约不允许适配器主动抓取远端 URL。"""

    if str(part.get("type", "")).strip().lower() != "image":
        return None
    media_type = str(part.get("media_type", "")).strip().lower()
    data = str(part.get("data", "")).strip()
    if not media_type or not data:
        return None
    if provider == "gemini":
        return {"inlineData": {"mimeType": media_type, "data": data}}
    if provider == "claude":
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}


def _text_parts(content: object) -> list[str]:
    result: list[str] = []
    for part in _content_parts(content):
        if str(part.get("type", "text")).strip().lower() == "text":
            value = part.get("text")
            if value is not None:
                result.append(str(value))
    if not result and isinstance(content, str):
        result.append(content)
    return result


def _tool_response_value(content: object) -> Mapping[str, object]:
    """把核心工具消息还原为 Gemini functionResponse 所需的对象。"""

    parts = _content_parts(content)
    if len(parts) == 1 and str(parts[0].get("type", "")).strip().lower() == "text":
        text = parts[0].get("text")
        if isinstance(text, str):
            try:
                decoded = json.loads(text)
            except (TypeError, ValueError, json.JSONDecodeError):
                decoded = None
            if isinstance(decoded, Mapping):
                return dict(decoded)
            return {"text": text}
    return dict(parts[0]) if parts else {}


def _tool_definitions(
    tools: Sequence[ToolDefinition], names: ProviderToolNames, provider: str
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for tool in tools:
        name = names.provider(tool.identity)
        description = str(tool.description or "")
        parameters = dict(tool.parameters)
        if provider == "claude":
            result.append({"name": name, "description": description, "input_schema": parameters})
        elif provider == "gemini":
            result.append({"name": name, "description": description, "parameters": parameters})
        else:
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": parameters,
                    },
                }
            )
    return result


def openai_messages(request: ChatRequest, names: ProviderToolNames) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in request.messages:
        payload: dict[str, Any] = {"role": message.role, "content": message.content}
        if isinstance(message.content, Mapping) or (
            isinstance(message.content, Sequence)
            and not isinstance(message.content, (str, bytes, bytearray))
        ):
            payload["content"] = [dict(part) for part in _content_parts(message.content)]
            for part in payload["content"]:
                image = _image_part(part, provider="openai")
                if image is not None:
                    part.clear()
                    part.update(image)
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {
                        "name": names.provider(call.identity),
                        "arguments": _json_arguments(call.arguments),
                    },
                }
                for call in message.tool_calls
            ]
        if message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
        if message.tool_name:
            payload["name"] = names.provider(message.tool_name)
        result.append(payload)
    return result


def _responses_content_parts(content: object, *, role: str) -> object:
    """把核心内容转换成 Responses input message 的内容项。"""

    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping) and "type" not in content:
        return json.dumps(dict(content), ensure_ascii=False, allow_nan=False)
    parts: list[dict[str, Any]] = []
    text_type = "output_text" if role == "assistant" else "input_text"
    for part in _content_parts(content):
        image = _image_part(part, provider="openai")
        if image is not None:
            image_url = image.get("image_url")
            if isinstance(image_url, Mapping) and image_url.get("url"):
                parts.append({"type": "input_image", "image_url": str(image_url["url"])})
            continue
        part_type = str(part.get("type", "text")).strip().lower()
        if part_type == "text":
            parts.append({"type": text_type, "text": str(part.get("text", ""))})
            continue
        if part_type == "image_url":
            image_url = part.get("image_url")
            if isinstance(image_url, Mapping):
                image_url = image_url.get("url", "")
            if image_url:
                parts.append({"type": "input_image", "image_url": str(image_url)})
    return parts


def _responses_tool_output(content: object) -> str | list[dict[str, Any]]:
    """把工具结果转换成 Responses function_call_output 的 output。"""

    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        return json.dumps(dict(content), ensure_ascii=False, allow_nan=False)
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        parts = _responses_content_parts(content, role="user")
        if isinstance(parts, list):
            return parts
    return str(content or "")


def openai_responses_input(request: ChatRequest, names: ProviderToolNames) -> list[dict[str, Any]]:
    """把核心消息转换成 OpenAI Responses input item。"""

    call_ids_by_name: dict[str, str] = {}
    for message in request.messages:
        for call in message.tool_calls:
            call_ids_by_name.setdefault(call.identity, call.call_id)

    result: list[dict[str, Any]] = []
    for message in request.messages:
        if message.role == "tool":
            call_id = message.tool_call_id or call_ids_by_name.get(message.tool_name, "")
            if not call_id:
                raise ValueError("Responses tool messages require a matching tool_call_id")
            result.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": _responses_tool_output(message.content),
                }
            )
            continue

        content = _responses_content_parts(message.content, role=message.role)
        if content is not None:
            result.append({"role": message.role, "content": content})
        for call in message.tool_calls:
            result.append(
                {
                    "type": "function_call",
                    "call_id": call.call_id,
                    "name": names.provider(call.identity),
                    "arguments": _json_arguments(call.arguments),
                }
            )
    return result


def gemini_contents(
    request: ChatRequest, names: ProviderToolNames
) -> tuple[str, list[dict[str, Any]]]:
    system: list[str] = []
    contents: list[dict[str, Any]] = []
    call_names: dict[str, str] = {}
    for message in request.messages:
        for call in message.tool_calls:
            call_names.setdefault(call.call_id, call.identity)
    for message in request.messages:
        if message.role == "system":
            system.extend(_text_parts(message.content))
            continue
        role = "model" if message.role == "assistant" else "user"
        parts: list[dict[str, Any]] = []
        if message.content is not None:
            for part in _content_parts(message.content):
                image = _image_part(part, provider="gemini")
                if image is not None:
                    parts.append(dict(image))
                elif str(part.get("type", "text")).strip().lower() == "text":
                    parts.append({"text": str(part.get("text", ""))})
        for call in message.tool_calls:
            function_call = {
                "name": names.provider(call.identity),
                "args": dict(call.arguments),
            }
            if call.call_id:
                function_call["id"] = call.call_id
            parts.append({"functionCall": function_call})
        if message.role == "tool":
            core_name = message.tool_name or call_names.get(message.tool_call_id, "")
            if not core_name:
                raise ValueError("Gemini tool messages require a matching tool identity")
            tool_name = names.provider(core_name)
            function_response: dict[str, Any] = {
                "name": tool_name,
                "response": _tool_response_value(message.content),
            }
            if message.tool_call_id:
                function_response["id"] = message.tool_call_id
            parts = [{"functionResponse": function_response}]
        if not parts:
            parts = [{"text": ""}]
        contents.append({"role": role, "parts": parts})
    return "\n\n".join(item for item in system if item), contents


def claude_messages(
    request: ChatRequest, names: ProviderToolNames
) -> tuple[str, list[dict[str, Any]]]:
    system: list[str] = []
    messages: list[dict[str, Any]] = []
    call_ids_by_name: dict[str, str] = {}
    for message in request.messages:
        for call in message.tool_calls:
            call_ids_by_name.setdefault(call.identity, call.call_id)
    for message in request.messages:
        if message.role in {"system", "developer"}:
            system.extend(_text_parts(message.content))
            continue
        if message.role == "tool":
            result_id = message.tool_call_id or call_ids_by_name.get(message.tool_name, "")
            if not result_id:
                raise ValueError("Claude tool messages require a matching tool_call_id")
            if isinstance(message.content, str):
                result_content: object = message.content
            elif isinstance(message.content, Mapping) and "type" not in message.content:
                result_content = json.dumps(
                    dict(message.content), ensure_ascii=False, allow_nan=False
                )
            else:
                result_blocks: list[dict[str, Any]] = []
                for part in _content_parts(message.content):
                    image = _image_part(part, provider="claude")
                    if image is not None:
                        result_blocks.append(dict(image))
                    elif str(part.get("type", "text")).strip().lower() == "text":
                        result_blocks.append({"type": "text", "text": str(part.get("text", ""))})
                result_content = result_blocks
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": result_id,
                            "content": result_content,
                        }
                    ],
                }
            )
            continue
        role = "assistant" if message.role == "assistant" else "user"
        blocks: list[dict[str, Any]] = []
        if message.content is not None:
            for part in _content_parts(message.content):
                image = _image_part(part, provider="claude")
                if image is not None:
                    blocks.append(dict(image))
                elif str(part.get("type", "text")).strip().lower() == "text":
                    blocks.append({"type": "text", "text": str(part.get("text", ""))})
        for call in message.tool_calls:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call.call_id,
                    "name": names.provider(call.identity),
                    "input": dict(call.arguments),
                }
            )
        if not blocks:
            blocks = [{"type": "text", "text": ""}]
        messages.append({"role": role, "content": blocks})
    return "\n\n".join(item for item in system if item), messages


def tool_choice(
    value: str | None, names: ProviderToolNames, provider: str
) -> dict[str, Any] | str | None:
    choice = None if value is None else str(value).strip().lower()
    if provider == "gemini":
        mode = {None: "AUTO", "auto": "AUTO", "required": "ANY", "none": "NONE"}.get(choice)
        return {"functionCallingConfig": {"mode": mode}} if mode else None
    if provider == "claude":
        return (
            {"type": {"auto": "auto", "required": "any"}.get(choice, "auto")}
            if choice != "none"
            else None
        )
    return choice or "auto"


def provider_tool_definitions(
    request: ChatRequest, names: ProviderToolNames, provider: str
) -> list[dict[str, Any]]:
    return _tool_definitions(request.tools, names, provider)


__all__ = [
    "ProviderToolNames",
    "build_tool_name_map",
    "claude_messages",
    "gemini_contents",
    "openai_messages",
    "openai_responses_input",
    "provider_tool_definitions",
    "tool_choice",
]
