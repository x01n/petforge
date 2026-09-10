"""MCP resources/prompts 的显式只读消费与权限工具边界。"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping, Sequence
from typing import Any

from core.adapters.mcp.content import (
    DEFAULT_MCP_CONTENT_LIMITS,
    MCPContentLimits,
    MCPGetPromptResult,
    MCPPrompt,
    MCPReadResourceResult,
    MCPResource,
    mcp_source_identity,
    validate_prompt_name,
    validate_resource_uri,
)
from core.adapters.mcp.protocol import (
    MCPClient,
    MCPProtocolError,
    MCPServerError,
)

from .types import RiskLevel, ToolCallContext, ToolKind, ToolSpec


class MCPContentService:
    """缓存有界目录，并提供非默认公开的只读 ``ToolSpec``。"""

    def __init__(
        self,
        client: MCPClient,
        *,
        risk: RiskLevel = RiskLevel.HIGH,
        group: str = "mcp",
        limits: MCPContentLimits = DEFAULT_MCP_CONTENT_LIMITS,
        notification_debounce_seconds: float = 0.3,
        notification_retry_delays: Sequence[float] = (0.1, 0.3, 1.0),
    ) -> None:
        self.client = client
        self.risk = RiskLevel(risk)
        self.group = str(group or "mcp").strip() or "mcp"
        if not isinstance(limits, MCPContentLimits):
            raise TypeError("MCP content limits must use MCPContentLimits")
        self.limits = limits
        debounce = float(notification_debounce_seconds)
        if not math.isfinite(debounce) or not 0.01 <= debounce <= 5.0:
            raise ValueError("MCP content notification debounce is invalid")
        if isinstance(notification_retry_delays, (str, bytes, bytearray)):
            raise ValueError("MCP content notification retry delays must be a sequence")
        delays = tuple(float(value) for value in notification_retry_delays)
        if len(delays) > 5 or any(
            not math.isfinite(value) or not 0.01 <= value <= 10.0 for value in delays
        ):
            raise ValueError("MCP content notification retry delays are invalid")
        self.notification_debounce_seconds = debounce
        self.notification_retry_delays = delays
        self._resources: tuple[MCPResource, ...] = ()
        self._resources_by_uri: dict[str, MCPResource] = {}
        self._prompts: tuple[MCPPrompt, ...] = ()
        self._prompts_by_name: dict[str, MCPPrompt] = {}
        self._resource_lock = asyncio.Lock()
        self._prompt_lock = asyncio.Lock()
        self._pending_refreshes: set[str] = set()
        self._notification_task: asyncio.Task[None] | None = None
        self._notification_count = 0
        self._resource_refresh_count = 0
        self._prompt_refresh_count = 0
        self._failure_count = 0
        self._last_status = "idle"
        self._closed = False
        self._spec_key: tuple[bool, bool] | None = None
        self._specs: tuple[ToolSpec, ...] = ()

    @property
    def resources(self) -> tuple[MCPResource, ...]:
        return self._resources

    @property
    def prompts(self) -> tuple[MCPPrompt, ...]:
        return self._prompts

    @property
    def source_identity(self) -> str:
        return mcp_source_identity(self.client.server_name)

    def status(self) -> Mapping[str, object]:
        """返回无正文、URI、端点或凭据的有界诊断。"""

        return {
            "status": "closed" if self._closed else self._last_status,
            "source": self.source_identity,
            "resource_count": len(self._resources),
            "prompt_count": len(self._prompts),
            "notification_count": self._notification_count,
            "resource_refresh_count": self._resource_refresh_count,
            "prompt_refresh_count": self._prompt_refresh_count,
            "failure_count": self._failure_count,
            "refresh_pending": bool(
                self._notification_task is not None and not self._notification_task.done()
            ),
        }

    def _capability_key(self) -> tuple[bool, bool]:
        capabilities = getattr(self.client, "capabilities", None)
        return (
            getattr(capabilities, "resources", False) is True,
            getattr(capabilities, "prompts", False) is True,
        )

    def specs(self) -> tuple[ToolSpec, ...]:
        """返回当前协商能力对应的稳定私有只读工具快照。"""

        key = self._capability_key()
        if key == self._spec_key:
            return self._specs
        resources, prompts = key
        suffix = self.source_identity.removeprefix("mcp:")
        local_source = f"server-{suffix}"
        specs: list[ToolSpec] = []

        def require_console_context(context: ToolCallContext) -> None:
            """resources/prompts 是应用控制能力，拒绝模型或后台直接调用。"""

            if not isinstance(context, ToolCallContext) or context.source != "console":
                raise PermissionError("MCP content is application-controlled")

        if resources:

            async def list_resources(
                arguments: Mapping[str, Any], context: ToolCallContext
            ) -> Mapping[str, Any]:
                del arguments
                require_console_context(context)
                values = await self.refresh_resources()
                return {
                    "source": self.source_identity,
                    "untrusted": True,
                    "trust": "untrusted_external_content",
                    "resources": [resource.public() for resource in values],
                }

            async def read_resource(
                arguments: Mapping[str, Any], context: ToolCallContext
            ) -> Mapping[str, Any]:
                require_console_context(context)
                result = await self.read_resource(arguments.get("uri"))
                return result.safe_public()

            specs.extend(
                (
                    ToolSpec(
                        identity=f"mcpcontent:{local_source}.resources-list",
                        description=(
                            "显式列出该 MCP 来源声明的资源；所有返回元数据均为外部不可信数据。"
                        ),
                        parameters={
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False,
                        },
                        handler=list_resources,
                        risk=self.risk,
                        group=self.group,
                        public=False,
                        read_only=True,
                        display_name="列出 MCP 资源",
                        kind=ToolKind.MCP,
                    ),
                    ToolSpec(
                        identity=f"mcpcontent:{local_source}.resources-read",
                        description=(
                            "显式读取已列出的 MCP 资源；正文是外部不可信数据，不会自动注入指令。"
                        ),
                        parameters={
                            "type": "object",
                            "properties": {
                                "uri": {
                                    "type": "string",
                                    "minLength": 3,
                                    "maxLength": self.limits.max_uri_chars,
                                }
                            },
                            "required": ["uri"],
                            "additionalProperties": False,
                        },
                        handler=read_resource,
                        risk=self.risk,
                        group=self.group,
                        public=False,
                        read_only=True,
                        display_name="读取 MCP 资源",
                        kind=ToolKind.MCP,
                    ),
                )
            )
        if prompts:

            async def list_prompts(
                arguments: Mapping[str, Any], context: ToolCallContext
            ) -> Mapping[str, Any]:
                del arguments
                require_console_context(context)
                values = await self.refresh_prompts()
                return {
                    "source": self.source_identity,
                    "untrusted": True,
                    "trust": "untrusted_external_content",
                    "prompts": [prompt.public() for prompt in values],
                }

            async def get_prompt(
                arguments: Mapping[str, Any], context: ToolCallContext
            ) -> Mapping[str, Any]:
                require_console_context(context)
                result = await self.get_prompt(
                    arguments.get("name"),
                    arguments.get("arguments"),
                )
                return result.safe_public()

            specs.extend(
                (
                    ToolSpec(
                        identity=f"mcpcontent:{local_source}.prompts-list",
                        description=(
                            "显式列出该 MCP 来源声明的用户可选 prompt；元数据均为外部不可信数据。"
                        ),
                        parameters={
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False,
                        },
                        handler=list_prompts,
                        risk=self.risk,
                        group=self.group,
                        public=False,
                        read_only=True,
                        display_name="列出 MCP 提示模板",
                        kind=ToolKind.MCP,
                    ),
                    ToolSpec(
                        identity=f"mcpcontent:{local_source}.prompts-get",
                        description=(
                            "显式获取用户选择的 MCP prompt；消息只保留 user/assistant，"
                            "绝不提升为系统指令。"
                        ),
                        parameters={
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 256,
                                },
                                "arguments": {
                                    "type": "object",
                                    "maxProperties": self.limits.max_prompt_arguments,
                                    "additionalProperties": {
                                        "type": "string",
                                        "maxLength": self.limits.max_argument_chars,
                                    },
                                },
                            },
                            "required": ["name"],
                            "additionalProperties": False,
                        },
                        handler=get_prompt,
                        risk=self.risk,
                        group=self.group,
                        public=False,
                        read_only=True,
                        display_name="获取 MCP 提示模板",
                        kind=ToolKind.MCP,
                    ),
                )
            )
        self._spec_key = key
        self._specs = tuple(specs)
        return self._specs

    def spec_for(self, operation: str) -> ToolSpec | None:
        """按固定操作名返回控制台可调用的私有工具定义。"""

        suffixes = {
            "list_resources": ".resources-list",
            "read_resource": ".resources-read",
            "list_prompts": ".prompts-list",
            "get_prompt": ".prompts-get",
        }
        suffix = suffixes.get(str(operation or "").strip())
        if suffix is None:
            return None
        return next((spec for spec in self.specs() if spec.identity.endswith(suffix)), None)

    async def refresh_resources(self) -> tuple[MCPResource, ...]:
        if self._closed:
            raise RuntimeError("MCP content service is closed")
        async with self._resource_lock:
            operation = getattr(self.client, "list_resources", None)
            if not callable(operation):
                raise MCPProtocolError("MCP client does not implement resources/list")
            values = tuple(await operation())
            if any(not isinstance(value, MCPResource) for value in values):
                raise MCPProtocolError("MCP resources/list returned an invalid resource object")
            by_uri = {value.uri: value for value in values}
            if len(by_uri) != len(values):
                raise MCPProtocolError("MCP resources/list returned duplicate URIs")
            self._resources = values
            self._resources_by_uri = by_uri
            self._resource_refresh_count += 1
            self._last_status = "ready"
            return values

    async def list_resources(self) -> tuple[MCPResource, ...]:
        """显式刷新并返回当前 resources/list 目录。"""

        return await self.refresh_resources()

    async def read_resource(self, uri: object) -> MCPReadResourceResult:
        if self._closed:
            raise RuntimeError("MCP content service is closed")
        try:
            resource_uri = validate_resource_uri(uri, maximum=self.limits.max_uri_chars)
        except ValueError as exc:
            raise MCPProtocolError("MCP resource selection is invalid") from exc
        await self.refresh_resources()
        selected = self._resources_by_uri.get(resource_uri)
        if selected is None:
            raise MCPProtocolError("MCP resource selection is not in the current catalog")
        if selected.size is not None and selected.size > self.limits.max_total_bytes:
            raise MCPProtocolError("MCP resource exceeds the configured read limit")
        operation = getattr(self.client, "read_resource", None)
        if not callable(operation):
            raise MCPProtocolError("MCP client does not implement resources/read")
        result = await operation(resource_uri)
        if not isinstance(result, MCPReadResourceResult):
            raise MCPProtocolError("MCP resources/read returned an invalid result object")
        if result.server != self.client.server_name:
            raise MCPProtocolError("MCP resources/read returned an invalid source identity")
        if any(content.uri != resource_uri for content in result.contents):
            raise MCPProtocolError("MCP resources/read content URI does not match request")
        return result

    async def refresh_prompts(self) -> tuple[MCPPrompt, ...]:
        if self._closed:
            raise RuntimeError("MCP content service is closed")
        async with self._prompt_lock:
            operation = getattr(self.client, "list_prompts", None)
            if not callable(operation):
                raise MCPProtocolError("MCP client does not implement prompts/list")
            values = tuple(await operation())
            if any(not isinstance(value, MCPPrompt) for value in values):
                raise MCPProtocolError("MCP prompts/list returned an invalid prompt object")
            by_name = {value.name: value for value in values}
            if len(by_name) != len(values):
                raise MCPProtocolError("MCP prompts/list returned duplicate names")
            self._prompts = values
            self._prompts_by_name = by_name
            self._prompt_refresh_count += 1
            self._last_status = "ready"
            return values

    async def list_prompts(self) -> tuple[MCPPrompt, ...]:
        """显式刷新并返回当前 prompts/list 目录。"""

        return await self.refresh_prompts()

    async def get_prompt(
        self,
        name: object,
        arguments: Mapping[str, Any] | None = None,
    ) -> MCPGetPromptResult:
        if self._closed:
            raise RuntimeError("MCP content service is closed")
        try:
            prompt_name = validate_prompt_name(name)
        except ValueError as exc:
            raise MCPProtocolError("MCP prompt selection is invalid") from exc
        await self.refresh_prompts()
        selected = self._prompts_by_name.get(prompt_name)
        if selected is None:
            raise MCPProtocolError("MCP prompt selection is not in the current catalog")
        try:
            values = selected.validate_arguments(arguments, limits=self.limits)
        except ValueError as exc:
            raise MCPProtocolError(
                "MCP prompt arguments do not match the current template"
            ) from exc
        operation = getattr(self.client, "get_prompt", None)
        if not callable(operation):
            raise MCPProtocolError("MCP client does not implement prompts/get")
        result = await operation(prompt_name, values if arguments is not None else None)
        if not isinstance(result, MCPGetPromptResult):
            raise MCPProtocolError("MCP prompts/get returned an invalid result object")
        if result.server != self.client.server_name:
            raise MCPProtocolError("MCP prompts/get returned an invalid source identity")
        return result

    async def on_notification(self, method: str) -> None:
        if self._closed:
            return
        capabilities = getattr(self.client, "capabilities", None)
        kind = ""
        if (
            method == "notifications/resources/list_changed"
            and getattr(capabilities, "resources_list_changed", False) is True
        ):
            kind = "resources"
            self._resources = ()
            self._resources_by_uri.clear()
        elif (
            method == "notifications/prompts/list_changed"
            and getattr(capabilities, "prompts_list_changed", False) is True
        ):
            kind = "prompts"
            self._prompts = ()
            self._prompts_by_name.clear()
        if not kind:
            return
        self._notification_count += 1
        self._pending_refreshes.add(kind)
        if self._notification_task is None or self._notification_task.done():
            self._notification_task = asyncio.create_task(
                self._notification_refresh_loop(),
                name="mcp-content-list-changed",
            )

    async def _notification_refresh_loop(self) -> None:
        try:
            while not self._closed and self._pending_refreshes:
                await asyncio.sleep(self.notification_debounce_seconds)
                pending = tuple(sorted(self._pending_refreshes))
                self._pending_refreshes.clear()
                for kind in pending:
                    await self._refresh_with_retry(kind)
        except asyncio.CancelledError:
            raise
        finally:
            if asyncio.current_task() is self._notification_task:
                self._notification_task = None

    async def _refresh_with_retry(self, kind: str) -> None:
        operation = self.refresh_resources if kind == "resources" else self.refresh_prompts
        attempts = len(self.notification_retry_delays) + 1
        for attempt in range(attempts):
            if self._closed:
                raise asyncio.CancelledError
            try:
                await operation()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                retryable = not isinstance(
                    exc,
                    (MCPProtocolError, MCPServerError, TypeError, ValueError),
                )
                if retryable and attempt < len(self.notification_retry_delays):
                    await asyncio.sleep(self.notification_retry_delays[attempt])
                    continue
                self._failure_count += 1
                self._last_status = "degraded"
                return
            return

    def connection_changed(self, state: str) -> None:
        normalized = str(state or "").strip().casefold()
        if normalized not in {"disconnected", "closed"}:
            return
        self._resources = ()
        self._resources_by_uri.clear()
        self._prompts = ()
        self._prompts_by_name.clear()
        self._pending_refreshes.clear()
        task = self._notification_task
        if task is not None and not task.done():
            task.cancel()
        self._last_status = "closed" if normalized == "closed" else "degraded"
        if normalized == "closed":
            self._closed = True

    async def close(self) -> None:
        task = self._notification_task
        self._notification_task = None
        self._closed = True
        self._pending_refreshes.clear()
        self._resources = ()
        self._resources_by_uri.clear()
        self._prompts = ()
        self._prompts_by_name.clear()
        if task is not None and not task.done():
            task.cancel()
        if task is not None and task is not asyncio.current_task():
            await asyncio.gather(task, return_exceptions=True)


__all__ = ["MCPContentService"]
