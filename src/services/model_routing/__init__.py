"""模型渠道配置、路由和显式重试策略。"""

from .channels import (
    SUPPORTED_PROTOCOLS,
    ChannelConfig,
    channel_from_mapping,
    load_channels,
    select_channels,
    sort_channels,
)
from .retry import RetryPolicy
from .router import ModelRouter, ModelRouterService, ModelRoutingError, RouteSelection, RouteSpec
from .vision import (
    VisionSummaryPolicy,
    build_vision_request,
    request_has_inline_images,
    strip_images_and_attach_summary,
    validate_inline_images,
)

__all__ = [
    "SUPPORTED_PROTOCOLS",
    "ChannelConfig",
    "ModelRouter",
    "ModelRouterService",
    "ModelRoutingError",
    "RetryPolicy",
    "RouteSelection",
    "RouteSpec",
    "channel_from_mapping",
    "load_channels",
    "select_channels",
    "sort_channels",
    "VisionSummaryPolicy",
    "build_vision_request",
    "request_has_inline_images",
    "strip_images_and_attach_summary",
    "validate_inline_images",
]
