"""桌宠插件发现与生命周期服务。"""

from .manager import PluginManager
from .types import (
    PluginContext,
    PluginDescriptor,
    PluginFactory,
    PluginFactoryLike,
    PluginLifecycle,
    PluginSettings,
    PluginState,
    PluginStatus,
)

__all__ = [
    "PluginContext",
    "PluginDescriptor",
    "PluginFactory",
    "PluginFactoryLike",
    "PluginLifecycle",
    "PluginManager",
    "PluginSettings",
    "PluginState",
    "PluginStatus",
]
