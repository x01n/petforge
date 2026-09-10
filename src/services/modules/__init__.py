"""内置服务模块的统一生命周期管理。"""

from .manager import ModuleManager
from .types import (
    ModuleContext,
    ModuleDescriptor,
    ModuleGenerationGate,
    ModuleState,
    ModuleStatus,
)

__all__ = [
    "ModuleContext",
    "ModuleDescriptor",
    "ModuleGenerationGate",
    "ModuleManager",
    "ModuleState",
    "ModuleStatus",
]
