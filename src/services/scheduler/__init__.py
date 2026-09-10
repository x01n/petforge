"""定时任务、事件触发器和受权限约束的随机行为。"""

from .activity import (
    MAX_IDLE_SECONDS,
    MAX_PENDING_INTERACTIONS,
    MAX_SYSTEM_IDLE_THRESHOLD_SECONDS,
    MIN_IDLE_SECONDS,
    MIN_SYSTEM_IDLE_THRESHOLD_SECONDS,
    SYSTEM_IDLE_PROVIDER_DISABLED,
    SYSTEM_IDLE_PROVIDER_WINDOWS,
    SYSTEM_IDLE_PROVIDER_X11,
    SYSTEM_IDLE_PROVIDERS,
    UserActivityTracker,
)
from .activity import (
    MAX_POLL_SECONDS as ACTIVITY_MAX_POLL_SECONDS,
)
from .activity import (
    MIN_POLL_SECONDS as ACTIVITY_MIN_POLL_SECONDS,
)
from .behavior import AutonomousMovementConfig, BehaviorAction, BehaviorPhase, BehaviorService
from .persistence import SchedulerStateStore
from .scheduler import (
    ScheduledTask,
    ScheduleExpressionError,
    SchedulerService,
    next_schedule_at,
    parse_interval,
    validate_schedule_expression,
)
from .triggers import (
    Trigger,
    TriggerService,
    normalize_trigger_conditions,
    trigger_conditions_match,
)
from .watcher import DesktopWindowWatcher, WindowWatcher

__all__ = [
    "BehaviorAction",
    "BehaviorPhase",
    "AutonomousMovementConfig",
    "BehaviorService",
    "UserActivityTracker",
    "MIN_IDLE_SECONDS",
    "MAX_IDLE_SECONDS",
    "ACTIVITY_MIN_POLL_SECONDS",
    "ACTIVITY_MAX_POLL_SECONDS",
    "MAX_PENDING_INTERACTIONS",
    "MAX_SYSTEM_IDLE_THRESHOLD_SECONDS",
    "MIN_SYSTEM_IDLE_THRESHOLD_SECONDS",
    "SYSTEM_IDLE_PROVIDER_DISABLED",
    "SYSTEM_IDLE_PROVIDER_X11",
    "SYSTEM_IDLE_PROVIDER_WINDOWS",
    "SYSTEM_IDLE_PROVIDERS",
    "SchedulerStateStore",
    "ScheduleExpressionError",
    "ScheduledTask",
    "SchedulerService",
    "Trigger",
    "TriggerService",
    "normalize_trigger_conditions",
    "trigger_conditions_match",
    "DesktopWindowWatcher",
    "WindowWatcher",
    "next_schedule_at",
    "parse_interval",
    "validate_schedule_expression",
]
