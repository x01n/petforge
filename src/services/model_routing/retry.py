"""模型路由对外导出的重试策略。

实现位于核心直连适配器包，确保 ``ChannelConfig`` 和运行时使用同一个
``RetryPolicy`` 类型，避免跨包实例检查产生分叉。
"""

from core.adapters.direct.retry import RetryPolicy

__all__ = ["RetryPolicy"]
