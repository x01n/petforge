"""桌宠对话编排和展示状态。"""

from .interaction_contract import (
    InteractionAction,
    InteractionActionKind,
    InteractionPhase,
    InteractionSnapshot,
    InteractionState,
    PetFeedback,
    model_readiness,
    pet_feedback,
)
from .presentation import BubbleSnapshot, PresentationService
from .service import ConversationResult, ConversationService, EventSink

__all__ = [
    "BubbleSnapshot",
    "InteractionAction",
    "InteractionActionKind",
    "InteractionPhase",
    "InteractionSnapshot",
    "InteractionState",
    "PetFeedback",
    "ConversationResult",
    "ConversationService",
    "EventSink",
    "PresentationService",
    "model_readiness",
    "pet_feedback",
]
