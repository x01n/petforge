"""Qt6/OpenGL 宿主。"""

from .audio import AudioOutputDeviceChoice, PlaybackReceipt, QtAudioPlayer
from .config_panel import ConfigurationPanel
from .console import PetConsoleWindow
from .dispatcher import QtMainThreadDispatcher
from .display_size import DISPLAY_SIZE_LABELS, DISPLAY_SIZE_PRESETS
from .fancyui import (
    FANCY_ICON_NAMES,
    FancyCard,
    FancyCommandBar,
    FancyInfoBar,
    FancyNavigationView,
    FancyStyleController,
    build_fancy_stylesheet,
    fancy_icon,
    pyside6_fancy_available,
)
from .hotkeys import HotkeyBinding, HotkeyConfig, HotkeyManager, HotkeyMode
from .microphone import CapturedPcm, QtMicrophoneCapture, QtPushToTalkController
from .model_setup import (
    ADAPTER_PROTOCOLS,
    API_KEY_ENV_ALIASES,
    PROTOCOL_LABELS,
    ModelSetupCard,
    ModelSetupDialog,
    ModelSetupDraft,
    adapter_for_protocol,
    model_setup_drafts,
    openai_base_url_hint,
    preferred_api_key_env,
    protocols_for_adapter,
)
from .opengl_host import PetOpenGLWindow, pyside6_available
from .vulkan_host import (
    VulkanPetHost,
    create_vulkan_host,
    prepare_vulkan_scenegraph,
    pyside6_vulkan_available,
)
from .web_console import WebControlSurfaceWindow
from .web_host import WebPetHost

__all__ = [
    "PetOpenGLWindow",
    "PetConsoleWindow",
    "DISPLAY_SIZE_LABELS",
    "DISPLAY_SIZE_PRESETS",
    "ConfigurationPanel",
    "QtAudioPlayer",
    "AudioOutputDeviceChoice",
    "PlaybackReceipt",
    "CapturedPcm",
    "QtMicrophoneCapture",
    "QtPushToTalkController",
    "QtMainThreadDispatcher",
    "FANCY_ICON_NAMES",
    "FancyCard",
    "FancyCommandBar",
    "FancyInfoBar",
    "FancyNavigationView",
    "FancyStyleController",
    "build_fancy_stylesheet",
    "fancy_icon",
    "pyside6_fancy_available",
    "HotkeyBinding",
    "HotkeyConfig",
    "HotkeyManager",
    "HotkeyMode",
    "ModelSetupCard",
    "ModelSetupDialog",
    "ModelSetupDraft",
    "model_setup_drafts",
    "ADAPTER_PROTOCOLS",
    "API_KEY_ENV_ALIASES",
    "adapter_for_protocol",
    "protocols_for_adapter",
    "openai_base_url_hint",
    "preferred_api_key_env",
    "PROTOCOL_LABELS",
    "WebPetHost",
    "VulkanPetHost",
    "create_vulkan_host",
    "prepare_vulkan_scenegraph",
    "pyside6_vulkan_available",
    "WebControlSurfaceWindow",
    "pyside6_available",
]
