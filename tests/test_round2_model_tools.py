"""第 2 轮 A 模块：换模工具与宿主换模入口的定向测试。

覆盖：
- ``register_builtin_tools`` 新增的 ``pet:list_models`` / ``pet:switch_model``
  的身份、可见性、参数校验与脱敏回执；
- 非法 ``model_key``（控制字符、超长、绝对路径、``..`` 逃逸）必须被拒绝；
- ``MutablePetController.switch_live2d_model`` 对相对路径逃逸、资源根缺失、
  空清单的目标模型拒绝为 unavailable；
- 空目录资源根上，``available_models`` 返回空元组不抛异常。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from app.runtime import MutablePetController
from config.resources import inspect_resources
from services.tools.builtins import register_builtin_tools
from services.tools.executor import ToolExecutionService
from services.tools.permissions import PermissionService
from services.tools.registry import ToolRegistry
from services.tools.types import RiskLevel, ToolCallContext

_PROFILE = "round2-profile"
_SESSION = "round2-session"


def _models_file(root: Path, relative: str, content: dict[str, Any]) -> Path:
    """在资源根下写一个通过 model3 引用校验的描述文件。"""

    descriptor = root / relative
    descriptor.parent.mkdir(parents=True, exist_ok=True)
    references = content.get("FileReferences", {})
    for name in (references.get("Moc"), *references.get("Textures", ())):
        if isinstance(name, str) and name:
            path = descriptor.parent / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x00" * 8)
    descriptor.write_text(json.dumps(content), encoding="utf-8")
    return descriptor


def _model3_payload() -> dict[str, Any]:
    """返回可直接写入描述文件的合法 model3 JSON。"""

    return {
        "Version": 3,
        "FileReferences": {
            "Moc": "model.moc3",
            "Textures": ["texture.webp"],
        },
    }


def _register(pet_controller: object) -> ToolRegistry:
    """用指定的假控制器注册全部内置工具。"""

    registry = ToolRegistry()
    register_builtin_tools(
        registry,
        platform=SimpleNamespace(),
        pet_controller=pet_controller,
    )
    return registry


def _context() -> ToolCallContext:
    return ToolCallContext(_PROFILE, _SESSION, "round2-model-tools-turn")


def _outcome(content: Mapping[str, Any]) -> dict[str, Any]:
    """把处理器返回值读成字典，兼容可序列化快照。"""

    return dict(content)


def test_round2_model_tools_identity_and_visibility() -> None:
    """两个工具必须注册进 pet_control 组且对模型可见。"""

    class FakePetController:
        def available_models(self) -> tuple[str, ...]:
            return ("live2d/model/model_a.model3.json",)

        def switch_live2d_model(self, model_key: str) -> dict[str, Any]:
            assert model_key == "live2d/model/model_a.model3.json"
            return {"status": "available", "model_key": model_key}

    registry = _register(FakePetController())
    assert "pet:list_models" in registry.identities()
    assert "pet:switch_model" in registry.identities()

    list_spec = registry.require("pet:list_models")
    assert list_spec.group == "pet_control"
    assert list_spec.public is True
    assert list_spec.read_only is True
    assert list_spec.risk is RiskLevel.LOW

    switch_spec = registry.require("pet:switch_model")
    assert switch_spec.group == "pet_control"
    assert switch_spec.public is True
    assert switch_spec.risk is RiskLevel.MEDIUM
    assert switch_spec.read_only is False

    names = {row["function"]["name"] for row in registry.schemas()}
    assert "pet:list_models" in names
    assert "pet:switch_model" in names


def test_round2_list_models_returns_relative_keys_and_count() -> None:
    """list 工具只回显资源相对键，并用 count 汇总数量。"""

    class FakePetController:
        def available_models(self) -> tuple[str, ...]:
            return (
                "live2d/model/model_b.model3.json",
                "live2d/model/model_a.model3.json",
                "",
            )

    registry = _register(FakePetController())
    outcome = asyncio.run(
        ToolExecutionService(
            registry,
            PermissionService(auto_allow_low_risk=True),
        ).execute(
            call_id="list-models",
            identity="pet:list_models",
            arguments={},
            context=_context(),
        )
    )

    assert outcome.status == "completed"
    assert outcome.content["status"] == "completed"
    assert outcome.content["count"] == 2
    assert outcome.content["models"] == [
        "live2d/model/model_b.model3.json",
        "live2d/model/model_a.model3.json",
    ]
    assert "count" in outcome.content and outcome.content["count"] == len(outcome.content["models"])


def test_round2_switch_model_invalid_key_is_rejected_before_controller() -> None:
    """空值与超长键由参数 schema 拒绝；控制字符键由处理器拒绝。"""

    class FakePetController:
        def __init__(self) -> None:
            self.called: list[str] = []

        def switch_live2d_model(self, model_key: str) -> dict[str, Any]:
            self.called.append(model_key)
            return {"status": "available"}

    controller = FakePetController()
    registry = _register(controller)
    executor = ToolExecutionService(
        registry,
        PermissionService(bypass_approval=True),
    )

    async def scenario() -> tuple[list[str], list[str]]:
        messages: list[str] = []
        for index, target in enumerate(("", "a" * 513)):
            outcome = await executor.execute(
                call_id=f"schema-rejected-{index}",
                identity="pet:switch_model",
                arguments={"model_key": target},
                context=_context(),
            )
            assert outcome.status == "denied"
            assert "error" in outcome.content
            assert "status" not in outcome.content
        outcome = await executor.execute(
            call_id="handler-rejected",
            identity="pet:switch_model",
            arguments={"model_key": "live2d/model/model_a.model3.json\x00tail"},
            context=_context(),
        )
        assert outcome.status == "failed"
        messages.append(str(outcome.content.get("status", "")))
        return messages, list(controller.called)

    messages, called = asyncio.run(scenario())
    assert messages == ["failed"]
    assert called == []


def test_round2_switch_model_receipt_never_contains_absolute_paths() -> None:
    """switch 工具必须剥离宿主回执中的绝对路径，只保留资源键与状态。"""

    class FakePetController:
        def switch_live2d_model(self, model_key: str) -> dict[str, Any]:
            return {
                "status": "available",
                "model_key": model_key,
                "model": "/home/user/resources/live2d/model/model_a.model3.json",
                "persistence_status": "pending",
            }

    registry = _register(FakePetController())
    outcome = asyncio.run(
        ToolExecutionService(
            registry,
            PermissionService(bypass_approval=True),
        ).execute(
            call_id="switch-receipt",
            identity="pet:switch_model",
            arguments={"model_key": "live2d/model/model_a.model3.json"},
            context=_context(),
        )
    )

    assert outcome.status == "completed"
    content = _outcome(outcome.content)
    assert content["status"] == "available"
    assert content["model_key"] == "live2d/model/model_a.model3.json"
    assert "model" not in content
    assert content["persistence_status"] == "pending"
    assert "/home/" not in json.dumps(content, ensure_ascii=False)


def test_round2_runtime_inventory_drives_available_models(tmp_path: Path) -> None:
    """真实 ResourceInventory 让 list 工具按资源相对键列出模型。"""

    root = tmp_path / "resources"
    _models_file(root, "live2d/model/model_a.model3.json", _model3_payload())
    _models_file(root, "live2d/model/model_b.model3.json", _model3_payload())

    class PetHost:
        def switch_live2d_model(self, *, model_key: str | None = None) -> dict[str, Any]:
            return {
                "status": "available",
                "model_key": model_key or "",
            }

    controller = MutablePetController(PetHost())
    controller.inventory = inspect_resources(root)
    assert controller.available_models() == (
        "live2d/model/model_a.model3.json",
        "live2d/model/model_b.model3.json",
    )


def test_round2_switch_model_relative_escape_is_rejected(tmp_path: Path) -> None:
    """相对路径逃逸、绝对路径与空清单目标都必须被拒绝为 unavailable。"""

    root = tmp_path / "resources"
    _models_file(root, "live2d/model/model_a.model3.json", _model3_payload())

    class PetHost:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def switch_live2d_model(self, *, model_key: str | None = None) -> dict[str, Any]:
            self.calls.append(model_key or "")
            return {"status": "available"}

    host = PetHost()

    controller = MutablePetController(host)
    controller.inventory = inspect_resources(root)

    for target in ("../secret.model3.json", "/etc/secret.model3.json"):
        result = controller.switch_live2d_model(target)
        assert result["status"] == "unavailable"
        assert "model_key" not in result or result.get("model_key") != target

    result = controller.switch_live2d_model("live2d/model/model_a.model3.json")
    assert result["status"] == "available"
    assert host.calls == ["live2d/model/model_a.model3.json"]

    empty = MutablePetController(PetHost())
    empty.inventory = inspect_resources(tmp_path / "empty-root")
    assert empty.switch_live2d_model("live2d/model/model_a.model3.json")["status"] == (
        "unavailable"
    )
