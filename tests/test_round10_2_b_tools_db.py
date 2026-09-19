"""第 10 轮方向 2：工具校验、命令参数边界与迁移事务回滚的覆盖补漏。

覆盖对象：
- ``src/services/tools/validation.py``：oneOf/anyOf/not/required 与
  string/integer/number/boolean/enum 的拒绝分支；
- ``src/services/tools/command.py``：白名单注入拒绝与超时钳制边界；
- ``src/db/database.py`` 的 ``SchemaMigrator._atomic``：损坏迁移失败时
  不留下半初始化状态，异常向上传播且后续幂等迁移可重开。

全部为纯内存单元测试，无网络、无模型、无 GUI；数据库用例仅使用
tmp_path 临时文件。
"""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from services.tools.command import (
    normalize_command_allowlist,
    normalize_command_argv,
    normalize_command_timeout,
)
from services.tools.validation import validate_arguments

# 直连加载 database.py，避免 db/__init__.py 的包初始化把本模块耦合到
# 与本用例无关的仓储模块。
_database_source = Path(__file__).resolve().parents[1] / "src" / "db" / "database.py"
_database_spec = importlib.util.spec_from_file_location("meapet_database_direct", _database_source)
assert _database_spec is not None and _database_spec.loader is not None
_database_module = importlib.util.module_from_spec(_database_spec)
_database_spec.loader.exec_module(_database_module)
SCHEMA_VERSION: int = _database_module.SCHEMA_VERSION
SchemaMigrator = _database_module.SchemaMigrator


def _errors(schema: dict[str, object], arguments: object) -> tuple[str, ...]:
    return tuple(f"{item.path}: {item.message}" for item in validate_arguments(schema, arguments))


def test_arguments_required_property_and_string_length_boundaries() -> None:
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 2, "maxLength": 4},
        },
        "required": ["name"],
        "additionalProperties": False,
    }
    assert not validate_arguments(schema, {"name": "abc"})
    errors = _errors(schema, {"name": "x", "extra": 1})
    assert "$.name: is too short" in errors
    assert "$.extra: is not allowed" in errors
    assert "$.name: is required" in _errors(schema, {})
    errors = _errors(schema, {"name": "abcde", "value": 1})
    assert "$.name: is too long" in errors
    assert "$.value: is not allowed" in errors


def test_arguments_object_capacity_and_property_name_pattern_rejection() -> None:
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "minProperties": 2,
        "maxProperties": 3,
        "propertyNames": {"pattern": "^[a-z]+$"},
        "additionalProperties": {"type": "boolean"},
    }
    assert not validate_arguments(schema, {"a": 1, "b": True})
    too_few = _errors(schema, {"a": 1})
    assert "$: has too few properties" in too_few
    too_many = _errors(schema, {"a": 1, "b": True, "c": False, "d": False})
    assert "$: has too many properties" in too_many
    invalid_names = _errors(schema, {"a": 1, "b": True, "bad-key": False})
    assert "$.bad-key: has an invalid name" in invalid_names


def test_arguments_rejects_invalid_pattern_schema_and_property_name_schema() -> None:
    string_schema = {"type": "string", "pattern": "["}
    assert "$: has an invalid pattern schema" in _errors(string_schema, "abc")
    object_schema = {
        "type": "object",
        "properties": {},
        "propertyNames": {"pattern": "("},
    }
    errors = _errors(object_schema, {"k": 1})
    assert "$: has an invalid property-name schema" in errors


def test_arguments_string_pattern_and_enum_rejection() -> None:
    schema = {
        "type": "object",
        "properties": {"mode": {"type": "string", "pattern": "^[a-z]+$", "enum": ["fast", "slow"]}},
    }
    assert not validate_arguments(schema, {"mode": "fast"})
    errors = _errors(schema, {"mode": "UP"})
    assert "$.mode: does not match the required pattern" in errors
    assert "$.mode: is not an allowed value" in errors


def test_arguments_integer_and_number_bounds_with_type_rejection() -> None:
    schema = {
        "type": "object",
        "properties": {"age": {"type": "integer", "minimum": 1, "maximum": 3}},
    }
    assert not validate_arguments(schema, {"age": 2})
    assert "$.age: must be an integer" in _errors(schema, {"age": True})
    assert "$.age: must be an integer" in _errors(schema, {"age": "2"})
    assert "$.age: is below the minimum" in _errors(schema, {"age": 0})
    assert "$.age: is above the maximum" in _errors(schema, {"age": 4})
    number_schema = {
        "type": "object",
        "properties": {"x": {"type": "number", "minimum": 1.5, "maximum": 5.5}},
    }
    assert "$.x: must be a number" in _errors(number_schema, {"x": False})
    assert "$.x: must be finite" in _errors(number_schema, {"x": float("nan")})
    assert "$.x: is below the minimum" in _errors(number_schema, {"x": 1.0})
    assert "$.x: is above the maximum" in _errors(number_schema, {"x": 6.0})


def test_arguments_array_capacity_and_unsupported_type() -> None:
    schema = {
        "type": "object",
        "properties": {
            "items": {"type": "array", "minItems": 1, "maxItems": 2, "items": {"type": "integer"}}
        },
    }
    assert not validate_arguments(schema, {"items": [1, 2]})
    assert "$.items: must be an array" in _errors(schema, {"items": "nope"})
    assert "$.items: has too few items" in _errors(schema, {"items": []})
    assert "$.items: has too many items" in _errors(schema, {"items": [1, 2, 3]})
    assert "$.items[0]: must be an integer" in _errors(schema, {"items": ["x"]})
    assert "$.value: has an unsupported schema type" in _errors(
        {"type": "object", "properties": {"value": {"type": "datetime"}}}, {"value": "x"}
    )


def test_arguments_one_of_any_of_and_not_combinator_branches() -> None:
    schema = {
        "type": "object",
        "properties": {
            "port": {
                "oneOf": [
                    {"type": "integer", "minimum": 1},
                    {"type": "integer", "minimum": 2},
                ]
            }
        },
    }
    assert not validate_arguments(schema, {"port": 1})
    errors = _errors(schema, {"port": 3})
    assert "$.port: must match exactly one allowed shape" in errors
    errors = _errors(schema, {"port": 0})
    assert "$.port: must match exactly one allowed shape" in errors

    any_schema = {
        "type": "object",
        "properties": {
            "kind": {
                "anyOf": [{"type": "string", "pattern": "^a"}, {"type": "string", "pattern": "^b"}]
            }
        },
    }
    assert not validate_arguments(any_schema, {"kind": "apple"})
    assert not validate_arguments(any_schema, {"kind": "banana"})
    assert "$.kind: must match an allowed shape" in _errors(any_schema, {"kind": "cherry"})

    not_schema = {
        "type": "object",
        "properties": {"ban": {"not": {"type": "string"}}},
    }
    assert not validate_arguments(not_schema, {"ban": 1})
    assert "$.ban: matches a disallowed shape" in _errors(not_schema, {"ban": "x"})


def test_command_allowlist_rejects_injection_and_inner_boundaries() -> None:
    allowlist = normalize_command_allowlist(("/usr/bin/printf",))
    assert allowlist == frozenset({"/usr/bin/printf"})

    argv = normalize_command_argv(["/usr/bin/printf", "hello"], allowlist=allowlist)
    assert argv == ("/usr/bin/printf", "hello")

    with pytest.raises(ValueError, match="must be strings"):
        normalize_command_allowlist(("/usr/bin/printf", b"/bin/ls"))
    with pytest.raises(ValueError, match="is invalid"):
        normalize_command_allowlist(("/usr/bin/printf", "  "))
    with pytest.raises(ValueError, match="is invalid"):
        normalize_command_allowlist(("/usr/bin/printf", "x" * 4097))
    with pytest.raises(ValueError, match="is invalid"):
        normalize_command_allowlist(("/usr/bin/printf", "bad\nentry"))
    with pytest.raises(ValueError, match="non-empty argv"):
        normalize_command_argv([], allowlist=allowlist)
    with pytest.raises(ValueError, match="non-empty argv"):
        normalize_command_argv(["/usr/bin/printf"] * 17, allowlist=allowlist)
    with pytest.raises(ValueError, match="must be strings"):
        normalize_command_argv(["/usr/bin/printf", 3], allowlist=allowlist)
    with pytest.raises(ValueError, match="is unsafe"):
        normalize_command_argv(["/usr/bin/printf", ""], allowlist=allowlist)
    with pytest.raises(ValueError, match="is unsafe"):
        normalize_command_argv(["/usr/bin/printf", "bad\rcommand"], allowlist=allowlist)
    with pytest.raises(PermissionError, match="allowlist"):
        normalize_command_argv(["/bin/ls"], allowlist=allowlist)
    with pytest.raises(PermissionError, match="allowlist"):
        normalize_command_argv(["/usr/bin/printf"], allowlist=frozenset())
    with pytest.raises(PermissionError, match="allowlist"):
        normalize_command_argv(["/bin/sh", "-c", "x & y"], allowlist=allowlist)


def test_command_timeout_clamps_bounds_and_rejects_invalid_values() -> None:
    assert normalize_command_timeout(1.5) == 1.5
    assert normalize_command_timeout("7") == 7.0
    assert normalize_command_timeout(999, maximum=30.0) == 30.0
    assert normalize_command_timeout(0.01, maximum=30.0) == 0.1
    assert normalize_command_timeout(0.1) == 0.1
    for invalid in (True, "nan", float("inf"), object()):
        with pytest.raises(ValueError, match="finite number"):
            normalize_command_timeout(invalid)


def test_schema_migration_rolls_back_corrupt_migration_atomically(tmp_path: Path) -> None:
    """损坏迁移失败时不得留下任何半初始化表；异常传播且连接可重新打开。"""

    database_path = tmp_path / "corrupt-migration.sqlite3"
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA foreign_keys = ON")
    migrator = SchemaMigrator()

    # 在事务尾部注入真实 DDL 失败：到这一步所有 CREATE TABLE 已执行，
    # 事务内任何一步失败都必须整体回滚。
    with patch.object(
        SchemaMigrator,
        "_ensure_memory_fts",
        side_effect=sqlite3.OperationalError("synthetic migration failure"),
    ):
        with pytest.raises(sqlite3.OperationalError, match="synthetic migration failure"):
            migrator.migrate(connection)
    connection.close()

    with sqlite3.connect(database_path) as reopened:
        surfaced = reopened.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'chat_history'"
        ).fetchone()[0]
        assert surfaced == 0

    recovered = sqlite3.connect(database_path)
    try:
        migrator.migrate(recovered)
        version = recovered.execute("SELECT MAX(version) FROM memory_schema_version").fetchone()[0]
        assert version == SCHEMA_VERSION
        assert (
            recovered.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'chat_history'"
            ).fetchone()[0]
            == 1
        )
    finally:
        recovered.close()
