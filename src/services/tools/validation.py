"""无第三方依赖的 JSON Schema 子集校验。"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from .types import ToolValidationError


def validate_arguments(
    schema: Mapping[str, Any], arguments: object
) -> tuple[ToolValidationError, ...]:
    errors: list[ToolValidationError] = []
    _validate(dict(schema), arguments, "$", errors)
    return tuple(errors)


def _validate(
    schema: Mapping[str, Any], value: object, path: str, errors: list[ToolValidationError]
) -> None:
    one_of = schema.get("oneOf")
    if isinstance(one_of, list):
        matches = 0
        for branch in one_of:
            if not isinstance(branch, Mapping):
                continue
            branch_errors: list[ToolValidationError] = []
            _validate(branch, value, path, branch_errors)
            if not branch_errors:
                matches += 1
        if matches != 1:
            errors.append(ToolValidationError(path, "must match exactly one allowed shape"))

    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        matched = False
        for branch in any_of:
            if not isinstance(branch, Mapping):
                continue
            branch_errors: list[ToolValidationError] = []
            _validate(branch, value, path, branch_errors)
            if not branch_errors:
                matched = True
                break
        if not matched:
            errors.append(ToolValidationError(path, "must match an allowed shape"))

    excluded = schema.get("not")
    if isinstance(excluded, Mapping):
        excluded_errors: list[ToolValidationError] = []
        _validate(excluded, value, path, excluded_errors)
        if not excluded_errors:
            errors.append(ToolValidationError(path, "matches a disallowed shape"))

    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, Mapping):
            errors.append(ToolValidationError(path, "must be an object"))
            return
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            properties = {}
        required = schema.get("required", ())
        if isinstance(required, list):
            for key in required:
                if key not in value:
                    errors.append(ToolValidationError(f"{path}.{key}", "is required"))
        minimum = schema.get("minProperties")
        maximum = schema.get("maxProperties")
        if isinstance(minimum, int) and len(value) < minimum:
            errors.append(ToolValidationError(path, "has too few properties"))
        if isinstance(maximum, int) and len(value) > maximum:
            errors.append(ToolValidationError(path, "has too many properties"))
        property_names = schema.get("propertyNames")
        if isinstance(property_names, Mapping):
            pattern = property_names.get("pattern")
            if isinstance(pattern, str):
                try:
                    compiled = re.compile(pattern)
                except re.error:
                    errors.append(ToolValidationError(path, "has an invalid property-name schema"))
                else:
                    for key in value:
                        if not isinstance(key, str) or compiled.search(key) is None:
                            errors.append(
                                ToolValidationError(f"{path}.{key}", "has an invalid name")
                            )
        additional = schema.get("additionalProperties")
        for key, nested in value.items():
            if key in properties:
                continue
            if additional is False:
                errors.append(ToolValidationError(f"{path}.{key}", "is not allowed"))
            elif isinstance(additional, Mapping):
                _validate(additional, nested, f"{path}.{key}", errors)
        for key, nested_schema in properties.items():
            if key in value and isinstance(nested_schema, Mapping):
                _validate(nested_schema, value[key], f"{path}.{key}", errors)
    elif expected == "array":
        if not isinstance(value, list):
            errors.append(ToolValidationError(path, "must be an array"))
            return
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if isinstance(minimum, int) and len(value) < minimum:
            errors.append(ToolValidationError(path, "has too few items"))
        if isinstance(maximum, int) and len(value) > maximum:
            errors.append(ToolValidationError(path, "has too many items"))
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                _validate(item_schema, item, f"{path}[{index}]", errors)
    elif expected == "string":
        if not isinstance(value, str):
            errors.append(ToolValidationError(path, "must be a string"))
            return
        if isinstance(schema.get("minLength"), int) and len(value) < schema["minLength"]:
            errors.append(ToolValidationError(path, "is too short"))
        if isinstance(schema.get("maxLength"), int) and len(value) > schema["maxLength"]:
            errors.append(ToolValidationError(path, "is too long"))
        pattern = schema.get("pattern")
        if isinstance(pattern, str):
            try:
                matches = re.search(pattern, value) is not None
            except re.error:
                errors.append(ToolValidationError(path, "has an invalid pattern schema"))
            else:
                if not matches:
                    errors.append(ToolValidationError(path, "does not match the required pattern"))
    elif expected == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append(ToolValidationError(path, "must be an integer"))
        else:
            _validate_number_bounds(schema, value, path, errors)
    elif expected == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(ToolValidationError(path, "must be a number"))
        else:
            if not math.isfinite(float(value)):
                errors.append(ToolValidationError(path, "must be finite"))
            else:
                _validate_number_bounds(schema, value, path, errors)
    elif expected == "boolean":
        if not isinstance(value, bool):
            errors.append(ToolValidationError(path, "must be a boolean"))
    elif expected is not None:
        errors.append(ToolValidationError(path, "has an unsupported schema type"))
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        errors.append(ToolValidationError(path, "is not an allowed value"))


def _validate_number_bounds(
    schema: Mapping[str, Any], value: int | float, path: str, errors: list[ToolValidationError]
) -> None:
    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    if isinstance(minimum, (int, float)) and value < minimum:
        errors.append(ToolValidationError(path, "is below the minimum"))
    if isinstance(maximum, (int, float)) and value > maximum:
        errors.append(ToolValidationError(path, "is above the maximum"))
