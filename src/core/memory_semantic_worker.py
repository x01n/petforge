"""SentenceTransformer 子进程协议；只接收父进程传入的本地模型配置。"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def main() -> int:
    """运行受限 JSONL 协议或环境审计。"""

    if len(sys.argv) == 3 and sys.argv[1] == "--audit-environment":
        key = str(sys.argv[2])
        print(json.dumps({"present": key in os.environ}), flush=True)
        return 0
    try:
        line = next(sys.stdin.buffer)
        request = json.loads(line.decode("utf-8"))
    except (StopIteration, UnicodeDecodeError, json.JSONDecodeError):
        return 2
    if not isinstance(request, dict) or request.get("operation") != "bootstrap":
        return 2
    response = _bootstrap(request)
    if response.get("status") != "ok":
        _write(response)
        return 0
    model = response.pop("_model")
    payload = response.pop("_payload")
    _write(response)
    for line in sys.stdin.buffer:
        try:
            request = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return 2
        if not isinstance(request, dict):
            return 2
        request_id = int(request.get("id", -1))
        operation = request.get("operation")
        if operation == "close":
            _write({"id": request_id, "status": "ok"})
            return 0
        texts = request.get("texts")
        if not isinstance(texts, list) or not texts:
            _write({"id": request_id, "status": "error", "error": "embedding_batch_invalid"})
            continue
        try:
            method: Callable[..., Any]
            if operation == "query":
                method = model.encode_query
            elif operation == "documents":
                method = model.encode_document
            else:
                raise ValueError("unsupported operation")
            vectors = method(
                tuple(str(item) for item in texts),
                batch_size=int(payload["batch_size"]),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            values = vectors.tolist()
            if operation == "query" and values and not isinstance(values[0], list):
                values = [values]
            _write({"id": request_id, "status": "ok", "vectors": values})
        except BaseException:
            _write({"id": request_id, "status": "error", "error": "embedding_failed"})
    return 0


def _bootstrap(request: dict[str, object]) -> dict[str, object]:
    """在任何第三方模型导入前设置离线开关。"""

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    try:
        package_version = version("sentence-transformers")
        from sentence_transformers import SentenceTransformer
    except (ImportError, PackageNotFoundError):
        return {"id": 0, "status": "error", "error": "dependency_unavailable"}
    model_path = Path(str(request.get("model_path") or "")).expanduser().resolve()
    if not model_path.is_dir():
        return {"id": 0, "status": "error", "error": "model_path_missing"}
    try:
        model = SentenceTransformer(
            str(model_path),
            device=str(request.get("device") or "cpu"),
            revision=str(request.get("model_revision") or ""),
            local_files_only=True,
            trust_remote_code=False,
            backend=str(request.get("backend") or "torch"),
        )
        dimension = int(model.get_sentence_embedding_dimension() or 0)
    except BaseException:
        return {"id": 0, "status": "error", "error": "model_load_failed"}
    if dimension != int(request.get("dimension") or 0):
        return {"id": 0, "status": "error", "error": "embedding_dimension_mismatch"}
    return {
        "id": 0,
        "status": "ok",
        "spec": {
            "provider": "sentence_transformers",
            "package_version": package_version,
            "model_id": str(request.get("model_id") or ""),
            "model_revision": str(request.get("model_revision") or ""),
            "dimension": dimension,
        },
        "_model": model,
        "_payload": request,
    }


def _write(value: dict[str, object]) -> None:
    payload = dict(value)
    payload.pop("_model", None)
    payload.pop("_payload", None)
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
