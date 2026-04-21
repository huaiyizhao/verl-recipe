"""Data-flow logger for debugging DataProto transformations.

Writes structured inspection records to a rotating log file so that every
stage of the pipeline (agent loop → postprocess → addition_process →
expand_intermediate → assemble_batch → trainer) can be audited offline.

Usage::

    from recipe.fully_async_gui_agent.data_flow_logger import log_dataproto

    log_dataproto(data, stage="after_addition_process", extra={"sample_id": sid})

The log file defaults to ``/tmp/dataproto_flow.log`` and can be overridden
via the ``DATAPROTO_FLOW_LOG`` environment variable.
"""

import datetime
import json
import os
import sys
import traceback
from typing import Any, Optional

import numpy as np
import torch

LOG_PATH = os.getenv("DATAPROTO_FLOW_LOG", "/tmp/dataproto_flow.log")
_ENABLED = os.getenv("DATAPROTO_FLOW_LOG_ENABLED", "1") == "1"


def _safe_repr(obj: Any, max_len: int = 200) -> str:
    """Best-effort short repr that never raises."""
    try:
        if isinstance(obj, torch.Tensor):
            return f"Tensor(shape={list(obj.shape)}, dtype={obj.dtype})"
        if isinstance(obj, np.ndarray):
            # Show first few elements for small arrays
            if obj.size <= 5:
                return f"ndarray(shape={list(obj.shape)}, dtype={obj.dtype}, val={obj.tolist()})"
            return f"ndarray(shape={list(obj.shape)}, dtype={obj.dtype})"
        r = repr(obj)
        if len(r) > max_len:
            return r[:max_len] + "..."
        return r
    except Exception:
        return f"<{type(obj).__name__}>"


def _inspect_tensor_dict(td) -> dict[str, str]:
    """Return {key: shape+dtype} for a TensorDict."""
    if td is None:
        return {}
    result = {}
    for k in td.keys():
        v = td[k]
        if hasattr(v, "shape"):
            result[k] = f"shape={list(v.shape)}, dtype={v.dtype}"
        else:
            result[k] = _safe_repr(v, 80)
    return result


def _inspect_non_tensor(nt: dict) -> dict[str, str]:
    """Return {key: type+shape+sample} for a non_tensor_batch dict."""
    if not nt:
        return {}
    result = {}
    for k, v in nt.items():
        if isinstance(v, np.ndarray):
            sample = ""
            if v.size > 0 and v.size <= 3:
                try:
                    sample = f", sample={v.tolist()}"
                except Exception:
                    sample = ""
            elif v.size > 3:
                try:
                    sample = f", first3={v.flat[:3].tolist()}"
                except Exception:
                    sample = ""
            result[k] = f"ndarray(shape={list(v.shape)}, dtype={v.dtype}{sample})"
        elif isinstance(v, list):
            result[k] = f"list(len={len(v)}, type={type(v[0]).__name__ if v else '?'})"
        else:
            result[k] = f"{type(v).__name__}: {_safe_repr(v, 80)}"
    return result


def _inspect_meta_info(mi: dict) -> dict[str, str]:
    """Return {key: type+short_repr} for meta_info."""
    if not mi:
        return {}
    result = {}
    for k, v in mi.items():
        if isinstance(v, list):
            result[k] = f"list(len={len(v)})"
        elif isinstance(v, dict):
            result[k] = f"dict(keys={list(v.keys())[:10]})"
        else:
            result[k] = _safe_repr(v, 80)
    return result


def log_dataproto(
    data,
    stage: str,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Log a DataProto (or dict) snapshot to the flow log file.

    Args:
        data: A DataProto object (or any object with .batch / .non_tensor_batch / .meta_info).
        stage: Human-readable pipeline stage name.
        extra: Optional dict of additional context (sample_id, row_idx, etc.).
    """
    if not _ENABLED:
        return

    try:
        record: dict[str, Any] = {
            "timestamp": datetime.datetime.now().isoformat(),
            "stage": stage,
            "pid": os.getpid(),
        }
        if extra:
            record["extra"] = {k: _safe_repr(v) for k, v in extra.items()}

        if hasattr(data, "batch"):
            record["batch_size"] = len(data) if hasattr(data, "__len__") else "?"
            record["batch_keys"] = _inspect_tensor_dict(data.batch)
            record["non_tensor_batch_keys"] = _inspect_non_tensor(
                data.non_tensor_batch if hasattr(data, "non_tensor_batch") else {}
            )
            record["meta_info_keys"] = _inspect_meta_info(
                data.meta_info if hasattr(data, "meta_info") else {}
            )
        elif isinstance(data, dict):
            record["dict_keys"] = list(data.keys())
            for k, v in data.items():
                record[f"val_{k}"] = _safe_repr(v, 120)
        else:
            record["data_type"] = type(data).__name__
            record["data_repr"] = _safe_repr(data, 300)

        line = json.dumps(record, ensure_ascii=False, default=str)

        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")

    except Exception:
        # Never let logging break the pipeline
        print(
            f"[DataFlowLogger] WARNING: failed to log stage={stage}: "
            f"{traceback.format_exc()}",
            file=sys.stderr,
            flush=True,
        )


def log_message(stage: str, message: str) -> None:
    """Log a free-form message to the flow log file."""
    if not _ENABLED:
        return
    try:
        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "stage": stage,
            "pid": os.getpid(),
            "message": message,
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass
