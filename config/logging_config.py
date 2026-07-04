from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logging(level: str | None = None) -> None:
    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    _ensure_run_id()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    _setup_console_log()
    _setup_all_log_file()
    _setup_image_flow_log_file()


def _ensure_run_id() -> str:
    run_id = os.getenv("LOG_RUN_ID")
    if run_id:
        return run_id
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_id = f"{run_id}_{os.getpid()}"
    os.environ["LOG_RUN_ID"] = run_id
    return run_id


def _setup_all_log_file() -> None:
    base_dir = Path(os.getenv("LOG_DIR", "logs"))
    base_dir.mkdir(parents=True, exist_ok=True)
    run_id = _ensure_run_id()
    default_name = f"all_{run_id}.log"
    log_path = str((base_dir / os.getenv("LOG_ALL_FILENAME", default_name)).resolve())
    root = logging.getLogger()
    if _has_file_handler(root, log_path):
        return
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    root.addHandler(handler)


def _setup_console_log() -> None:
    root = logging.getLogger()
    if any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    root.addHandler(handler)


def _setup_image_flow_log_file() -> None:
    base_dir = Path(os.getenv("LOG_DIR", "logs"))
    base_dir.mkdir(parents=True, exist_ok=True)
    run_id = _ensure_run_id()

    image_flow_logger = logging.getLogger("image_flow")
    image_flow_logger.setLevel(logging.INFO)
    image_flow_logger.propagate = False

    image_flow_path = str((base_dir / f"image_flow_{run_id}.jsonl").resolve())
    if not _has_file_handler(image_flow_logger, image_flow_path):
        image_flow_handler = logging.FileHandler(image_flow_path, encoding="utf-8")
        image_flow_handler.setFormatter(logging.Formatter("%(message)s"))
        image_flow_logger.addHandler(image_flow_handler)


def _has_file_handler(logger: logging.Logger, path: str) -> bool:
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            if str(Path(getattr(handler, "baseFilename", "")).resolve()) == path:
                return True
    return False
