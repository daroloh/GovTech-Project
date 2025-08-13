from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict


def get_logger(name: str) -> logging.Logger:
	logger = logging.getLogger(name)
	if not logger.handlers:
		logger.setLevel(logging.INFO)
		handler = logging.StreamHandler()
		formatter = logging.Formatter(
			"%(asctime)s | %(levelname)s | %(name)s | %(message)s"
		)
		handler.setFormatter(formatter)
		logger.addHandler(handler)
	return logger


def save_json(data: Dict[str, Any], path: str) -> None:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(data, f, indent=2, default=str)


def utc_now_str() -> str:
	return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
