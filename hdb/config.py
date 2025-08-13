import os
from dataclasses import dataclass
from typing import Optional

import yaml
from dotenv import load_dotenv


@dataclass
class Paths:
	duckdb_path: str
	model_dir: str
	metrics_path: str
	logs_dir: str
	features_table: str
	clean_table: str
	raw_table: str


@dataclass
class TrainingConfig:
	target: str
	test_size: float
	random_state: int
	model_type: str
	n_estimators: int
	max_depth: Optional[int]
	discount_rate: float


@dataclass
class APIConfig:
	host: str
	port: int


@dataclass
class LLMConfig:
	provider: str
	model: str
	max_tokens: int
	temperature: float


@dataclass
class ProjectConfig:
	name: str
	version: str
	paths: Paths
	training: TrainingConfig
	api: APIConfig
	llm: LLMConfig


def load_config(path: str = "config.yaml") -> ProjectConfig:
	# Load environment variables from .env if present
	load_dotenv()
	with open(path, "r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f)
	paths = Paths(**cfg["paths"])
	training = TrainingConfig(**cfg["training"])
	api = APIConfig(**cfg["api"])
	llm = LLMConfig(**cfg["llm"])
	project = ProjectConfig(
		name=cfg["project"]["name"],
		version=cfg["project"]["version"],
		paths=paths,
		training=training,
		api=api,
		llm=llm,
	)
	# ensure dirs
	os.makedirs(os.path.dirname(paths.duckdb_path), exist_ok=True)
	os.makedirs(paths.model_dir, exist_ok=True)
	os.makedirs(os.path.dirname(paths.metrics_path), exist_ok=True)
	os.makedirs(paths.logs_dir, exist_ok=True)
	return project
