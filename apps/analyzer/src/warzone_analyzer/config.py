from __future__ import annotations

import os
from pathlib import Path

import yaml

from .models import AnalyzerConfig, DebugConfig, FightDetectionConfig, OcrConfig, Region, SamplingConfig


def load_config(config_path: Path | None) -> AnalyzerConfig:
    path = config_path or _default_config_path()
    if not path.exists():
        return AnalyzerConfig()

    with path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file) or {}

    return AnalyzerConfig(
        sampling=SamplingConfig(**(raw_config.get("sampling") or {})),
        fight_detection=FightDetectionConfig(**(raw_config.get("fight_detection") or {})),
        ocr=OcrConfig(**(raw_config.get("ocr") or {})),
        debug=DebugConfig(**(raw_config.get("debug") or {})),
        regions={
            name: Region(**region_config)
            for name, region_config in (raw_config.get("regions") or {}).items()
        },
    )


def _default_config_path() -> Path:
    env_path = os.getenv("WARZONE_ANALYZER_CONFIG")
    if env_path:
        return Path(env_path)
    return Path(__file__).resolve().parents[2] / "config" / "default.yml"
