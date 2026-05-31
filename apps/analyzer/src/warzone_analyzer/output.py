from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .models import AnalyzerEvent, AnalyzerSummary, FightSegment
from .video import SampledFrame


class AnalyzerOutput:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.debug_frames_dir = output_dir / "debug_frames"
        self.debug_crops_dir = output_dir / "debug_crops"
        self.events_path = output_dir / "events.jsonl"
        self.summary_path = output_dir / "summary.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug_frames_dir.mkdir(parents=True, exist_ok=True)
        self.debug_crops_dir.mkdir(parents=True, exist_ok=True)
        self.events_path.write_text("", encoding="utf-8")

    def write_event(self, event: AnalyzerEvent) -> None:
        with self.events_path.open("a", encoding="utf-8") as event_file:
            event_file.write(event.to_json() + "\n")

    def write_summary(self, summary: AnalyzerSummary) -> None:
        with self.summary_path.open("w", encoding="utf-8") as summary_file:
            json.dump(summary.to_dict(), summary_file, indent=2, ensure_ascii=False)
            summary_file.write("\n")

    def write_fights(self, fights: list[FightSegment]) -> None:
        path = self.output_dir / "fights.json"
        payload = [fight.to_dict() for fight in fights]
        with path.open("w", encoding="utf-8") as fights_file:
            json.dump(payload, fights_file, indent=2, ensure_ascii=False)
            fights_file.write("\n")

    def save_debug_frame(self, sampled_frame: SampledFrame, prefix: str) -> str:
        filename = f"{prefix}_{sampled_frame.timestamp_ms:010d}_{sampled_frame.frame_index}.jpg"
        path = self.debug_frames_dir / filename
        cv2.imwrite(str(path), sampled_frame.image)
        return str(path.relative_to(self.output_dir))

    def save_debug_crop(self, image: np.ndarray, group: str, prefix: str, sampled_frame: SampledFrame) -> str:
        crop_dir = self.debug_crops_dir / group
        crop_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{prefix}_{sampled_frame.timestamp_ms:010d}_{sampled_frame.frame_index}.jpg"
        path = crop_dir / filename
        cv2.imwrite(str(path), image)
        return str(path.relative_to(self.output_dir))
