from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class EventType(str, Enum):
    FIGHT_STARTED = "fight_started"
    FIGHT_ENDED = "fight_ended"
    ACTIVITY_SAMPLE = "activity_sample"
    MATCH_ID_DETECTED = "match_id_detected"
    TEAM_DETECTED = "team_detected"
    STATE_CHANGED = "state_changed"
    TEAM_FEED_EVENT = "team_feed_event"
    KILL = "kill"
    KNOCK = "knock"
    DEATH = "death"
    TEAMMATE_DOWN = "teammate_down"
    TEAMMATE_REVIVED = "teammate_revived"


class MatchState(str, Enum):
    LOBBY = "lobby"
    LOADING = "loading"
    CINEMATIC = "cinematic"
    GAMEPLAY = "gameplay"
    SPECTATING_OR_DEAD = "spectating_or_dead"
    UNKNOWN = "unknown"


@dataclass
class Region:
    x: float
    y: float
    width: float
    height: float


@dataclass
class SamplingConfig:
    fps: float = 3.0
    max_frames: Optional[int] = None


@dataclass
class FightDetectionConfig:
    motion_threshold: float = 0.035
    min_active_seconds: float = 2.0
    idle_end_seconds: float = 5.0
    min_duration_seconds: float = 8.0
    review_after_seconds: float = 240.0
    ignore_initial_seconds: float = 90.0


@dataclass
class OcrConfig:
    enabled: bool = True
    interval_seconds: float = 5.0
    tesseract_cmd: str = "tesseract"
    languages: str = "eng+rus+jpn"
    min_match_id_length: int = 16
    min_match_id_votes: int = 2
    save_crops: bool = True


@dataclass
class DebugConfig:
    save_transition_frames: bool = True
    save_every_n_sampled_frames: int = 0


@dataclass
class AnalyzerConfig:
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    fight_detection: FightDetectionConfig = field(default_factory=FightDetectionConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    regions: dict[str, Region] = field(default_factory=dict)


@dataclass
class AnalyzerEvent:
    type: EventType
    timestamp_ms: int
    frame_index: int
    confidence: float
    source: str
    details: dict[str, object] = field(default_factory=dict)
    debug_frame: Optional[str] = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["type"] = self.type.value
        return payload

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


@dataclass
class FightSegment:
    fight_id: int
    started_at_ms: int
    start_frame_index: int
    warzone_match_id: Optional[str] = None
    state: str = MatchState.UNKNOWN.value
    needs_review: bool = False
    evidence: dict[str, object] = field(default_factory=dict)
    ended_at_ms: Optional[int] = None
    end_frame_index: Optional[int] = None
    duration_ms: Optional[int] = None
    start_debug_frame: Optional[str] = None
    end_debug_frame: Optional[str] = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class AnalyzerSummary:
    input: str
    output_dir: Path
    sampled_frames: int
    events: int
    fights: int
    duration_ms: Optional[int]
    warzone_match_id: Optional[str] = None
    warzone_match_ids: list[str] = field(default_factory=list)
    team_members: list[str] = field(default_factory=list)
    team_history: list[dict[str, object]] = field(default_factory=list)
    state_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
