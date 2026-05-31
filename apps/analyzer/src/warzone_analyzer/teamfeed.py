from __future__ import annotations

import re
from dataclasses import dataclass

from .models import AnalyzerConfig, AnalyzerEvent, EventType
from .ocr import TesseractOcr
from .output import AnalyzerOutput
from .regions import crop_region
from .video import SampledFrame


@dataclass
class ParsedFeedLine:
    event_type: EventType
    actor: str | None
    target: str | None
    raw_text: str
    relation: str


class TeamFeedDetector:
    def __init__(self, config: AnalyzerConfig, ocr: TesseractOcr, output: AnalyzerOutput) -> None:
        self._config = config
        self._ocr = ocr
        self._output = output
        self._last_ocr_ms: int | None = None
        self._seen: set[tuple[str, int]] = set()

    def process(self, sampled_frame: SampledFrame, team_members: list[str]) -> list[AnalyzerEvent]:
        region = self._config.regions.get("team_feed")
        if region is None or not self._should_run(sampled_frame.timestamp_ms):
            return []

        crop = crop_region(sampled_frame.image, region)
        crop_path = None
        if self._config.ocr.save_crops:
            crop_path = self._output.save_debug_crop(crop, "team_feed", "team_feed", sampled_frame)

        result = self._ocr.read_text(crop, mode="text")
        events = []
        for parsed in parse_team_feed_text(result.normalized, team_members):
            key = (parsed.raw_text, sampled_frame.timestamp_ms // 3000)
            if key in self._seen:
                continue
            self._seen.add(key)
            events.append(
                AnalyzerEvent(
                    type=parsed.event_type,
                    timestamp_ms=sampled_frame.timestamp_ms,
                    frame_index=sampled_frame.frame_index,
                    confidence=0.45,
                    source="team_feed_ocr",
                    details={
                        "actor": parsed.actor,
                        "target": parsed.target,
                        "relation": parsed.relation,
                        "raw_text": parsed.raw_text,
                        "crop": crop_path,
                    },
                )
            )
        return events

    def _should_run(self, timestamp_ms: int) -> bool:
        if self._last_ocr_ms is None:
            self._last_ocr_ms = timestamp_ms
            return True
        interval_ms = max(int(self._config.ocr.interval_seconds * 500), 1000)
        if timestamp_ms - self._last_ocr_ms >= interval_ms:
            self._last_ocr_ms = timestamp_ms
            return True
        return False


def parse_team_feed_text(text: str, team_members: list[str]) -> list[ParsedFeedLine]:
    events = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if len(line) < 3:
            continue

        event_type = _event_type_from_line(line)
        if event_type is None:
            continue

        actor, target = _extract_actor_target(line, team_members)
        relation = _relation_for(actor, target, team_members)
        events.append(
            ParsedFeedLine(
                event_type=event_type,
                actor=actor,
                target=target,
                raw_text=line,
                relation=relation,
            )
        )
    return events


def _event_type_from_line(line: str) -> EventType | None:
    lowered = line.lower()
    if any(token in lowered for token in ["нок", "knock", "downed", "сбил", "ранен"]):
        return EventType.KNOCK
    if any(token in lowered for token in ["убил", "ликвид", "kill", "eliminat", "устран"]):
        return EventType.KILL
    if any(token in lowered for token in ["умер", "dead", "killed by", "убит"]):
        return EventType.DEATH
    return None


def _extract_actor_target(line: str, team_members: list[str]) -> tuple[str | None, str | None]:
    actor = _find_team_member(line, team_members)
    target = None

    separators = ["->", "›", ">", " убил ", " нокнул ", " knocked ", " downed ", " killed "]
    for separator in separators:
        if separator in line:
            parts = [part.strip(" -:|") for part in line.split(separator, 1)]
            if len(parts) == 2:
                actor = actor or _best_name_fragment(parts[0])
                target = _best_name_fragment(parts[1])
                break

    return actor, target


def _find_team_member(line: str, team_members: list[str]) -> str | None:
    simplified_line = _simplify(line)
    for member in team_members:
        simplified_member = _simplify(member)
        if simplified_member and simplified_member in simplified_line:
            return member
    return None


def _best_name_fragment(text: str) -> str | None:
    cleaned = re.sub(r"[^\w\s\u3040-\u30ff\u3400-\u9fff\u0400-\u04ff-]", " ", text, flags=re.UNICODE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if 2 <= len(cleaned) <= 32:
        return cleaned
    return None


def _relation_for(actor: str | None, target: str | None, team_members: list[str]) -> str:
    actor_in_team = actor in team_members if actor else False
    target_in_team = target in team_members if target else False
    if actor_in_team and not target_in_team:
        return "team_did"
    if target_in_team and not actor_in_team:
        return "team_received"
    if actor_in_team and target_in_team:
        return "team_internal_or_ambiguous"
    return "unknown"


def _simplify(value: str) -> str:
    return re.sub(r"[\W_]+", "", value, flags=re.UNICODE).lower()
