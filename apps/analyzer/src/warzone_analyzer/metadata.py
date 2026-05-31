from __future__ import annotations

from collections import Counter
import re
import unicodedata

from .models import AnalyzerConfig, AnalyzerEvent, EventType
from .ocr import StableTextVote, TesseractOcr
from .output import AnalyzerOutput
from .regions import crop_region
from .video import SampledFrame


class MatchIdDetector:
    def __init__(self, config: AnalyzerConfig, ocr: TesseractOcr, output: AnalyzerOutput) -> None:
        self._config = config
        self._ocr = ocr
        self._output = output
        self._vote = StableTextVote()
        self._last_ocr_ms: int | None = None
        self._last_emitted: str | None = None

    @property
    def best_match_id(self) -> str | None:
        return self._last_emitted

    @property
    def match_ids(self) -> list[str]:
        return self._vote.values(self._config.ocr.min_match_id_votes)

    def process(self, sampled_frame: SampledFrame) -> list[AnalyzerEvent]:
        region = self._config.regions.get("match_id")
        if region is None or not self._should_run(sampled_frame.timestamp_ms):
            return []

        crop = crop_region(sampled_frame.image, region)
        crop_path = None
        if self._config.ocr.save_crops:
            crop_path = self._output.save_debug_crop(crop, "match_id", "match_id", sampled_frame)

        result = self._ocr.read_text(crop, mode="match_id")
        if len(result.normalized) < self._config.ocr.min_match_id_length:
            return []
        if not result.normalized.isdigit():
            return []

        self._vote.add(result.normalized)
        current = result.normalized
        if self._vote.count(current) < self._config.ocr.min_match_id_votes:
            return []
        if current == self._last_emitted:
            return []

        self._last_emitted = current
        return [
            AnalyzerEvent(
                type=EventType.MATCH_ID_DETECTED,
                timestamp_ms=sampled_frame.timestamp_ms,
                frame_index=sampled_frame.frame_index,
                confidence=result.confidence,
                source="match_id_ocr",
                details={
                    "warzone_match_id": current,
                    "raw_text": result.text,
                    "votes": self._vote.count(current),
                    "crop": crop_path,
                },
            )
        ]

    def _should_run(self, timestamp_ms: int) -> bool:
        if self._last_ocr_ms is None:
            self._last_ocr_ms = timestamp_ms
            return True
        interval_ms = int(self._config.ocr.interval_seconds * 1000)
        if timestamp_ms - self._last_ocr_ms >= interval_ms:
            self._last_ocr_ms = timestamp_ms
            return True
        return False


class TeamDetector:
    def __init__(self, config: AnalyzerConfig, ocr: TesseractOcr, output: AnalyzerOutput) -> None:
        self._config = config
        self._ocr = ocr
        self._output = output
        self._members: list[str] = []
        self._candidates: Counter[str] = Counter()
        self._history: list[dict[str, object]] = []
        self._last_ocr_ms: int | None = None

    @property
    def members(self) -> list[str]:
        return self._members

    @property
    def history(self) -> list[dict[str, object]]:
        return self._history

    def process(self, sampled_frame: SampledFrame, prefer_lobby: bool) -> list[AnalyzerEvent]:
        region_name = "lobby_party" if prefer_lobby else "squad_hud"
        region = self._config.regions.get(region_name)
        if region is None or not self._should_run(sampled_frame.timestamp_ms):
            return []

        crop = crop_region(sampled_frame.image, region)
        crop_path = None
        if self._config.ocr.save_crops:
            crop_path = self._output.save_debug_crop(crop, "team", region_name, sampled_frame)

        result = self._ocr.read_text(crop, mode="text")
        detected = self._extract_members(result.normalized, prefer_lobby=prefer_lobby)
        for member in detected:
            self._candidates[member] += 1
        stable_detected = [member for member, count in self._candidates.most_common() if count >= 2]
        merged = self._merge_members(self._members, stable_detected)
        if merged == self._members:
            return []

        self._members = merged
        self._history.append(
            {
                "timestamp_ms": sampled_frame.timestamp_ms,
                "frame_index": sampled_frame.frame_index,
                "source": region_name,
                "team_members": self._members,
            }
        )
        return [
            AnalyzerEvent(
                type=EventType.TEAM_DETECTED,
                timestamp_ms=sampled_frame.timestamp_ms,
                frame_index=sampled_frame.frame_index,
                confidence=min(len(self._members) / 4, 1.0),
                source=f"{region_name}_ocr",
                details={"team_members": self._members, "raw_text": result.text, "crop": crop_path},
            )
        ]

    def _should_run(self, timestamp_ms: int) -> bool:
        if self._last_ocr_ms is None:
            self._last_ocr_ms = timestamp_ms
            return True
        interval_ms = int(self._config.ocr.interval_seconds * 1000)
        if timestamp_ms - self._last_ocr_ms >= interval_ms:
            self._last_ocr_ms = timestamp_ms
            return True
        return False

    def _extract_members(self, text: str, prefer_lobby: bool) -> list[str]:
        if prefer_lobby:
            return _parse_lobby_names(text)
        return _parse_hud_names(text)

    @staticmethod
    def _merge_members(current: list[str], detected: list[str]) -> list[str]:
        merged = list(current)
        for member in detected:
            if member and member not in merged:
                merged.append(member)
        return merged[:4]


UI_STOPWORDS = {
    "активная",
    "возрождение",
    "искать",
    "бой",
    "карта",
    "rebirth",
    "island",
    "loading",
    "match",
    "spectating",
    "наблюдение",
    "союзник",
    "ранен",
}


def _parse_lobby_names(text: str) -> list[str]:
    names = []
    for line in text.splitlines():
        candidate = _cleanup_name_candidate(line, from_lobby=True)
        if _looks_like_player_name(candidate):
            names.append(candidate)
    return _dedupe(names)[:4]


def _parse_hud_names(text: str) -> list[str]:
    names = []
    for line in text.splitlines():
        candidate = _cleanup_name_candidate(line, from_lobby=False)
        if _looks_like_player_name(candidate):
            names.append(candidate)
    return _dedupe(names)[:4]


def _cleanup_name_candidate(line: str, from_lobby: bool) -> str:
    cleaned = line.strip()
    cleaned = re.sub(r"^\s*\d+\s+", "", cleaned)
    cleaned = re.sub(r"\$+\s*\d+|\d+\s*\$", "", cleaned)
    cleaned = re.sub(r"^\[[^\]]{1,8}\]\s*", "", cleaned)
    cleaned = re.sub(r"[•●○◯◎◇◆□■△▲▽▼]+", " ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" -:|/\\.,;")
    cleaned = _normalize_known_ocr_name(cleaned)
    if from_lobby:
        cleaned = re.sub(r"^\d{1,4}\s+", "", cleaned)
    return cleaned.strip()


def _normalize_known_ocr_name(name: str) -> str:
    simplified = re.sub(r"[^A-Za-z0-9]+", "", name).lower()
    if "pascha" in simplified or "pasha" in simplified:
        return "Pascha"
    if "afonas" in simplified or "atonas" in simplified or "phantom" in simplified:
        return "afonas_phantomas"
    if "sirius" in simplified or "sinus" in simplified or "sirius1or" in simplified:
        return "Sirius1or"
    if "リ" in name and "チャ" in name:
        return "リチャードツ"
    return name


def _looks_like_player_name(candidate: str) -> bool:
    if not candidate or len(candidate) < 2:
        return False
    lowered = candidate.lower()
    if any(stopword in lowered for stopword in UI_STOPWORDS):
        return False
    if re.fullmatch(r"[\d\s_$.,:;|/\\+-]+", candidate):
        return False
    meaningful = re.sub(r"[\s$.,:;|/\\+-]", "", candidate)
    if len(meaningful) < 2:
        return False
    return _is_valid_cod_nickname(candidate)


def _is_valid_cod_nickname(candidate: str) -> bool:
    if len(candidate) < 2 or len(candidate) > 16:
        return False
    if candidate[0] in {" ", "_"} or candidate[-1] in {" ", "_"}:
        return False
    if "  " in candidate or "__" in candidate:
        return False
    if re.search(r"[!@#$%^&*()?/\\\[\]]", candidate):
        return False
    return all(_is_allowed_nickname_char(char) for char in candidate)


def _is_allowed_nickname_char(char: str) -> bool:
    if char in {" ", "_"}:
        return True
    category = unicodedata.category(char)
    return category[0] in {"L", "N"}


def _dedupe(values: list[str]) -> list[str]:
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
