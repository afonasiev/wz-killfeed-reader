from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
import unicodedata

import cv2
import numpy as np

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


@dataclass
class TeamMemberProfile:
    name: str
    color_hex: str | None
    source: str
    first_seen_ms: int
    last_seen_ms: int
    votes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "color_hex": self.color_hex,
            "source": self.source,
            "first_seen_ms": self.first_seen_ms,
            "last_seen_ms": self.last_seen_ms,
            "votes": self.votes,
        }


class TeamDetector:
    def __init__(self, config: AnalyzerConfig, ocr: TesseractOcr, output: AnalyzerOutput) -> None:
        self._config = config
        self._ocr = ocr
        self._output = output
        self._members: list[str] = []
        self._candidates: Counter[str] = Counter()
        self._profiles: dict[str, TeamMemberProfile] = {}
        self._history: list[dict[str, object]] = []
        self._last_ocr_ms: int | None = None

    @property
    def members(self) -> list[str]:
        return self._members

    @property
    def history(self) -> list[dict[str, object]]:
        return self._history

    @property
    def profiles(self) -> list[dict[str, object]]:
        ordered = sorted(
            (profile for profile in self._profiles.values() if profile.name in self._members and profile.votes >= 2),
            key=lambda profile: profile.first_seen_ms,
        )
        return [profile.to_dict() for profile in ordered]

    def process(self, sampled_frame: SampledFrame, prefer_lobby: bool) -> list[AnalyzerEvent]:
        region_name = "lobby_party" if prefer_lobby else "squad_hud"
        region = self._config.regions.get(region_name)
        if region is None or not self._should_run(sampled_frame.timestamp_ms):
            return []

        crop = crop_region(sampled_frame.image, region)
        crop_path = None
        if self._config.ocr.save_crops:
            crop_path = self._output.save_debug_crop(crop, "team", region_name, sampled_frame)

        row_detections = [] if prefer_lobby else self._extract_hud_rows(crop, sampled_frame)
        row_names = [detection["name"] for detection in row_detections]

        result = self._ocr.read_text(crop, mode="text")
        block_detected = self._extract_members(result.normalized, prefer_lobby=prefer_lobby)
        detected = _dedupe([*row_names, *block_detected])
        for member in detected:
            self._candidates[member] += 1
        for detection in row_detections:
            self._update_profile(
                name=str(detection["name"]),
                color_hex=detection.get("color_hex") if isinstance(detection.get("color_hex"), str) else None,
                source="squad_hud_row",
                timestamp_ms=sampled_frame.timestamp_ms,
            )

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
                "team_colors": self.profiles,
            }
        )
        return [
            AnalyzerEvent(
                type=EventType.TEAM_DETECTED,
                timestamp_ms=sampled_frame.timestamp_ms,
                frame_index=sampled_frame.frame_index,
                confidence=min(len(self._members) / 4, 1.0),
                source=f"{region_name}_ocr",
                details={
                    "team_members": self._members,
                    "team_colors": self.profiles,
                    "row_detections": row_detections,
                    "raw_text": result.text,
                    "crop": crop_path,
                },
            )
        ]

    def _should_run(self, timestamp_ms: int) -> bool:
        if self._last_ocr_ms is None:
            self._last_ocr_ms = timestamp_ms
            return True
        interval_ms = max(int(self._config.ocr.interval_seconds * 250), 750)
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

    def _extract_hud_rows(self, crop: np.ndarray, sampled_frame: SampledFrame) -> list[dict[str, object]]:
        detections = []
        for index, row in enumerate(_extract_squad_hud_rows(crop)):
            crop_path = None
            if self._config.ocr.save_crops:
                crop_path = self._output.save_debug_crop(
                    row["image"],
                    "team",
                    f"squad_hud_row_{index}",
                    sampled_frame,
                )
            text = self._ocr.read_text(row["image"], mode="text").normalized
            names = _parse_hud_names(text)
            if not names:
                continue
            detections.append(
                {
                    "name": names[0],
                    "color_hex": row["color_hex"],
                    "raw_text": text,
                    "crop": crop_path,
                    "row_y": row["y"],
                }
            )
        return detections

    def _update_profile(self, name: str, color_hex: str | None, source: str, timestamp_ms: int) -> None:
        existing = self._profiles.get(name)
        if existing is None:
            self._profiles[name] = TeamMemberProfile(
                name=name,
                color_hex=color_hex,
                source=source,
                first_seen_ms=timestamp_ms,
                last_seen_ms=timestamp_ms,
                votes=1,
            )
            return
        existing.last_seen_ms = timestamp_ms
        existing.votes += 1
        if color_hex:
            existing.color_hex = color_hex
        existing.source = source


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


def _extract_squad_hud_rows(crop: np.ndarray) -> list[dict[str, object]]:
    if crop.size == 0:
        return []

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = ((saturation > 55) & (value > 75)).astype(np.uint8) * 255
    left_limit = max(int(crop.shape[1] * 0.82), 1)
    mask[:, left_limit:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bars = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width < crop.shape[1] * 0.22 or not 2 <= height <= 12:
            continue
        if x < crop.shape[1] * 0.08 or y < crop.shape[0] * 0.03:
            continue
        bars.append((x, y, width, height))

    rows = []
    for x, y, width, height in sorted(bars, key=lambda item: item[1]):
        center_y = y + height // 2
        if any(abs(int(row["y"]) - center_y) < 18 for row in rows):
            continue
        y1 = max(y - 24, 0)
        y2 = min(y + 5, crop.shape[0])
        x1 = max(int(crop.shape[1] * 0.05), 0)
        x2 = min(max(x + width, int(crop.shape[1] * 0.56)), left_limit)
        name_image = crop[y1:y2, x1:x2]
        color_hex = _dominant_hex_color(crop[y1:y, x1:x2]) or _dominant_hex_color(crop[y : y + height, x : x + width])
        rows.append({"image": name_image, "y": center_y, "color_hex": color_hex})
    return rows[:4]


def _dominant_hex_color(image: np.ndarray) -> str | None:
    if image.size == 0:
        return None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = (saturation > 65) & (value > 80)
    pixels = image[mask]
    if len(pixels) == 0:
        return None
    # OpenCV uses BGR; median is stable against icons and glow.
    bgr = np.median(pixels, axis=0).astype(int)
    return f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"


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
    cleaned = re.sub(r"\$?\d{2,}\$?", " ", cleaned)
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
    if len(set(meaningful.lower())) <= 2 and len(meaningful) > 3:
        return False
    if re.search(r"(.)\1{2,}", meaningful):
        return False
    if _script_switches(candidate) > 2:
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


def _script_switches(value: str) -> int:
    scripts = []
    for char in value:
        if char in {" ", "_"} or char.isdigit():
            continue
        name = unicodedata.name(char, "")
        if "CYRILLIC" in name:
            script = "cyrillic"
        elif "HIRAGANA" in name or "KATAKANA" in name or "CJK" in name:
            script = "cjk"
        elif "LATIN" in name:
            script = "latin"
        else:
            script = "other"
        if not scripts or scripts[-1] != script:
            scripts.append(script)
    return max(len(scripts) - 1, 0)


def _dedupe(values: list[str]) -> list[str]:
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
