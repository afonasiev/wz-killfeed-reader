from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np

from .models import AnalyzerConfig, AnalyzerEvent, EventType
from .ocr import TesseractOcr, normalize_ocr_lines
from .output import AnalyzerOutput
from .regions import crop_region
from .video import SampledFrame


@dataclass
class FeedName:
    display: str
    nickname: str
    clan_tag: str | None
    confidence: float


@dataclass
class ParsedFeedLine:
    event_type: EventType
    actor: FeedName | None
    target: FeedName | None
    raw_text: str
    relation: str
    confidence: float
    evidence: dict[str, object]


@dataclass
class FeedRow:
    image: np.ndarray
    y1: int
    y2: int
    split_x: int | None
    confidence: float
    red_bbox: tuple[int, int, int, int] | None = None


@dataclass
class PendingFeedEvent:
    first_timestamp_ms: int
    first_frame_index: int
    observations: list[ParsedFeedLine]
    emitted: bool = False


class TeamFeedDetector:
    def __init__(self, config: AnalyzerConfig, ocr: TesseractOcr, output: AnalyzerOutput) -> None:
        self._config = config
        self._ocr = ocr
        self._output = output
        self._last_ocr_ms: int | None = None
        self._seen: set[tuple[str, str | None, str | None, int]] = set()
        self._pending: dict[tuple[str, str | None, int, str], PendingFeedEvent] = {}

    def process(
        self,
        sampled_frame: SampledFrame,
        team_members: list[str],
        team_profiles: list[dict[str, object]] | None = None,
    ) -> list[AnalyzerEvent]:
        region = self._config.regions.get("team_feed")
        if region is None or not self._should_run(sampled_frame.timestamp_ms):
            return []
        if not team_members:
            return []

        crop = crop_region(sampled_frame.image, region)
        crop_path = None
        if self._config.ocr.save_crops:
            crop_path = self._output.save_debug_crop(crop, "team_feed", "team_feed", sampled_frame)

        parsed_lines = self._parse_visual_rows(crop, sampled_frame, team_members, team_profiles or [])
        if not parsed_lines:
            result = self._ocr.read_text(crop, mode="feed_sparse")
            parsed_lines = parse_team_feed_text(result.normalized, team_members)

        events = []
        for parsed in parsed_lines:
            if team_members and parsed.relation == "unknown":
                continue
            stable = self._add_observation(parsed, sampled_frame)
            if stable is None:
                continue
            actor_name = stable.actor.nickname if stable.actor else None
            target_name = stable.target.nickname if stable.target else None
            if target_name is None and stable.event_type in {EventType.KILL, EventType.KNOCK}:
                continue
            key = (stable.event_type.value, actor_name, target_name, sampled_frame.timestamp_ms // 3000)
            if key in self._seen:
                continue
            self._seen.add(key)
            events.append(
                AnalyzerEvent(
                    type=stable.event_type,
                    timestamp_ms=sampled_frame.timestamp_ms,
                    frame_index=sampled_frame.frame_index,
                    confidence=stable.confidence,
                    source="team_feed_ocr",
                    details={
                        "actor": actor_name,
                        "target": target_name,
                        "actor_display": stable.actor.display if stable.actor else None,
                        "target_display": stable.target.display if stable.target else None,
                        "actor_clan_tag": stable.actor.clan_tag if stable.actor else None,
                        "target_clan_tag": stable.target.clan_tag if stable.target else None,
                        "relation": stable.relation,
                        "raw_text": stable.raw_text,
                        "crop": stable.evidence.get("row_crop") or crop_path,
                        "evidence": stable.evidence,
                    },
                )
            )
        return events

    def _add_observation(self, parsed: ParsedFeedLine, sampled_frame: SampledFrame) -> ParsedFeedLine | None:
        actor_name = parsed.actor.nickname if parsed.actor else None
        row_bucket = int(float(parsed.evidence.get("row_y", 0)) // 24)
        key = (parsed.event_type.value, actor_name, row_bucket, parsed.relation)
        pending = self._pending.get(key)
        if pending is None or sampled_frame.timestamp_ms - pending.first_timestamp_ms > 4500:
            pending = PendingFeedEvent(
                first_timestamp_ms=sampled_frame.timestamp_ms,
                first_frame_index=sampled_frame.frame_index,
                observations=[],
            )
            self._pending[key] = pending
        pending.observations.append(parsed)
        if pending.emitted:
            return None
        if len(pending.observations) < 2:
            return None
        target_names = [observation.target.nickname for observation in pending.observations if observation.target is not None]
        if len(pending.observations) < 3 and len(set(target_names)) > 1:
            return None
        pending.emitted = True
        return _merge_observations(pending.observations)

    def _parse_visual_rows(
        self,
        crop: np.ndarray,
        sampled_frame: SampledFrame,
        team_members: list[str],
        team_profiles: list[dict[str, object]],
    ) -> list[ParsedFeedLine]:
        parsed = []
        for index, row in enumerate(_extract_feed_rows(crop)):
            row_crop_path = None
            if self._config.ocr.save_crops:
                row_crop_path = self._output.save_debug_crop(
                    row.image,
                    "team_feed",
                    f"team_feed_row_{index}",
                    sampled_frame,
                )

            full = self._ocr.read_text(row.image, mode="feed_line").normalized
            left_img, right_img = _split_row_image(row)
            left_text = self._read_feed_name(left_img) if left_img.size else ""
            right_text = self._read_feed_name(right_img) if right_img.size else ""
            parsed_line = parse_visual_feed_line(
                full_text=full,
                left_text=left_text,
                right_text=right_text,
                team_members=team_members,
                team_profiles=team_profiles,
                actor_color_hex=_dominant_hex_color(left_img),
                row_confidence=row.confidence,
                row_y=row.y1,
                row_crop_path=row_crop_path,
            )
            if parsed_line is not None:
                parsed.append(parsed_line)
        return parsed

    def _read_feed_name(self, image: np.ndarray) -> str:
        return _join_ocr_texts(
            self._ocr.read_text(image, mode="feed_name").normalized,
            self._ocr.read_text(image, mode="feed_raw").normalized,
        )

    def _should_run(self, timestamp_ms: int) -> bool:
        if self._last_ocr_ms is None:
            self._last_ocr_ms = timestamp_ms
            return True
        interval_ms = 250
        if timestamp_ms - self._last_ocr_ms >= interval_ms:
            self._last_ocr_ms = timestamp_ms
            return True
        return False


def parse_visual_feed_line(
    full_text: str,
    left_text: str,
    right_text: str,
    team_members: list[str],
    team_profiles: list[dict[str, object]] | None,
    actor_color_hex: str | None,
    row_confidence: float,
    row_y: int | None,
    row_crop_path: str | None,
) -> ParsedFeedLine | None:
    full = normalize_ocr_lines(full_text)
    left = normalize_ocr_lines(left_text)
    right = normalize_ocr_lines(right_text)
    actor_from_color = _find_team_member_by_color(actor_color_hex, team_profiles or [])
    actor = _parse_feed_name(left) or _find_team_member_name(full, team_members) or actor_from_color
    if actor_from_color is not None and (actor is None or not _name_in_team(actor.nickname, team_members)):
        actor = actor_from_color
    target = _parse_feed_name(right)

    if actor is None and target is None:
        return None
    if actor is None and full:
        actor = _parse_feed_name(full)
    if actor is None:
        return None

    explicit_event_type = _event_type_from_line(full)
    if target is None and explicit_event_type is None:
        return None
    event_type = explicit_event_type or EventType.KILL
    relation = _relation_for(actor.nickname if actor else None, target.nickname if target else None, team_members)
    confidence = min(max(row_confidence, 0.35) + (0.15 if target else 0.0), 0.82)
    raw_text = " | ".join(part for part in [left, right, full] if part)
    return ParsedFeedLine(
        event_type=event_type,
        actor=actor,
        target=target,
        raw_text=raw_text,
        relation=relation,
        confidence=confidence,
        evidence={
            "parser": "visual_row",
            "left_text": left,
            "right_text": right,
            "full_text": full,
            "actor_color_hex": actor_color_hex,
            "row_y": row_y,
            "row_crop": row_crop_path,
        },
    )


def _join_ocr_texts(*texts: str) -> str:
    lines = []
    for text in texts:
        for line in normalize_ocr_lines(text).splitlines():
            if line and line not in lines:
                lines.append(line)
    return "\n".join(lines)


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
        relation = _relation_for(actor.nickname if actor else None, target.nickname if target else None, team_members)
        events.append(
            ParsedFeedLine(
                event_type=event_type,
                actor=actor,
                target=target,
                raw_text=line,
                relation=relation,
                confidence=0.45,
                evidence={"parser": "text_line"},
            )
        )
    return events


def _merge_observations(observations: list[ParsedFeedLine]) -> ParsedFeedLine:
    actor = _best_feed_name([observation.actor for observation in observations])
    target = _best_feed_name([observation.target for observation in observations])
    latest = observations[-1]
    raw_texts = [observation.raw_text for observation in observations if observation.raw_text]
    crops = [observation.evidence.get("row_crop") for observation in observations if observation.evidence.get("row_crop")]
    target_names = [observation.target.nickname for observation in observations if observation.target is not None]
    target_counter = Counter(target_names)
    target_agreement = target_counter.most_common(1)[0][1] / len(target_names) if target_names else 0.0
    confidence = min(max(observation.confidence for observation in observations) + 0.08 * (len(observations) - 1), 0.92)
    if target_names and target_agreement < 0.67:
        confidence = max(confidence - 0.22, 0.4)
    evidence = dict(latest.evidence)
    evidence.update(
        {
            "parser": "temporal_vote",
            "votes": len(observations),
            "raw_texts": raw_texts[-6:],
            "row_crops": crops[-6:],
            "target_candidates": _name_candidates([observation.target for observation in observations]),
            "actor_candidates": _name_candidates([observation.actor for observation in observations]),
            "target_agreement": target_agreement,
            "needs_review": target_agreement < 0.67,
        }
    )
    return ParsedFeedLine(
        event_type=latest.event_type,
        actor=actor,
        target=target,
        raw_text=" | ".join(raw_texts[-3:]),
        relation=latest.relation,
        confidence=confidence,
        evidence=evidence,
    )


def _best_feed_name(values: list[FeedName | None]) -> FeedName | None:
    candidates = [value for value in values if value is not None]
    if not candidates:
        return None
    by_name = Counter(candidate.nickname for candidate in candidates)
    return max(
        candidates,
        key=lambda candidate: (
            by_name[candidate.nickname],
            _name_information_score(candidate.nickname),
            candidate.confidence,
        ),
    )


def _name_information_score(name: str) -> float:
    simplified = _simplify(name)
    ascii_count = len(re.findall(r"[A-Za-z0-9]", name))
    symbol_penalty = len(re.findall(r"[^A-Za-z0-9А-Яа-я\u3040-\u30ff\u3400-\u9fff _-]", name))
    return len(simplified) + ascii_count * 0.6 - symbol_penalty * 1.5


def _name_candidates(values: list[FeedName | None]) -> list[str]:
    counter = Counter(value.display for value in values if value is not None)
    return [name for name, _ in counter.most_common()]


def _extract_feed_rows(crop: np.ndarray) -> list[FeedRow]:
    if crop.size == 0:
        return []
    mask = _feed_text_mask(crop)
    search_width = crop.shape[1]
    red_rows = _extract_red_anchored_rows(crop, mask, search_width)
    if red_rows:
        return red_rows
    return []


def _extract_red_anchored_rows(crop: np.ndarray, mask: np.ndarray, search_width: int) -> list[FeedRow]:
    red_mask = _red_text_mask(crop[:, :search_width])
    red_components = _feed_components(red_mask)
    if not red_components:
        return []

    groups: list[list[tuple[int, int, int, int]]] = []
    for component in red_components:
        x, y, w, h = component
        if x < crop.shape[1] * 0.04:
            continue
        center_y = y + h / 2
        matched = False
        for group in groups:
            group_center = np.mean([gy + gh / 2 for _, gy, _, gh in group])
            if abs(center_y - group_center) <= max(8, crop.shape[0] * 0.025):
                group.append(component)
                matched = True
                break
        if not matched:
            groups.append([component])

    rows = []
    for group in groups:
        red_x1 = min(x for x, _, _, _ in group)
        red_y1 = min(y for _, y, _, _ in group)
        red_x2 = max(x + w for x, _, w, _ in group)
        red_y2 = max(y + h for _, y, _, h in group)
        if red_x2 - red_x1 < 14 or red_y2 - red_y1 < 5:
            continue
        x1 = max(red_x1 - int(crop.shape[1] * 0.30), 0)
        y1 = max(red_y1 - 16, 0)
        x2 = min(red_x2 + 45, search_width)
        y2 = min(red_y2 + 16, crop.shape[0])
        if x2 - x1 < 70 or y2 - y1 < 12:
            continue
        row_image = crop[y1:y2, x1:x2]
        split_x = _detect_weapon_split(row_image)
        red_pixels = int((_red_text_mask(row_image) > 0).sum())
        confidence = min(max(red_pixels / max((x2 - x1) * (y2 - y1) * 0.08, 1), 0.35), 0.78)
        red_bbox = (red_x1 - x1, red_y1 - y1, red_x2 - x1, red_y2 - y1)
        rows.append(FeedRow(image=row_image, y1=y1, y2=y2, split_x=split_x, confidence=confidence, red_bbox=red_bbox))
    return sorted(rows, key=lambda row: row.y1)[:6]


def _feed_components(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    prepared = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(prepared, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    components = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < 8 or h < 3 or w < 2:
            continue
        if h > mask.shape[0] * 0.22 or w > mask.shape[1] * 0.55:
            continue
        components.append((x, y, w, h))
    return sorted(components, key=lambda item: (item[1], item[0]))


def _feed_text_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    colored = ((saturation > 55) & (value > 85)).astype(np.uint8) * 255
    bright = ((saturation < 75) & (value > 175)).astype(np.uint8) * 255
    return cv2.bitwise_or(colored, bright)


def _red_text_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    red = ((hue < 12) | (hue > 168)) & (saturation > 55) & (value > 70)
    return red.astype(np.uint8) * 255


def _detect_weapon_split(row: np.ndarray) -> int | None:
    if row.size == 0:
        return None
    hsv = cv2.cvtColor(row, cv2.COLOR_BGR2HSV)
    white_mask = ((hsv[:, :, 1] < 60) & (hsv[:, :, 2] > 150)).astype(np.uint8)
    projection = white_mask.sum(axis=0)
    if projection.max(initial=0) < 2:
        return None
    width = row.shape[1]
    left_bound = int(width * 0.18)
    right_bound = int(width * 0.78)
    if right_bound <= left_bound:
        return None
    segment = projection[left_bound:right_bound]
    split = int(np.argmax(segment)) + left_bound
    return split if 0 < split < width else None


def _split_row_image(row: FeedRow) -> tuple[np.ndarray, np.ndarray]:
    image = row.image
    if image.size == 0:
        return image, image
    if row.red_bbox is not None:
        red_x1, red_y1, red_x2, red_y2 = row.red_bbox
        y1 = max(red_y1 - 6, 0)
        y2 = min(red_y2 + 7, image.shape[0])
        left = image[y1:y2, : max(red_x1 - 8, 1)]
        right = image[y1:y2, max(red_x1 - 5, 0) : min(red_x2 + 8, image.shape[1])]
        return _trim_empty_columns(left), _trim_empty_columns(right)
    split_x = row.split_x or int(image.shape[1] * 0.50)
    gap = max(10, int(image.shape[1] * 0.045))
    left = image[:, : max(split_x - gap, 1)]
    right = image[:, min(split_x + gap, image.shape[1] - 1) :]
    return _trim_empty_columns(left), _trim_empty_columns(right)


def _trim_empty_columns(image: np.ndarray) -> np.ndarray:
    if image.size == 0:
        return image
    mask = _feed_text_mask(image)
    columns = np.where((mask > 0).sum(axis=0) > 1)[0]
    if len(columns) < 2:
        return image
    x1 = max(int(columns[0]) - 4, 0)
    x2 = min(int(columns[-1]) + 5, image.shape[1])
    return image[:, x1:x2]


def _event_type_from_line(line: str) -> EventType | None:
    lowered = line.lower()
    if any(token in lowered for token in ["нок", "knock", "downed", "сбил", "ранен"]):
        return EventType.KNOCK
    if any(token in lowered for token in ["убил", "ликвид", "kill", "eliminat", "устран"]):
        return EventType.KILL
    if any(token in lowered for token in ["умер", "dead", "killed by", "убит"]):
        return EventType.DEATH
    return None


def _extract_actor_target(line: str, team_members: list[str]) -> tuple[FeedName | None, FeedName | None]:
    actor = _find_team_member_name(line, team_members)
    target = None

    separators = ["->", "›", ">", " убил ", " нокнул ", " knocked ", " downed ", " killed "]
    for separator in separators:
        if separator in line:
            parts = [part.strip(" -:|") for part in line.split(separator, 1)]
            if len(parts) == 2:
                actor = actor or _parse_feed_name(parts[0])
                target = _parse_feed_name(parts[1])
                break

    return actor, target


def _find_team_member_name(line: str, team_members: list[str]) -> FeedName | None:
    simplified_line = _simplify(line)
    for member in team_members:
        simplified_member = _simplify(member)
        if simplified_member and simplified_member in simplified_line:
            return FeedName(display=member, nickname=member, clan_tag=None, confidence=0.85)
    return None


def _parse_feed_name(text: str) -> FeedName | None:
    cleaned = _cleanup_feed_text(text)
    if not cleaned:
        return None

    clan_tag = None
    tag_match = re.search(r"[\[\(【]\s*([^\]\)】]{1,6})\s*[\]\)】]", cleaned)
    if tag_match is not None:
        clan_tag = _cleanup_clan_tag(tag_match.group(1))
        cleaned = (cleaned[: tag_match.start()] + " " + cleaned[tag_match.end() :]).strip()

    cleaned = re.sub(r"^[Lし]\s+(?=[\w\u3040-\u30ff\u3400-\u9fff\u0400-\u04ff])", "", cleaned)
    nickname = _best_name_fragment(cleaned)
    if nickname is None:
        return None
    display = f"[{clan_tag}] {nickname}" if clan_tag else nickname
    confidence = 0.78 if clan_tag else 0.62
    return FeedName(display=display, nickname=nickname, clan_tag=clan_tag, confidence=confidence)


def _cleanup_feed_text(text: str) -> str:
    text = normalize_ocr_lines(text)
    text = text.replace("｜", "|").replace("：", ":")
    text = re.sub(r"[•●○◯◎◇◆□■△▲▽▼]+", " ", text)
    text = re.sub(r"\b(?:xp|score|assist|damage|armor|plate)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(" -:|/\\.,;")


def _cleanup_clan_tag(tag: str) -> str | None:
    tag = re.sub(r"\s+", "", tag.strip())
    tag = tag.replace("し", "L")
    tag = re.sub(r"[^\w\u3040-\u30ff\u3400-\u9fff\u0400-\u04ff-]", "", tag, flags=re.UNICODE)
    if 1 <= len(tag) <= 6:
        return tag
    return None


def _best_name_fragment(text: str) -> str | None:
    cleaned = re.sub(r"[^\w\s\u3040-\u30ff\u3400-\u9fff\u0400-\u04ff-]", " ", text, flags=re.UNICODE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" _-")
    candidates = [part.strip(" _-") for part in re.split(r"\s{2,}|[|:;]", cleaned) if part.strip(" _-")]
    candidates.append(cleaned)
    valid = [candidate for candidate in candidates if _looks_like_nickname(candidate)]
    if not valid:
        return None
    return max(valid, key=lambda value: (len(_simplify(value)), -len(value)))


def _looks_like_nickname(candidate: str) -> bool:
    if len(candidate) < 2 or len(candidate) > 16:
        return False
    if candidate[0] in {" ", "_"} or candidate[-1] in {" ", "_"}:
        return False
    if "  " in candidate or "__" in candidate:
        return False
    if re.search(r"[!@#$%^&*()?/\\\[\]]", candidate):
        return False
    meaningful = re.sub(r"[\s_-]", "", candidate)
    if len(meaningful) < 2:
        return False
    if re.fullmatch(r"(?:[\w\u3040-\u30ff\u3400-\u9fff\u0400-\u04ff]\s+){2,}[\w\u3040-\u30ff\u3400-\u9fff\u0400-\u04ff]", candidate):
        return False
    if len(set(re.findall(r"[A-Za-zА-Яа-я0-9\u3040-\u30ff\u3400-\u9fff]", candidate))) <= 1:
        return False
    return all(_is_allowed_name_char(char) for char in candidate)


def _is_allowed_name_char(char: str) -> bool:
    if char in {" ", "_", "-"}:
        return True
    category = unicodedata.category(char)
    return category[0] in {"L", "N"}


def _dominant_hex_color(image: np.ndarray) -> str | None:
    if image.size == 0:
        return None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = (saturation > 55) & (value > 80)
    pixels = image[mask]
    if len(pixels) == 0:
        return None
    bgr = np.median(pixels, axis=0).astype(int)
    return f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"


def _find_team_member_by_color(color_hex: str | None, team_profiles: list[dict[str, object]]) -> FeedName | None:
    if not color_hex:
        return None
    color_rgb = _hex_to_rgb(color_hex)
    if color_rgb is None:
        return None

    best_name = None
    best_distance = 999.0
    for profile in team_profiles:
        profile_color = profile.get("color_hex")
        profile_name = profile.get("name")
        if not isinstance(profile_color, str) or not isinstance(profile_name, str):
            continue
        profile_rgb = _hex_to_rgb(profile_color)
        if profile_rgb is None:
            continue
        hue_distance = _hue_distance(color_rgb, profile_rgb)
        if hue_distance > 0.12:
            continue
        distance = _rgb_distance(color_rgb, profile_rgb)
        score = distance + hue_distance * 260
        if score < best_distance:
            best_distance = score
            best_name = profile_name

    if best_name is None or best_distance > 110:
        return None
    return FeedName(display=best_name, nickname=best_name, clan_tag=None, confidence=max(0.45, 0.9 - best_distance / 180))


def _hex_to_rgb(color_hex: str) -> tuple[int, int, int] | None:
    match = re.fullmatch(r"#?([0-9a-fA-F]{6})", color_hex)
    if match is None:
        return None
    value = match.group(1)
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _rgb_distance(left: tuple[int, int, int], right: tuple[int, int, int]) -> float:
    return float(sum((a - b) ** 2 for a, b in zip(left, right)) ** 0.5)


def _hue_distance(left: tuple[int, int, int], right: tuple[int, int, int]) -> float:
    left_hue = _rgb_to_hue(left)
    right_hue = _rgb_to_hue(right)
    distance = abs(left_hue - right_hue)
    return min(distance, 1.0 - distance)


def _rgb_to_hue(color: tuple[int, int, int]) -> float:
    red, green, blue = [channel / 255 for channel in color]
    maximum = max(red, green, blue)
    minimum = min(red, green, blue)
    if maximum == minimum:
        return 0.0
    if maximum == red:
        hue = (green - blue) / (maximum - minimum)
    elif maximum == green:
        hue = 2.0 + (blue - red) / (maximum - minimum)
    else:
        hue = 4.0 + (red - green) / (maximum - minimum)
    return (hue / 6.0) % 1.0


def _relation_for(actor: str | None, target: str | None, team_members: list[str]) -> str:
    actor_in_team = _name_in_team(actor, team_members)
    target_in_team = _name_in_team(target, team_members)
    if actor_in_team and not target_in_team:
        return "team_did"
    if target_in_team and not actor_in_team:
        return "team_received"
    if actor_in_team and target_in_team:
        return "team_internal_or_ambiguous"
    return "unknown"


def _name_in_team(name: str | None, team_members: list[str]) -> bool:
    if not name:
        return False
    simplified_name = _simplify(name)
    return any(simplified_name == _simplify(member) for member in team_members)


def _simplify(value: str) -> str:
    return re.sub(r"[\W_]+", "", value, flags=re.UNICODE).lower()
