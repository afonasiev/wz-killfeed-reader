from __future__ import annotations

import cv2
import numpy as np

from .models import AnalyzerConfig, AnalyzerEvent, EventType
from .regions import crop_region
from .video import SampledFrame


class FightActivityDetector:
    """Coarse fight boundary detector based on visual activity in the combat region."""

    def __init__(self, config: AnalyzerConfig) -> None:
        self._config = config
        self._previous_gray: np.ndarray | None = None
        self._active_since_ms: int | None = None
        self._last_active_ms: int | None = None
        self._in_fight = False

    def process(self, sampled_frame: SampledFrame) -> list[AnalyzerEvent]:
        score = self._motion_score(sampled_frame.image)
        threshold = self._config.fight_detection.motion_threshold
        is_active = score >= threshold
        events: list[AnalyzerEvent] = []

        if is_active:
            if self._active_since_ms is None:
                self._active_since_ms = sampled_frame.timestamp_ms
            self._last_active_ms = sampled_frame.timestamp_ms
        elif not self._in_fight:
            self._active_since_ms = None

        if not self._in_fight and self._active_since_ms is not None:
            active_for = (sampled_frame.timestamp_ms - self._active_since_ms) / 1000
            if active_for >= self._config.fight_detection.min_active_seconds:
                self._in_fight = True
                events.append(
                    self._event(
                        EventType.FIGHT_STARTED,
                        sampled_frame,
                        confidence=min(score / threshold, 1.0),
                        details={"motion_score": score, "active_for_seconds": active_for},
                    )
                )

        if self._in_fight and self._last_active_ms is not None:
            idle_for = (sampled_frame.timestamp_ms - self._last_active_ms) / 1000
            if idle_for >= self._config.fight_detection.idle_end_seconds:
                self._in_fight = False
                self._active_since_ms = None
                self._last_active_ms = None
                events.append(
                    self._event(
                        EventType.FIGHT_ENDED,
                        sampled_frame,
                        confidence=1.0,
                        details={"motion_score": score, "idle_for_seconds": idle_for},
                    )
                )

        return events

    def force_close(self, sampled_frame: SampledFrame, reason: str) -> AnalyzerEvent | None:
        if not self._in_fight:
            self._active_since_ms = None
            self._last_active_ms = None
            return None

        self._in_fight = False
        self._active_since_ms = None
        self._last_active_ms = None
        return self._event(
            EventType.FIGHT_ENDED,
            sampled_frame,
            confidence=0.75,
            details={"reason": reason},
        )

    def _motion_score(self, frame: np.ndarray) -> float:
        region = self._config.regions.get("center_combat")
        target = crop_region(frame, region) if region else frame
        gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if self._previous_gray is None:
            self._previous_gray = gray
            return 0.0

        diff = cv2.absdiff(self._previous_gray, gray)
        self._previous_gray = gray
        changed_pixels = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        return float(np.count_nonzero(changed_pixels) / changed_pixels.size)

    @staticmethod
    def _event(
        event_type: EventType,
        sampled_frame: SampledFrame,
        confidence: float,
        details: dict[str, object],
    ) -> AnalyzerEvent:
        return AnalyzerEvent(
            type=event_type,
            timestamp_ms=sampled_frame.timestamp_ms,
            frame_index=sampled_frame.frame_index,
            confidence=confidence,
            source="motion_detector",
            details=details,
        )
