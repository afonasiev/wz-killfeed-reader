from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class SampledFrame:
    image: np.ndarray
    frame_index: int
    timestamp_ms: int


class VideoOpenError(RuntimeError):
    pass


def iter_sampled_frames(
    source: str,
    target_fps: float,
    max_frames: int | None = None,
    start_at_seconds: float = 0.0,
    duration_seconds: float | None = None,
) -> Iterator[SampledFrame]:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise VideoOpenError(f"Could not open video source: {source}")

    native_fps = capture.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0:
        native_fps = target_fps

    if start_at_seconds > 0:
        capture.set(cv2.CAP_PROP_POS_MSEC, start_at_seconds * 1000)

    end_at_ms = None
    if duration_seconds is not None:
        end_at_ms = int((start_at_seconds + duration_seconds) * 1000)

    step = max(int(round(native_fps / target_fps)), 1)
    frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
    sampled_count = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            timestamp_ms = int(capture.get(cv2.CAP_PROP_POS_MSEC))
            if end_at_ms is not None and timestamp_ms > end_at_ms:
                break

            if frame_index % step == 0:
                yield SampledFrame(image=frame, frame_index=frame_index, timestamp_ms=timestamp_ms)
                sampled_count += 1
                if max_frames is not None and sampled_count >= max_frames:
                    break

            frame_index += 1
    finally:
        capture.release()


def video_duration_ms(source: str) -> int | None:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        return None

    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps <= 0 or frame_count <= 0:
            return None
        return int((frame_count / fps) * 1000)
    finally:
        capture.release()
