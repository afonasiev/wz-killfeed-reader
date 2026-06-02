from __future__ import annotations

import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Protocol

import cv2
import numpy as np

from .models import AnalyzerConfig


@dataclass
class OcrResult:
    text: str
    normalized: str
    confidence: float


class OcrReader(Protocol):
    def read_text(self, image: np.ndarray, mode: str = "text", cache_key: str | None = None) -> OcrResult:
        ...


@dataclass
class _CachedOcrState:
    fingerprint: np.ndarray | None = None
    result: OcrResult | None = None
    pending: Future[OcrResult] | None = None
    pending_fingerprint: np.ndarray | None = None


class TesseractOcr:
    def __init__(self, config: AnalyzerConfig) -> None:
        self._config = config

    def read_text(self, image: np.ndarray, mode: str = "text", cache_key: str | None = None) -> OcrResult:
        if not self._config.ocr.enabled:
            return OcrResult(text="", normalized="", confidence=0.0)

        prepared = _prepare_for_ocr(image, mode)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            cv2.imwrite(str(temp_path), prepared)
            command = [
                self._config.ocr.tesseract_cmd,
                str(temp_path),
                "stdout",
                "-l",
                self._config.ocr.languages,
                "--psm",
                _psm_for_mode(mode),
            ]
            if mode == "match_id":
                command.extend(["-c", "tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"])

            completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=8)
            text = completed.stdout.strip()
            normalized = normalize_match_id(text) if mode == "match_id" else normalize_ocr_lines(text)
            confidence = 0.0 if completed.returncode else min(len(normalized) / 24, 1.0)
            return OcrResult(text=text, normalized=normalized, confidence=confidence)
        except (OSError, subprocess.SubprocessError):
            return OcrResult(text="", normalized="", confidence=0.0)
        finally:
            temp_path.unlink(missing_ok=True)


class AsyncCachedOcr:
    def __init__(self, engine: OcrReader, config: AnalyzerConfig) -> None:
        self._engine = engine
        self._config = config
        self._executor = ThreadPoolExecutor(max_workers=max(config.ocr.worker_threads, 1))
        self._states: dict[str, _CachedOcrState] = {}
        self._lock = Lock()

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)

    def read_text(self, image: np.ndarray, mode: str = "text", cache_key: str | None = None) -> OcrResult:
        if not self._config.ocr.enabled:
            return OcrResult(text="", normalized="", confidence=0.0)
        if image.size == 0:
            return OcrResult(text="", normalized="", confidence=0.0)

        key = cache_key or mode
        fingerprint = _ocr_fingerprint(image)

        with self._lock:
            state = self._states.setdefault(key, _CachedOcrState())
            self._collect_completed(state)

            if state.result is not None and state.fingerprint is not None:
                if _fingerprint_mse(state.fingerprint, fingerprint) <= self._config.ocr.cache_mse_threshold:
                    return state.result

            if state.pending is not None:
                return state.result or OcrResult(text="", normalized="", confidence=0.0)

            if self._pending_count() >= self._config.ocr.max_pending_tasks:
                return state.result or OcrResult(text="", normalized="", confidence=0.0)

            state.pending_fingerprint = fingerprint
            state.pending = self._executor.submit(self._engine.read_text, image.copy(), mode, None)
            return state.result or OcrResult(text="", normalized="", confidence=0.0)

    @staticmethod
    def _collect_completed(state: _CachedOcrState) -> None:
        if state.pending is None or not state.pending.done():
            return
        try:
            state.result = state.pending.result()
            state.fingerprint = state.pending_fingerprint
        except Exception:
            state.result = OcrResult(text="", normalized="", confidence=0.0)
            state.fingerprint = state.pending_fingerprint
        finally:
            state.pending = None
            state.pending_fingerprint = None

    def _pending_count(self) -> int:
        return sum(1 for state in self._states.values() if state.pending is not None)


class StableTextVote:
    def __init__(self) -> None:
        self._counter: Counter[str] = Counter()

    def add(self, value: str) -> None:
        if value:
            self._counter[value] += 1

    def best(self) -> str | None:
        if not self._counter:
            return None
        return self._counter.most_common(1)[0][0]

    def count(self, value: str) -> int:
        return self._counter[value]

    def values(self, min_count: int = 1) -> list[str]:
        return [value for value, count in self._counter.most_common() if count >= min_count]


def normalize_match_id(text: str) -> str:
    return re.sub(r"[^0-9]", "", text)


def normalize_ocr_lines(text: str) -> str:
    lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def _psm_for_mode(mode: str) -> str:
    if mode in {"match_id", "feed_line", "feed_name"}:
        return "7"
    if mode == "feed_raw":
        return "8"
    if mode == "feed_sparse":
        return "11"
    return "6"


def _prepare_for_ocr(image: np.ndarray, mode: str) -> np.ndarray:
    scale = 6 if mode in {"feed_line", "feed_name", "feed_raw"} else 4 if mode == "match_id" else 3
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if mode == "feed_raw":
        return resized
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    if mode == "match_id":
        gray = cv2.convertScaleAbs(gray, alpha=2.4, beta=10)
        return cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)[1]
    if mode in {"feed_line", "feed_name", "feed_sparse"}:
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        colored_text = (saturation > 45) & (value > 70)
        white_text = (saturation < 80) & (value > 170)
        mask = (colored_text | white_text).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        prepared = np.full(mask.shape, 255, dtype=np.uint8)
        prepared[mask > 0] = 0
        return cv2.medianBlur(prepared, 3)
    gray = cv2.convertScaleAbs(gray, alpha=1.8, beta=8)
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def _ocr_fingerprint(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32) / 255.0


def _fingerprint_mse(previous: np.ndarray, current: np.ndarray) -> float:
    if previous.shape != current.shape:
        return 1.0
    delta = previous - current
    return float(np.mean(delta * delta))
