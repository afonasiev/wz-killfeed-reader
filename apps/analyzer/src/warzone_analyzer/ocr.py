from __future__ import annotations

import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .models import AnalyzerConfig


@dataclass
class OcrResult:
    text: str
    normalized: str
    confidence: float


class TesseractOcr:
    def __init__(self, config: AnalyzerConfig) -> None:
        self._config = config

    def read_text(self, image: np.ndarray, mode: str = "text") -> OcrResult:
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
                "7" if mode == "match_id" else "6",
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


def _prepare_for_ocr(image: np.ndarray, mode: str) -> np.ndarray:
    scale = 4 if mode == "match_id" else 3
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    if mode == "match_id":
        gray = cv2.convertScaleAbs(gray, alpha=2.4, beta=10)
        return cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)[1]
    gray = cv2.convertScaleAbs(gray, alpha=1.8, beta=8)
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
