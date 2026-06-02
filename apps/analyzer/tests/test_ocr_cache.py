import time

import numpy as np

from warzone_analyzer.models import AnalyzerConfig, OcrConfig
from warzone_analyzer.ocr import AsyncCachedOcr, OcrResult


class FakeOcr:
    def __init__(self) -> None:
        self.calls = 0

    def read_text(self, image, mode="text", cache_key=None):
        self.calls += 1
        time.sleep(0.02)
        value = str(int(image.mean()))
        return OcrResult(text=value, normalized=value, confidence=1.0)


def test_async_cached_ocr_reuses_stable_roi_and_resubmits_changed_roi():
    engine = FakeOcr()
    config = AnalyzerConfig(
        ocr=OcrConfig(
            async_enabled=True,
            worker_threads=1,
            max_pending_tasks=4,
            cache_mse_threshold=0.0001,
        )
    )
    ocr = AsyncCachedOcr(engine, config)
    try:
        first = np.zeros((24, 24, 3), dtype=np.uint8)
        changed = np.full((24, 24, 3), 255, dtype=np.uint8)

        assert ocr.read_text(first, cache_key="roi").normalized == ""
        time.sleep(0.04)
        assert ocr.read_text(first, cache_key="roi").normalized == "0"
        assert ocr.read_text(first, cache_key="roi").normalized == "0"
        assert engine.calls == 1

        assert ocr.read_text(changed, cache_key="roi").normalized == "0"
        time.sleep(0.04)
        assert ocr.read_text(changed, cache_key="roi").normalized == "255"
        assert engine.calls == 2
    finally:
        ocr.close()
