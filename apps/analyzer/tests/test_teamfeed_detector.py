import cv2
import numpy as np

from warzone_analyzer.models import AnalyzerConfig, OcrConfig, Region
from warzone_analyzer.ocr import OcrResult
from warzone_analyzer.output import AnalyzerOutput
from warzone_analyzer.teamfeed import TeamFeedDetector, _extract_feed_rows
from warzone_analyzer.video import SampledFrame


class FakeFeedOcr:
    def read_text(self, image, mode="text", cache_key=None):
        key = cache_key or ""
        if key.endswith(":left:name") or key.endswith(":left:raw"):
            return OcrResult(text="Sirius1or", normalized="Sirius1or", confidence=0.9)
        if key.endswith(":right:name") or key.endswith(":right:raw"):
            return OcrResult(text="Enemy", normalized="Enemy", confidence=0.9)
        return OcrResult(text="", normalized="", confidence=0.0)


def test_left_feed_emits_kill_without_team_members(tmp_path):
    image = np.zeros((240, 640, 3), dtype=np.uint8)
    y = 150
    cv2.putText(image, "Sirius1or", (35, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 80), 2, cv2.LINE_AA)
    cv2.line(image, (165, y - 7), (195, y - 7), (245, 245, 245), 3, cv2.LINE_AA)
    cv2.line(image, (178, y - 12), (190, y - 2), (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(image, "Enemy", (215, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 230), 2, cv2.LINE_AA)

    assert _extract_feed_rows(image)

    config = AnalyzerConfig(
        ocr=OcrConfig(save_crops=True),
        regions={"team_feed": Region(x=0, y=0, width=1, height=1)},
    )
    detector = TeamFeedDetector(config, FakeFeedOcr(), AnalyzerOutput(tmp_path))
    first_frame = SampledFrame(image=image, frame_index=1, timestamp_ms=0)
    second_frame = SampledFrame(image=image, frame_index=2, timestamp_ms=300)

    assert detector.process(first_frame, team_members=[]) == []
    events = detector.process(second_frame, team_members=[])

    assert [event.type.value for event in events] == ["kill"]
    assert events[0].source == "team_feed_ocr"
    assert events[0].details["actor"] == "Sirius1or"
    assert events[0].details["target"] == "Enemy"
    assert events[0].details["relation"] == "unknown"
