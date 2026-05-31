import numpy as np

from warzone_analyzer.detectors import FightActivityDetector
from warzone_analyzer.models import AnalyzerConfig, FightDetectionConfig
from warzone_analyzer.video import SampledFrame


def test_fight_starts_after_sustained_activity():
    config = AnalyzerConfig(
        fight_detection=FightDetectionConfig(
            motion_threshold=0.01,
            min_active_seconds=1.0,
            idle_end_seconds=2.0,
        )
    )
    detector = FightActivityDetector(config)
    frames = [
        SampledFrame(image=np.zeros((32, 32, 3), dtype=np.uint8), frame_index=0, timestamp_ms=0),
        SampledFrame(image=np.full((32, 32, 3), 255, dtype=np.uint8), frame_index=1, timestamp_ms=500),
        SampledFrame(image=np.zeros((32, 32, 3), dtype=np.uint8), frame_index=2, timestamp_ms=1500),
    ]

    events = []
    for frame in frames:
        events.extend(detector.process(frame))

    assert [event.type.value for event in events] == ["fight_started"]

