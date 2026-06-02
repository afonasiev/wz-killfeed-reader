import numpy as np

from warzone_analyzer.detectors import FightActivityDetector
from warzone_analyzer.models import AnalyzerConfig, AnalyzerEvent, EventType, FightDetectionConfig, MatchState
from warzone_analyzer.pipeline import _summarize_actions, _update_fights
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


def test_action_inside_open_fight_is_added_to_fight_actions():
    config = AnalyzerConfig(fight_detection=FightDetectionConfig(min_duration_seconds=0))
    fights = []
    pending = []
    open_fight = _update_fights(
        fights=fights,
        open_fight=None,
        event=AnalyzerEvent(EventType.FIGHT_STARTED, 1000, 10, 0.9, "test"),
        config=config,
        warzone_match_id="1234567890123456",
        state=MatchState.GAMEPLAY,
        pending_action_events=pending,
    )

    open_fight = _update_fights(
        fights=fights,
        open_fight=open_fight,
        event=AnalyzerEvent(
            EventType.KILL,
            1500,
            15,
            0.8,
            "team_feed_ocr",
            details={"actor": "SquadMate", "target": "Enemy", "relation": "team_did", "evidence": {"actor_color_hex": "#00ff00"}},
        ),
        config=config,
        warzone_match_id="1234567890123456",
        state=MatchState.GAMEPLAY,
        pending_action_events=pending,
    )

    assert open_fight is not None
    assert open_fight.to_dict()["fight_uid"] == "1234567890123456:1"
    assert len(open_fight.actions) == 1
    assert open_fight.actions[0]["action_id"] == "1234567890123456:1:action:1"
    assert open_fight.actions[0]["team_member"] == "SquadMate"
    assert open_fight.actions[0]["target_enemy"] == "Enemy"


def test_pending_action_near_fight_start_is_attached():
    config = AnalyzerConfig(fight_detection=FightDetectionConfig(action_attach_tolerance_ms=3000, min_duration_seconds=0))
    pending = [
        AnalyzerEvent(
            EventType.KNOCK,
            900,
            9,
            0.75,
            "team_feed_ocr",
            details={"actor": "SquadMate", "target": "Enemy", "relation": "team_did", "evidence": {}},
        )
    ]

    open_fight = _update_fights(
        fights=[],
        open_fight=None,
        event=AnalyzerEvent(EventType.FIGHT_STARTED, 2000, 20, 0.9, "test"),
        config=config,
        warzone_match_id="match-1",
        state=MatchState.GAMEPLAY,
        pending_action_events=pending,
    )

    assert open_fight is not None
    assert pending == []
    assert len(open_fight.actions) == 1
    assert open_fight.actions[0]["fight_uid"] == "match-1:1"
    assert open_fight.actions[0]["type"] == "knock"


def test_action_summary_counts_by_member():
    config = AnalyzerConfig(fight_detection=FightDetectionConfig(min_duration_seconds=0))
    pending = []
    open_fight = _update_fights(
        fights=[],
        open_fight=None,
        event=AnalyzerEvent(EventType.FIGHT_STARTED, 1000, 10, 0.9, "test"),
        config=config,
        warzone_match_id="match-1",
        state=MatchState.GAMEPLAY,
        pending_action_events=pending,
    )
    assert open_fight is not None
    _update_fights(
        fights=[],
        open_fight=open_fight,
        event=AnalyzerEvent(EventType.KILL, 1500, 15, 0.8, "team_feed_ocr", details={"actor": "SquadMate", "target": "Enemy", "relation": "team_did", "evidence": {}}),
        config=config,
        warzone_match_id="match-1",
        state=MatchState.GAMEPLAY,
        pending_action_events=pending,
    )

    action_counts, team_summary = _summarize_actions([open_fight])

    assert dict(action_counts) == {"kill": 1}
    assert team_summary == {"SquadMate": {"kill": 1, "total": 1}}
