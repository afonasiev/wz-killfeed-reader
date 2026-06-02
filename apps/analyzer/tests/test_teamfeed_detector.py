import cv2
import numpy as np

from warzone_analyzer.models import AnalyzerConfig, OcrConfig, Region
from warzone_analyzer.ocr import OcrResult
from warzone_analyzer.output import AnalyzerOutput
from warzone_analyzer.teamfeed import TeamFeedDetector, _extract_feed_rows, parse_visual_feed_line
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
    cv2.putText(image, "Enemy", (190, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 230), 2, cv2.LINE_AA)

    assert _extract_feed_rows(image)

    config = AnalyzerConfig(
        ocr=OcrConfig(save_crops=True),
        regions={"team_feed": Region(x=0, y=0, width=1, height=1)},
    )
    detector = TeamFeedDetector(config, FakeFeedOcr(), AnalyzerOutput(tmp_path))
    first_frame = SampledFrame(image=image, frame_index=1, timestamp_ms=0)
    second_frame = SampledFrame(image=image, frame_index=2, timestamp_ms=1200)

    assert detector.process(first_frame, team_members=[]) == []
    events = detector.process(second_frame, team_members=[])

    assert [event.type.value for event in events] == ["kill"]
    assert events[0].source == "team_feed_ocr"
    assert events[0].details["actor"] == "Sirius1or"
    assert events[0].details["target"] == "Enemy"
    assert events[0].details["relation"] == "unknown"


def test_visual_parser_uses_one_white_icon_for_kill():
    parsed = parse_visual_feed_line(
        full_text="",
        left_text="SquadMate",
        right_text="EnemyOne",
        team_members=["SquadMate"],
        team_profiles=None,
        actor_color_hex="#00d250",
        visual_event_type=None,
        white_icon_count=1,
        row_confidence=0.7,
        row_y=24,
        row_crop_path=None,
    )

    assert parsed is not None
    assert parsed.event_type.value == "kill"
    assert parsed.actor.nickname == "SquadMate"
    assert parsed.target.nickname == "EnemyOne"
    assert parsed.evidence["white_icon_count"] == 1


def test_visual_parser_uses_two_white_icons_for_knock():
    parsed = parse_visual_feed_line(
        full_text="",
        left_text="SquadMate",
        right_text="EnemyTwo",
        team_members=["SquadMate"],
        team_profiles=None,
        actor_color_hex="#00d250",
        visual_event_type=None,
        white_icon_count=2,
        row_confidence=0.7,
        row_y=24,
        row_crop_path=None,
    )

    assert parsed is not None
    assert parsed.event_type.value == "knock"
    assert parsed.target.nickname == "EnemyTwo"


def test_visual_parser_falls_back_to_team_color_when_actor_ocr_is_empty():
    parsed = parse_visual_feed_line(
        full_text="",
        left_text="",
        right_text="EnemyRed",
        team_members=["SquadMate"],
        team_profiles=[{"name": "SquadMate", "color_hex": "#00d250"}],
        actor_color_hex="#00d050",
        visual_event_type=None,
        white_icon_count=1,
        row_confidence=0.7,
        row_y=24,
        row_crop_path=None,
    )

    assert parsed is not None
    assert parsed.event_type.value == "kill"
    assert parsed.actor.nickname == "SquadMate"
    assert parsed.evidence["team_member_profile_candidate"] == "SquadMate"


def test_enemy_name_candidates_use_red_crop_result():
    parsed = parse_visual_feed_line(
        full_text="",
        left_text="SquadMate",
        right_text="",
        team_members=["SquadMate"],
        team_profiles=None,
        actor_color_hex="#00d250",
        visual_event_type=None,
        white_icon_count=1,
        row_confidence=0.7,
        row_y=24,
        row_crop_path=None,
        enemy_name_candidates=["RedEnemy"],
        enemy_crop_path="debug_crops/team_feed/enemy.jpg",
        red_bbox=(100, 4, 150, 18),
    )

    assert parsed is not None
    assert parsed.target.nickname == "RedEnemy"
    assert parsed.evidence["enemy_red_crop"] == "debug_crops/team_feed/enemy.jpg"
    assert parsed.evidence["enemy_name_color"] == "red"
