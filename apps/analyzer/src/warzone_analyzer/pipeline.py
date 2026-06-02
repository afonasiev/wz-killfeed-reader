from __future__ import annotations

from pathlib import Path

from .detectors import FightActivityDetector
from .metadata import MatchIdDetector, TeamDetector
from .models import AnalyzerConfig, AnalyzerEvent, AnalyzerSummary, EventType, FightSegment, MatchState
from .ocr import AsyncCachedOcr, OcrReader, TesseractOcr
from .output import AnalyzerOutput
from .state import MatchStateDetector
from .teamfeed import TeamFeedDetector
from .video import iter_sampled_frames, video_duration_ms


def analyze_source(
    source: str,
    output_dir: Path,
    config: AnalyzerConfig,
    start_at_seconds: float = 0.0,
    duration_seconds: float | None = None,
) -> AnalyzerSummary:
    output = AnalyzerOutput(output_dir)
    activity_detector = FightActivityDetector(config)
    state_detector = MatchStateDetector(config)
    base_ocr = TesseractOcr(config)
    async_ocr = AsyncCachedOcr(base_ocr, config) if config.ocr.async_enabled else None
    ocr: OcrReader = async_ocr or base_ocr
    match_id_detector = MatchIdDetector(config, ocr, output)
    team_detector = TeamDetector(config, ocr, output)
    team_feed_detector = TeamFeedDetector(config, ocr, output)
    sampled_frames = 0
    event_count = 0
    fights: list[FightSegment] = []
    open_fight: FightSegment | None = None
    current_state = MatchState.UNKNOWN
    state_counts: dict[str, int] = {}

    try:
        for sampled_frame in iter_sampled_frames(
            source,
            target_fps=config.sampling.fps,
            max_frames=config.sampling.max_frames,
            start_at_seconds=start_at_seconds,
            duration_seconds=duration_seconds,
        ):
            sampled_frames += 1
            detected_state, state_evidence = state_detector.detect(sampled_frame.image)
            state_counts[detected_state.value] = state_counts.get(detected_state.value, 0) + 1
            events: list[AnalyzerEvent] = []

            if detected_state != current_state:
                current_state = detected_state
                state_crop = None
                region = config.regions.get("gameplay_markers")
                if region is not None and config.ocr.save_crops:
                    from .regions import crop_region

                    state_crop = output.save_debug_crop(
                        crop_region(sampled_frame.image, region),
                        "state",
                        detected_state.value,
                        sampled_frame,
                    )
                events.append(
                    AnalyzerEvent(
                        type=EventType.STATE_CHANGED,
                        timestamp_ms=sampled_frame.timestamp_ms,
                        frame_index=sampled_frame.frame_index,
                        confidence=0.65,
                        source="match_state_detector",
                        details={"state": detected_state.value, "evidence": state_evidence, "crop": state_crop},
                    )
                )

            events.extend(match_id_detector.process(sampled_frame))
            events.extend(team_detector.process(sampled_frame, prefer_lobby=detected_state == MatchState.LOBBY))

            can_process_match_events = sampled_frame.timestamp_ms >= int(config.fight_detection.ignore_initial_seconds * 1000)
            if can_process_match_events and detected_state in {MatchState.GAMEPLAY, MatchState.SPECTATING_OR_DEAD}:
                events.extend(activity_detector.process(sampled_frame))
                events.extend(team_feed_detector.process(sampled_frame, team_detector.members, team_detector.profiles))
            elif can_process_match_events and detected_state == MatchState.UNKNOWN:
                events.extend(team_feed_detector.process(sampled_frame, team_detector.members, team_detector.profiles))
                close_event = activity_detector.force_close(sampled_frame, reason=f"state_changed_to_{detected_state.value}")
                if close_event is not None:
                    events.append(close_event)
            else:
                close_event = activity_detector.force_close(sampled_frame, reason=f"state_changed_to_{detected_state.value}")
                if close_event is not None:
                    events.append(close_event)

            should_save_periodic = (
                config.debug.save_every_n_sampled_frames > 0
                and sampled_frames % config.debug.save_every_n_sampled_frames == 0
            )
            if should_save_periodic:
                output.save_debug_frame(sampled_frame, "sample")

            for event in events:
                if config.debug.save_transition_frames:
                    event.debug_frame = output.save_debug_frame(sampled_frame, event.type.value)
                open_fight = _update_fights(
                    fights=fights,
                    open_fight=open_fight,
                    event=event,
                    config=config,
                    warzone_match_id=match_id_detector.best_match_id,
                    state=detected_state,
                )
                output.write_event(event)
                event_count += 1
    finally:
        if async_ocr is not None:
            async_ocr.close()

    if open_fight is not None:
        if _finalize_fight(open_fight, config):
            fights.append(open_fight)

    output.write_fights(fights)
    summary = AnalyzerSummary(
        input=source,
        output_dir=output_dir,
        sampled_frames=sampled_frames,
        events=event_count,
        fights=len(fights),
        duration_ms=video_duration_ms(source),
        warzone_match_id=match_id_detector.best_match_id,
        warzone_match_ids=match_id_detector.match_ids,
        team_members=team_detector.members,
        team_colors=team_detector.profiles,
        team_history=team_detector.history,
        state_counts=state_counts,
    )
    output.write_summary(summary)
    return summary


def _update_fights(
    fights: list[FightSegment],
    open_fight: FightSegment | None,
    event: AnalyzerEvent,
    config: AnalyzerConfig,
    warzone_match_id: str | None,
    state: MatchState,
) -> FightSegment | None:
    if event.type == EventType.FIGHT_STARTED:
        if open_fight is not None:
            if _finalize_fight(open_fight, config):
                fights.append(open_fight)

        return FightSegment(
            fight_id=len(fights) + 1,
            started_at_ms=event.timestamp_ms,
            start_frame_index=event.frame_index,
            warzone_match_id=warzone_match_id,
            state=state.value,
            evidence={"start": event.details},
            start_debug_frame=event.debug_frame,
        )

    if event.type == EventType.FIGHT_ENDED and open_fight is not None:
        if warzone_match_id and open_fight.warzone_match_id is None:
            open_fight.warzone_match_id = warzone_match_id
        open_fight.ended_at_ms = event.timestamp_ms
        open_fight.end_frame_index = event.frame_index
        open_fight.duration_ms = event.timestamp_ms - open_fight.started_at_ms
        open_fight.evidence["end"] = event.details
        open_fight.end_debug_frame = event.debug_frame
        if _finalize_fight(open_fight, config):
            fights.append(open_fight)
        return None

    if event.type == EventType.MATCH_ID_DETECTED and open_fight is not None:
        open_fight.warzone_match_id = str(event.details.get("warzone_match_id") or open_fight.warzone_match_id)

    return open_fight


def _finalize_fight(fight: FightSegment, config: AnalyzerConfig) -> bool:
    if fight.duration_ms is None:
        fight.needs_review = True
        fight.evidence["review_reason"] = "fight_not_closed"
        return True

    duration_seconds = fight.duration_ms / 1000
    if duration_seconds < config.fight_detection.min_duration_seconds:
        return False
    elif duration_seconds > config.fight_detection.review_after_seconds:
        fight.needs_review = True
        fight.evidence["review_reason"] = "too_long_without_killfeed_confirmation"
    return True
