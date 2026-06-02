from __future__ import annotations

from collections import Counter
from pathlib import Path

from .detectors import FightActivityDetector
from .metadata import MatchIdDetector, TeamDetector
from .models import AnalyzerConfig, AnalyzerEvent, AnalyzerSummary, EventType, FightSegment, MatchState, fight_uid
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
    all_events: list[AnalyzerEvent] = []
    fights: list[FightSegment] = []
    open_fight: FightSegment | None = None
    pending_action_events: list[AnalyzerEvent] = []
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
                region = config.regions.get("center_combat")
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
                    pending_action_events=pending_action_events,
                )
                all_events.append(event)
                event_count += 1
    finally:
        if async_ocr is not None:
            async_ocr.close()

    if open_fight is not None:
        _attach_pending_actions(open_fight, pending_action_events, config.fight_detection.action_attach_tolerance_ms)
    if open_fight is not None:
        if _finalize_fight(open_fight, config):
            fights.append(open_fight)

    for event in all_events:
        output.write_event(event)
    output.write_fights(fights)
    action_counts, team_action_summary = _summarize_actions(fights)
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
        actions=sum(action_counts.values()),
        action_counts=dict(action_counts),
        team_action_summary=team_action_summary,
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
    pending_action_events: list[AnalyzerEvent],
) -> FightSegment | None:
    if event.type == EventType.FIGHT_STARTED:
        if open_fight is not None:
            if _finalize_fight(open_fight, config):
                fights.append(open_fight)

        fight = FightSegment(
            fight_id=len(fights) + 1,
            started_at_ms=event.timestamp_ms,
            start_frame_index=event.frame_index,
            warzone_match_id=warzone_match_id,
            state=state.value,
            evidence={"start": event.details},
            start_debug_frame=event.debug_frame,
        )
        _attach_pending_actions(fight, pending_action_events, config.fight_detection.action_attach_tolerance_ms)
        return fight

    if event.type == EventType.FIGHT_ENDED and open_fight is not None:
        if warzone_match_id and open_fight.warzone_match_id is None:
            open_fight.warzone_match_id = warzone_match_id
            _sync_fight_action_identity(open_fight)
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
        _sync_fight_action_identity(open_fight)

    if _is_action_event(event):
        if open_fight is not None:
            _append_action(open_fight, event)
        elif _append_to_recent_fight(fights, event, config.fight_detection.action_attach_tolerance_ms):
            return open_fight
        else:
            pending_action_events.append(event)
            _trim_pending_actions(pending_action_events, event.timestamp_ms, config.fight_detection.action_attach_tolerance_ms)

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


def _is_action_event(event: AnalyzerEvent) -> bool:
    return event.type in {EventType.KILL, EventType.KNOCK, EventType.DEATH, EventType.TEAM_FEED_EVENT}


def _attach_pending_actions(fight: FightSegment, pending_action_events: list[AnalyzerEvent], tolerance_ms: int) -> None:
    remaining = []
    for action_event in pending_action_events:
        if fight.started_at_ms - tolerance_ms <= action_event.timestamp_ms <= fight.started_at_ms + tolerance_ms:
            _append_action(fight, action_event)
        else:
            remaining.append(action_event)
    pending_action_events[:] = remaining


def _append_to_recent_fight(fights: list[FightSegment], event: AnalyzerEvent, tolerance_ms: int) -> bool:
    if not fights:
        return False
    fight = fights[-1]
    ended_at_ms = fight.ended_at_ms
    if ended_at_ms is None or event.timestamp_ms > ended_at_ms + tolerance_ms:
        return False
    if event.timestamp_ms < fight.started_at_ms - tolerance_ms:
        return False
    _append_action(fight, event)
    return True


def _append_action(fight: FightSegment, event: AnalyzerEvent) -> None:
    uid = fight_uid(fight.warzone_match_id, fight.fight_id)
    event.details["fight_id"] = fight.fight_id
    event.details["fight_uid"] = uid
    event.details["warzone_match_id"] = fight.warzone_match_id
    action = _event_to_action(fight, event)
    dedupe_key = (
        action["type"],
        action.get("team_member"),
        action.get("target_enemy"),
        int(event.timestamp_ms) // 3000,
    )
    for existing in fight.actions:
        existing_key = (
            existing["type"],
            existing.get("team_member"),
            existing.get("target_enemy"),
            int(existing["timestamp_ms"]) // 3000,
        )
        if existing_key == dedupe_key:
            return
    action["action_id"] = f"{uid}:action:{len(fight.actions) + 1}"
    fight.actions.append(action)


def _event_to_action(fight: FightSegment, event: AnalyzerEvent) -> dict[str, object]:
    details = event.details
    evidence = details.get("evidence") if isinstance(details.get("evidence"), dict) else {}
    team_member = details.get("actor")
    target_enemy = details.get("target")
    if details.get("relation") == "team_received":
        team_member = details.get("target")
        target_enemy = details.get("actor")

    return {
        "action_id": "",
        "fight_uid": fight_uid(fight.warzone_match_id, fight.fight_id),
        "fight_id": fight.fight_id,
        "warzone_match_id": fight.warzone_match_id,
        "timestamp_ms": event.timestamp_ms,
        "frame_index": event.frame_index,
        "type": event.type.value,
        "action_kind": details.get("action_kind") or evidence.get("action_kind"),
        "team_member": team_member,
        "team_member_color_hex": details.get("actor_color_hex") or evidence.get("actor_color_hex"),
        "team_member_profile_candidate": evidence.get("team_member_profile_candidate"),
        "target_enemy": target_enemy,
        "enemy_name_color": "red" if target_enemy else None,
        "relation": details.get("relation"),
        "confidence": event.confidence,
        "raw_text": details.get("raw_text"),
        "evidence": evidence,
        "debug_frame": event.debug_frame,
        "crop": details.get("crop") or evidence.get("row_crop"),
        "needs_review": bool(evidence.get("needs_review") or not team_member or (event.type in {EventType.KILL, EventType.KNOCK} and not target_enemy)),
    }


def _trim_pending_actions(pending_action_events: list[AnalyzerEvent], timestamp_ms: int, tolerance_ms: int) -> None:
    pending_action_events[:] = [
        event for event in pending_action_events if timestamp_ms - event.timestamp_ms <= tolerance_ms
    ]


def _sync_fight_action_identity(fight: FightSegment) -> None:
    uid = fight_uid(fight.warzone_match_id, fight.fight_id)
    for index, action in enumerate(fight.actions, start=1):
        action["fight_uid"] = uid
        action["warzone_match_id"] = fight.warzone_match_id
        action["action_id"] = f"{uid}:action:{index}"


def _summarize_actions(fights: list[FightSegment]) -> tuple[Counter[str], dict[str, dict[str, int]]]:
    action_counts: Counter[str] = Counter()
    team_summary: dict[str, Counter[str]] = {}
    for fight in fights:
        for action in fight.actions:
            action_type = str(action.get("type") or "unknown")
            action_counts[action_type] += 1
            member = str(action.get("team_member") or "unknown_team_member")
            team_summary.setdefault(member, Counter())[action_type] += 1
            team_summary[member]["total"] += 1
    return action_counts, {member: dict(counts) for member, counts in team_summary.items()}
