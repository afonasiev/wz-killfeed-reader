from __future__ import annotations

import cv2
import numpy as np

from .models import AnalyzerConfig, MatchState
from .regions import crop_region


class MatchStateDetector:
    def __init__(self, config: AnalyzerConfig) -> None:
        self._config = config

    def detect(self, frame: np.ndarray) -> tuple[MatchState, dict[str, object]]:
        lobby_party_bright = self._bright_ratio(frame, "lobby_party", 155)
        squad_hud_bright = self._bright_ratio(frame, "squad_hud", 155)
        match_id_bright = self._bright_ratio(frame, "match_id", 155)
        center_combat_bright = self._bright_ratio(frame, "center_combat", 155)
        center_combat_hot = self._bright_ratio(frame, "center_combat", 210)

        has_lobby_party = lobby_party_bright >= 0.02
        has_squad_hud = squad_hud_bright >= 0.025
        has_bright_menu_bottom = match_id_bright >= 0.18
        has_gameplay_activity = center_combat_bright >= 0.06 or center_combat_hot >= 0.015

        evidence = {
            "lobby_party_bright": lobby_party_bright,
            "squad_hud_bright": squad_hud_bright,
            "match_id_bright": match_id_bright,
            "center_combat_bright": center_combat_bright,
            "center_combat_hot": center_combat_hot,
        }

        if has_lobby_party and has_bright_menu_bottom:
            return MatchState.LOBBY, evidence
        if has_bright_menu_bottom:
            return MatchState.LOADING, evidence
        if has_squad_hud and has_gameplay_activity:
            return MatchState.GAMEPLAY, evidence
        if has_gameplay_activity and not has_lobby_party:
            return MatchState.SPECTATING_OR_DEAD, evidence
        if center_combat_bright >= 0.03 and not has_squad_hud:
            return MatchState.CINEMATIC, evidence
        return MatchState.UNKNOWN, evidence

    def _bright_ratio(self, frame: np.ndarray, region_name: str, brightness: int) -> float:
        region = self._config.regions.get(region_name)
        if region is None:
            return 0.0

        crop = crop_region(frame, region)
        if crop.size == 0:
            return 0.0

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(np.count_nonzero(gray > brightness) / gray.size)
