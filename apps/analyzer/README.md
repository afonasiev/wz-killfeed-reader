# Analyzer

The analyzer is responsible for reading Warzone video sources and producing normalized events.

Supported source types are delegated to OpenCV/FFmpeg:

- saved files, for example `/data/input/match.mp4`;
- RTMP URLs;
- HTTP/HLS URLs if the local OpenCV build supports them;
- camera/capture devices later, if needed.

## Current Pipeline

```text
source -> sampled frames -> coarse fight detector -> events.jsonl/debug_frames
```

The first detector is intentionally simple: it detects fight start/end from sustained visual activity in the configured `center_combat` region. It is meant to prove the ingestion and event pipeline before adding Warzone-specific OCR/CV.

The current pipeline also adds a lightweight OCR/state layer:

- match state detection gates fight events so lobby/loading/cinematic frames do not start fights;
- Tesseract OCR reads the unique Warzone ID from the lower-left `match_id` region;
- Tesseract OCR scans squad regions for known team nicknames;
- OCR crops are saved under `debug_crops/` for manual tuning.

Team detection is dynamic. It accepts 1-4 visible squad members and validates names as Unicode Call of Duty nicknames rather than relying on a fixed roster. Leading clan tags in square brackets are stripped before validation.

Next detectors should plug into the same frame loop:

- killfeed OCR detector: `kill`, `knock`;
- squad state detector: `teammate_down`, `death`, `teammate_revived`;
- HUD stats detector: current kills/damage when visible;
- fight aggregator: merges low-level events into one fight window.

## Commands

```bash
python -m warzone_analyzer analyze /path/to/video.mp4 --output-dir /tmp/warzone-run
python -m warzone_analyzer analyze "rtmp://localhost/live/warzone" --output-dir /tmp/warzone-live
```

Docker:

```bash
docker compose run --rm analyzer analyze /data/input/match.mp4 --output-dir /data/output/match
docker compose run --rm analyzer analyze "/data/test_fragments/[RUS]Call of Duty warzone.mp4" --start-at 600 --duration 180 --output-dir /data/output/window-600
```

## Tuning

Edit `config/default.yml`:

- `sampling.fps` controls how many frames per second are analyzed;
- `fight_detection.motion_threshold` controls sensitivity;
- `fight_detection.min_active_seconds` prevents one-frame false starts;
- `fight_detection.idle_end_seconds` controls when a fight is considered over;
- `regions.*` contains normalized screen regions for 16:9 captures.
- `ocr.min_match_id_votes` controls how many repeated OCR reads are required before a match ID is accepted.
- `ocr.async_enabled` moves expensive OCR calls off the frame loop and returns cached stable ROI results while new text is processed.
- `ocr.cache_mse_threshold` controls how much a cropped text region may change before OCR is scheduled again.
- `ocr.worker_threads` and `ocr.max_pending_tasks` bound OCR CPU use and queue growth for stream processing.

For long recordings, use `--start-at` and `--duration` while tuning. The analyzer writes both low-level `events.jsonl` and aggregated `fights.json`, because one source video can contain many separate fights.
