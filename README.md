# Warzone Stream Parser

Monorepo for detecting Warzone match/fight events from recorded videos or live streams.

Current focus:

- read a saved video file or stream URL;
- sample frames at a controlled FPS;
- detect coarse fight start/end events;
- write machine-readable event output;
- save debug frames for later CV/OCR tuning.

Telegram bot and backend API are intentionally deferred until the analyzer pipeline is reliable.

## Layout

```text
apps/
  analyzer/          Python service/CLI for video and stream analysis
data/
  input/             Local video files, ignored by git
  output/            Analyzer results, ignored by git
```

## Quick Start

```bash
docker compose build analyzer
docker compose run --rm analyzer analyze /data/input/sample.mp4 --output-dir /data/output/sample
```

Or run locally:

```bash
cd apps/analyzer
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m warzone_analyzer analyze ../../data/input/sample.mp4 --output-dir ../../data/output/sample
```

For a stream, pass the stream URL instead of a file path:

```bash
python -m warzone_analyzer analyze "rtmp://localhost/live/warzone" --output-dir ../../data/output/live
```

## Output

The analyzer writes:

- `events.jsonl` - one JSON event per line;
- `fights.json` - detected fight windows;
- `summary.json` - aggregate run metadata;
- `debug_frames/` - sampled frames around detected transitions.
- `debug_crops/` - OCR/state crops grouped by `match_id`, `team`, and `state`.

## Working With Long Videos

For long recordings, analyze short windows first and tune the config before running the whole file:

```bash
docker compose run --rm analyzer analyze "/data/test_fragments/[RUS]Call of Duty warzone.mp4" \
  --start-at 600 \
  --duration 180 \
  --output-dir /data/output/warzone-window-600
```

When detection looks reasonable, run without `--duration` to process the full recording.

`summary.json` includes stable OCR metadata:

- `warzone_match_id` - last confirmed unique Warzone fight/match identifier;
- `warzone_match_ids` - all confirmed identifiers seen in the analyzed video;
- `team_members` - detected squad nicknames;
- `state_counts` - frame-sample counts by detected state.

Detected nicknames are validated as Call of Duty-style Unicode names:
2-16 characters; Unicode letters/numbers plus spaces and `_`; no leading/trailing or duplicated spaces/underscores; no `! @ # $ % ^ & * ( ) ? / \` or clan-tag brackets.
