# anom_hunter

Scores a local folder of WAV files with the published protected model
`brahma_ca_03252_anomaly`, writes merge-safe decision sidecars next to each
file, and saves CA evolution render bundles viewable in the app.

## Requirements

- Python 3.10+
- `identdynamics` SDK with a valid API key (`API_KEY` in `hp_games/hp_ca.py`)
- `brahma_ca_03252_anomaly` published and accessible to the configured token
- `brahma_cellular` package (Pipeline, WolframRule, AnomalyDetectionRule, GroupingRule)

## Usage

```
python anom_hunter.py <folder> [options]
```

Run from the `anom_hunter_294502/` directory or any location — the script
resolves `hp_games/` from its own location automatically.

### Examples

```bash
# Score everything in a folder with defaults
python anom_hunter.py /data/audio

# Custom frequency band and time resolution
python anom_hunter.py /data/audio --fmin 500 --fmax 8000 --delta-t 0.005

# Score only the first 5 files under 200 MB, skip writing sidecars
python anom_hunter.py /data/audio --limit 5 --max-size-mb 200 --dry-run

# Decisions only — skip CA evolution render bundles
python anom_hunter.py /data/audio --no-render
```

## CLI Flags

| Flag | Default | Description |
|---|---|---|
| `folder` | *(required)* | Local folder of WAV files to process |
| `--fmin HZ` | `1000` | Lower frequency band edge in Hz |
| `--fmax HZ` | `8000` | Upper frequency band edge in Hz |
| `--delta-t SECONDS` | `0.01` | Target time resolution between STFT frames; hop = round(delta_t × sample_rate) |
| `--threshold` | `0.65` | Detection score threshold in [0, 1] |
| `--steps` | `4` | CA evolution steps for the render bundle |
| `--evolve-steps` | *(same as --steps)* | CA generations to render (override --steps for render only) |
| `--limit N` | *(all)* | Process at most N files that are under the size cap |
| `--max-size-mb MB` | *(no cap)* | Skip files larger than this many MB |
| `--output-root PATH` | *(random)* | Explicit output folder; defaults to `anom_hunter_out<rand>` |
| `--no-render` | — | Skip writing CA evolution bundles (decisions only) |
| `--dry-run` | — | Score and render but do not write decision sidecars |

## Output

Each invocation creates an output folder (e.g. `anom_hunter_out3a7f1c2d/`) containing:

```
anom_hunter_out<rand>/
  index.json              — summary of all files processed
  <stem>/                 — per-file CA evolution render bundle
    frames/
    manifest.json
```

Decision sidecars (`.decisions.json`) are written **next to the source audio
files** in the input folder, not in the output folder. Multiple models
accumulate in the same sidecar — existing records from other models are
preserved on each run.

## Detection Format

Each detection in the sidecar includes:

```json
{
  "dt": 1.23,
  "end_sec": 1.45,
  "signature": "brahma_ca_03252_anomaly",
  "decision": "detection",
  "reason": "ca band detection",
  "frame": 123,
  "active_freq": "1000-8000 Hz",
  "fmin_hz": 1000.0,
  "fmax_hz": 8000.0
}
```
