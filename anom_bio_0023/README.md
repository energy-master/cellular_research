# anom_bio_0023

Cellular-automata detector tuned for **short, single-frame anomalies** in the
100–140 kHz band — the range where harbour porpoise (and similar small
cetacean) echolocation clicks live. A porpoise click is a ~130 µs narrowband
pulse, so at 1 ms/frame STFT resolution it typically occupies just one
frame — this pipeline is configured to fire on that.

The script walks either a single WAV or an entire folder, runs the CA
against each file, and writes a merge-safe `<base>.decisions.json` sidecar
next to each audio file so the webapp overlays detections on open.

## What makes it different

Compared to `hp_local_big_data_ca`, three things are dialled for
transient echoes:

| Stage | Standard `hp_ca` | `anom_bio_0023` |
|---|---|---|
| `GroupingRule.min_frame_span` | 2 | **1** |
| `GroupingRule.min_cells` | 5 | **1** |
| `GroupingRule.min_bin_span` | 1 | **1** |
| CA `steps` | 3 | **1** (low so single-frame hits don't smear) |
| Default band | 100–150 kHz | **100–140 kHz** |
| Default `delta-t` | 0.001 s | **0.001 s** |
| `min_sigma` (anomaly z-score) | 1.5 | **1.5** |
| Default rule chain | `WolframRule(90) → AnomalyDetectionRule` | **`SpectralFluxRule → AnomalyDetectionRule`** (`--rule flux_anomaly`) |

## Requirements

- Python 3.10+
- `identdynamics` SDK (with `fourier_for_path(desired_delta_t=…)` support)
- `brahma_cellular` (Pipeline, WolframRule, AnomalyDetectionRule, GroupingRule)
- Sibling `hp_games/` folder (script auto-adds it to `sys.path` for
  `hp_ca` helpers: `crop_fourier_band`, `render_evolution`,
  `write_decisions_sidecar`, `new_run_id`)

## Usage

```bash
python anom_bio_0023.py <target> [options]
```

`<target>` may be **either a folder of WAVs or a single WAV file**.

### Examples

```bash
# Score every WAV in a folder with defaults (100-140 kHz, 1 ms/frame)
python anom_bio_0023.py /data/hydrophone

# Score just one file
python anom_bio_0023.py /data/hydrophone/20240712_143201.wav

# Narrow the band and boost sensitivity
python anom_bio_0023.py /data/hydrophone --fmin 110000 --fmax 130000 --min-sigma 1.5

# Smoke test: 5 files under 200 MB, don't write sidecars
python anom_bio_0023.py /data/hydrophone --limit 5 --max-size-mb 200 --dry-run

# Decisions only, no render bundles
python anom_bio_0023.py /data/hydrophone --no-render
```

## CLI Flags

| Flag | Default | Description |
|---|---|---|
| `target` | *(required)* | Folder OR single audio file to process |
| `--fmin HZ` | `100000` | Lower frequency band edge in Hz |
| `--fmax HZ` | `140000` | Upper frequency band edge in Hz |
| `--delta-t SECONDS` | `0.001` | Target time resolution between STFT frames; passed to the SDK as `desired_delta_t` |
| `--threshold` | `0.3` | Detection score threshold in [0, 1] |
| `--steps` | `1` | CA evolution steps (kept low so single-frame hits don't smear across neighbours) |
| `--min-sigma` | `1.5` | Anomaly z-score cutoff for `AnomalyDetectionRule` (used by `anomaly`, `flux_anomaly`, `edge_anomaly`) |
| `--rule` | `flux_anomaly` | CA rule chain preset — see below |
| `--evolve-steps N` | *(same as --steps)* | CA generations to render (visualization only) |
| `--limit N` | *(all)* | Process at most N files that pass the size cap (folder mode) |
| `--max-size-mb MB` | *(no cap)* | Skip files larger than this (folder mode) |
| `--output-root PATH` | *(random)* | Explicit output folder for render bundles; defaults to `anom_bio_out<rand>` |
| `--no-render` | — | Skip CA evolution render bundles (decisions only) |
| `--dry-run` | — | Score and render but do not write decision sidecars |

## `--rule` presets

All presets end with a tight `GroupingRule(min_cells=1, min_frame_span=1, min_bin_span=1)` so single-cell / single-frame activations survive as detections. Pick the one that best matches the signal you're hunting:

| Preset | Chain | Best for |
|---|---|---|
| `anomaly` | `WolframRule(90) → AnomalyDetectionRule → Grouping` | Harmonic / tonal content; slow-varying textures |
| `flux` | `SpectralFluxRule → Grouping` | **Sharp onsets** — click transients light up per-bin energy jumps |
| `flux_anomaly` *(default)* | `SpectralFluxRule → AnomalyDetectionRule → Grouping` | Onsets that also stand out statistically from local background |
| `outlier` | `LocalOutlierRule → Grouping` | Cells above local median + `k·MAD`; robust to steady noise |
| `edge_anomaly` | `EdgeGatedAnomalyRule → Grouping` | Anomalies located on spectral/temporal edges (both bright *and* transient) |

### Which to try first for porpoise clicks

Porpoise clicks are extremely short broadband-ish pulses. Onset energy is the strongest signal:

1. `flux_anomaly` (default) — should be the most sensitive out of the box.
2. `flux` — if `flux_anomaly` is too strict (anomaly requires background contrast that a quiet click may not exceed).
3. `outlier` — robust alternative if the recording has a lot of steady tonal noise that anomaly/flux both react to.
4. `edge_anomaly` — for cleaner recordings where clicks are the only sharp transients.

## Output

Each invocation creates an output folder for the render bundles (unless
`--no-render`):

```
anom_bio_out<rand>/
  index.json              — summary of all files processed
  <stem>/                 — per-file CA evolution render bundle
    frames/
    manifest.json
```

**Decision sidecars** (`<stem>.decisions.json`) are written **next to
each source audio file**, not in the output folder. Sidecars are
merge-safe: existing records from other models (e.g. `hp_ca`,
`brahma_ca_03252_anomaly`) are preserved; only records with matching
`signature == "anom_bio_0023"` are replaced on a re-run.

## Detection Record Shape

Each entry in `<stem>.decisions.json` looks like:

```json
{
  "dt": 12.345,
  "end_sec": 12.348,
  "signature": "anom_bio_0023",
  "decision": "detection",
  "reason": "ca band detection",
  "frame": 12345,
  "active_freq": "100000-140000 Hz",
  "fmin_hz": 100000.0,
  "fmax_hz": 140000.0
}
```

## Tuning notes

- **Too many detections?** Raise `--min-sigma` (2.5, 3.0) or `--threshold`
  (0.55, 0.65).
- **Missing quiet clicks?** Lower `--min-sigma` (1.5) and/or `--threshold`
  (0.3).
- **Detections merging that should be separate?** Lower `--steps` (already
  at 1); check STFT resolution — bumping `--delta-t` down further (e.g.
  0.0005) gives half-ms frames.
- **Detections splitting that should be one event?** Raise `--steps` to
  2–3 so the CA can bridge across near-neighbour frames.
