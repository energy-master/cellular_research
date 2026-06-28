# CA Evolution bundle — `ca-evolution/1`

A drop-in folder format for visualizing a cellular-automata **grid** as it
evolves over **CA generations** (the dynamical-system step axis — not acoustic
time). Local models produce the folder via `ca_render.py`; the IDent Dynamics app
renders it.

## App behavior (the contract)

- **Auto-mount on drop.** When a folder is opened/dropped into the app, if its
  root contains a `manifest.json` with `"schema": "ca-evolution/1"`, the app
  **automatically** shows the **CA Evolution panel**. The manifest's presence is
  the sole trigger.
- **No toolbar icon / menu entry.** The panel is manifest-triggered only; there
  is nothing for the user to click to enable it.
- **No audio or server round-trip required.** The bundle is self-contained.

## Folder layout

```
<bundle>/
  manifest.json            # required — the discriminator + render contract
  frames/
    step_0000.png          # CA generation 0
    step_0001.png          # CA generation 1
    ...                    # frames[i] === CA step i
  evolution.mp4            # optional — frames merged for smooth playback
```

## `manifest.json`

```json
{
  "schema": "ca-evolution/1",
  "model_name": "hp_ca",
  "n_steps": 13,
  "fps": 6,
  "colormap": "viridis",
  "value_range": [0.0, 1.0],
  "grid": { "rows": 64, "cols": 120, "orig_frames": 120, "orig_bins": 64 },
  "axes": {
    "row": "frequency_bin (low at bottom)",
    "col": "acoustic_frame",
    "frame": "ca_step"
  },
  "frames": ["frames/step_0000.png", "frames/step_0012.png"],
  "video": "evolution.mp4",
  "source": {}
}
```

| Field | Meaning |
|-------|---------|
| `schema` | Discriminator. Must be `ca-evolution/1` for auto-mount. |
| `frames` | Ordered relative PNG paths. **Index = CA generation**; length = `n_steps`. |
| `video` | Optional merged mp4 (relative path), or `null`. |
| `n_steps` / `fps` | Frame count and intended playback rate. |
| `colormap` / `value_range` | Colormap name and `[vmin, vmax]` used — for an optional legend. |
| `grid` | Rendered `rows × cols` plus original `frames × bins` (rows were decimated if `cols < orig_frames`). |
| `axes` | Human labels: row = frequency (low at bottom), col = acoustic frame, frame = CA step. |
| `source` | Free-form provenance (stream/file/run id/rule). |

## Viewer behavior

1. Read `manifest.json`; confirm `schema == "ca-evolution/1"`.
2. Render the current generation as an image:
   - **frame mode:** show `frames[i]`; a slider over `0 … n_steps-1` selects the
     generation (frame-accurate stepping). Play advances at `fps`.
   - **video mode:** play/scrub `video` if present (smooth playback). Slider
     position maps to `i / n_steps` of the timeline.
3. Each image is already oriented frequency-up / acoustic-frame-across; display
   as-is. Optionally show a colormap legend from `colormap` + `value_range`.

## Producing a bundle

`ca_render.py` in this folder:

```python
from brahma_cellular import WolframRule, AnomalyDetectionRule, GroupingRule
from ca_render import render_fourier

rule = WolframRule(rule_number=90, steps=1) | AnomalyDetectionRule() | GroupingRule()
render_fourier(fourier, rule, steps=12, out_dir="hp_ca-<runid>",
               fps=6, model_name="hp_ca", source={"stream": "hp", "file": "..."})
```

This writes `frames/step_*.png`, `evolution.mp4` (if ffmpeg is present), and the
`manifest.json` that triggers the panel.
