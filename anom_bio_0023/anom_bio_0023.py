#!/usr/bin/env python3
# Vixen Intelligence c.2026

"""anom_bio_0023 — CA tuned for single-frame ms-echo hits in local audio.

Runs a cellular-automata pipeline aimed at short (millisecond-scale)
transient echoes in the 100-140 kHz band — e.g. harbour porpoise clicks —
against either a single WAV or an entire folder. Unlike the general
hp_local_big_data_ca, this pipeline is configured so single-frame
anomalies register as detections (GroupingRule with ``min_frame_span=1``
/ ``min_cells=1``), so a tight burst spanning only one STFT frame still
fires.

Decisions are written next to each audio file as a merge-safe
``<base>.decisions.json`` sidecar (multiple models accumulate; existing
records from other models are preserved). A per-file CA evolution bundle
is optionally written under one random ``anom_bio_out<rand>/`` root so
the app can visualize the CA grid.

Usage::

    python anom_bio_0023.py /path/to/audio                    # a folder
    python anom_bio_0023.py /path/to/file.wav                 # one file
    python anom_bio_0023.py /path/to/audio --fmin 110000 --fmax 130000
    python anom_bio_0023.py /path/to/audio --delta-t 0.001 --min-sigma 2.5
    python anom_bio_0023.py /path/to/audio --limit 5 --dry-run
    python anom_bio_0023.py /path/to/audio --no-render        # decisions only
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass

# Resolve hp_games helpers from the sibling folder.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hp_games"))

from brahma_cellular import (
    Pipeline,
    WolframRule,
    AnomalyDetectionRule,
    EdgeGatedAnomalyRule,
    GroupingRule,
    LocalOutlierRule,
    SpectralFluxRule,
)
from identdynamics import (
    ApiError,
    fourier_for_path,
    list_local_audio_files,
    save_scores_local,
)
from hp_ca import (
    crop_fourier_band,
    new_run_id,
    render_evolution,
    write_decisions_sidecar,
)

#: Model label written into decision sidecars as the ``signature`` field.
MODEL_NAME = "anom_bio_0023"

#: Prefix for the random per-invocation output folder.
OUTPUT_PREFIX = "anom_bio_out"

_DEFAULT_FMIN = 100_000.0
_DEFAULT_FMAX = 140_000.0
_DEFAULT_DELTA_T = 0.001
_DEFAULT_THRESHOLD = 0.3
_DEFAULT_STEPS = 1
_DEFAULT_MIN_SIGMA = 1.5
_DEFAULT_RULE = "flux_anomaly"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FileOutcome:
    """Result of running the CA on one local file."""

    file: str
    path: str
    out_dir: str
    run_id: str | None = None
    n_frames: int = 0
    n_detections: int = 0
    error: str | None = None


# ---------------------------------------------------------------------------
# CA pipeline tuned for single-frame ms-echo hits
# ---------------------------------------------------------------------------

#: Named CA rule chains. All share the same tight GroupingRule
#: (single-frame / single-cell) so any surviving activation fires. Pick with
#: ``--rule``; sensitivity knobs like ``--min-sigma`` apply where relevant.
RULE_PRESETS: tuple[str, ...] = (
    "anomaly",         # WolframRule(90) | AnomalyDetectionRule
    "flux",            # SpectralFluxRule (per-bin onset energy)
    "flux_anomaly",    # SpectralFluxRule | AnomalyDetectionRule
    "outlier",         # LocalOutlierRule (median + MAD)
    "edge_anomaly",    # EdgeGatedAnomalyRule (edge-gated anomaly)
)


def _tight_grouping() -> GroupingRule:
    """Grouping tuned for single-frame single-cell transient hits."""
    return GroupingRule(min_cells=1, min_frame_span=1, min_bin_span=1)


def build_bio_pipeline(rule_name: str, steps: int, threshold: float,
                       min_sigma: float) -> Pipeline:
    """Build a CA pipeline for one of :data:`RULE_PRESETS`.

    All presets end in the tight :func:`_tight_grouping` so a single active
    cell in a single frame produces a detection — appropriate for
    sub-millisecond echoes at 1 ms/frame resolution.

    Args:
        rule_name: One of :data:`RULE_PRESETS`.
        steps: CA evolution steps (only used by presets containing
            :class:`WolframRule`; kept low so single-frame hits survive).
        threshold: Per-frame score threshold in [0, 1].
        min_sigma: Local z-score cutoff for :class:`AnomalyDetectionRule`
            (used by ``anomaly`` and ``flux_anomaly``) and for the anomaly
            side of ``edge_anomaly``.

    Returns:
        A configured :class:`brahma_cellular.Pipeline`.

    Raises:
        ValueError: If ``rule_name`` is not one of :data:`RULE_PRESETS`.
    """
    if rule_name == "anomaly":
        chain = (WolframRule(rule_number=90, steps=steps)
                 | AnomalyDetectionRule(min_sigma=min_sigma)
                 | _tight_grouping())
    elif rule_name == "flux":
        chain = SpectralFluxRule() | _tight_grouping()
    elif rule_name == "flux_anomaly":
        chain = (SpectralFluxRule()
                 | AnomalyDetectionRule(min_sigma=min_sigma)
                 | _tight_grouping())
    elif rule_name == "outlier":
        chain = LocalOutlierRule() | _tight_grouping()
    elif rule_name == "edge_anomaly":
        chain = EdgeGatedAnomalyRule(anomaly_min_sigma=min_sigma) | _tight_grouping()
    else:
        raise ValueError(f"unknown --rule {rule_name!r}; "
                         f"choose from {RULE_PRESETS}")
    return Pipeline(rule=chain, model_name=MODEL_NAME,
                    steps=steps, threshold=threshold, label=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_output_root() -> str:
    return f"{OUTPUT_PREFIX}{uuid.uuid4().hex[:8]}"


def _stage(rel: str, msg: str, since: float | None = None) -> float:
    now = time.perf_counter()
    extra = f" [{now - since:.1f}s]" if since is not None else ""
    print(f"[anom_bio]   {rel}: {msg}{extra}", flush=True)
    return now


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(path: str, folder: str, root: str,
                 fmin: float, fmax: float, desired_delta_t: float,
                 threshold: float, steps: int, min_sigma: float,
                 rule_name: str,
                 render: bool, evolve_steps: int | None,
                 ) -> tuple[FileOutcome, list | None, dict | None]:
    """Run the ms-echo CA on one file; render its CA evolution.

    Returns:
        ``(outcome, detections, extras)`` — ``detections`` and ``extras`` are
        ``None`` on error. ``extras`` carries the raw per-frame ``scores`` and
        the ``stft`` params needed by :func:`identdynamics.save_scores_local`.
    """
    name = os.path.basename(path)
    rel = os.path.relpath(path, folder)
    stem = os.path.splitext(name)[0]
    out_dir = os.path.join(root, stem)
    outcome = FileOutcome(file=name, path=rel, out_dir=out_dir)
    size_mb = os.path.getsize(path) / 1e6

    try:
        t = _stage(rel, f"decoding + STFT ({size_mb:.0f} MB)")
        fourier = fourier_for_path(path, desired_delta_t=desired_delta_t)
        stft_info = fourier.get("stft", {})
        actual_hop = stft_info.get("hop", 0)
        actual_sr = stft_info.get("sample_rate", fourier.get("sample_rate", 0))
        actual_dt = (actual_hop / actual_sr) if actual_sr else 0.0

        band_fourier = crop_fourier_band(fourier, fmin, fmax)
        t = _stage(rel, f"{fourier['frames']} frames x {fourier['bins']} bins "
                        f"-> band {fmin:.0f}-{fmax:.0f} Hz ({band_fourier['bins']} bins), "
                        f"hop={actual_hop} ({actual_dt * 1000:.2f} ms/frame)",
                   since=t)

        pipeline = build_bio_pipeline(rule_name=rule_name, steps=steps,
                                      threshold=threshold, min_sigma=min_sigma)
        t = _stage(rel, f"evolving CA [rule={rule_name}, steps={steps}, "
                        f"min_sigma={min_sigma}]")
        result = pipeline.run(band_fourier)
        result.pop("_ca", None)
        detections = result["detections"]
        for det in detections:
            det["fmin"] = fmin
            det["fmax"] = fmax

        run_id = new_run_id(MODEL_NAME)
        outcome.run_id = run_id
        outcome.n_frames = len(result["scores"])
        outcome.n_detections = len(detections)
        t = _stage(rel, f"{outcome.n_detections} detections", since=t)

        if render:
            t = _stage(rel, "rendering CA evolution")
            render_evolution(
                band_fourier,
                evolve_steps if evolve_steps is not None else steps,
                out_dir,
                source={
                    "folder": folder,
                    "file": rel,
                    "run_id": run_id,
                    "band_hz": [fmin, fmax],
                    "model": MODEL_NAME,
                    "rule": rule_name,
                },
            )
            _stage(rel, "rendered", since=t)

        print(f"[anom_bio] {rel}: {outcome.n_detections} detections, "
              f"run_id={run_id}", flush=True)
        extras = {"scores": result["scores"], "stft": stft_info}
        return outcome, detections, extras
    except Exception as exc:  # noqa: BLE001 -- never let one file abort the batch
        outcome.error = f"{type(exc).__name__}: {exc}"
        print(f"[anom_bio] {rel}: ERROR {outcome.error}")
        return outcome, None, None


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_bio(target: str,
            fmin: float = _DEFAULT_FMIN, fmax: float = _DEFAULT_FMAX,
            desired_delta_t: float = _DEFAULT_DELTA_T,
            threshold: float = _DEFAULT_THRESHOLD,
            steps: int = _DEFAULT_STEPS,
            min_sigma: float = _DEFAULT_MIN_SIGMA,
            rule_name: str = _DEFAULT_RULE,
            render: bool = True, evolve_steps: int | None = None,
            limit: int | None = None, max_size_mb: float | None = None,
            output_root: str | None = None,
            dry_run: bool = False) -> tuple[str, list[FileOutcome]]:
    """Score the ms-echo CA over a single WAV or every WAV in a folder.

    ``target`` may be either a directory (all WAVs are processed, honouring
    ``--limit`` and ``--max-size-mb``) or a single audio file. In the
    single-file case, the sidecar is written next to that file and the
    scanned "folder" recorded in the index is the file's parent directory.
    """
    target = os.path.abspath(target)
    if os.path.isfile(target):
        folder = os.path.dirname(target)
        files = [target]
    elif os.path.isdir(target):
        folder = target
        files = list_local_audio_files(target)
        files = [p for p in files if not os.path.basename(p).startswith("._")]
    else:
        raise FileNotFoundError(target)

    root = output_root or _new_output_root()
    os.makedirs(root, exist_ok=True)

    band_label = f"{int(fmin)}-{int(fmax)} Hz"
    cap_bytes = int(max_size_mb * 1024 * 1024) if max_size_mb else None

    print(f"[anom_bio] folder={folder!r} files={len(files)} -> root={root!r} "
          f"band={band_label} delta_t={desired_delta_t}s threshold={threshold} "
          f"rule={rule_name} steps={steps} min_sigma={min_sigma}")

    outcomes: list[FileOutcome] = []
    n_sidecars = 0
    n_score_sidecars = 0
    n_processed = 0

    for path in files:
        rel = os.path.relpath(path, folder)
        size_mb = os.path.getsize(path) / 1e6

        if cap_bytes is not None and os.path.getsize(path) > cap_bytes:
            stem = os.path.splitext(os.path.basename(path))[0]
            skipped = FileOutcome(
                file=os.path.basename(path), path=rel,
                out_dir=os.path.join(root, stem),
                error=f"skipped: {size_mb:.0f} MB > {max_size_mb:.0f} MB cap")
            print(f"[anom_bio] {rel}: {skipped.error}")
            outcomes.append(skipped)
            continue

        if limit is not None and n_processed >= limit:
            break
        n_processed += 1
        print(f"[anom_bio] ({n_processed}{f'/{limit}' if limit else ''}) "
              f"{rel} ({size_mb:.0f} MB)")

        outcome, dets, extras = process_file(
            path, folder, root,
            fmin, fmax, desired_delta_t, threshold, steps, min_sigma,
            rule_name, render, evolve_steps)
        outcomes.append(outcome)
        gc.collect()

        if not dry_run and dets:
            try:
                write_decisions_sidecar(
                    os.path.dirname(path), os.path.basename(path),
                    MODEL_NAME, (fmin, fmax), dets)
                n_sidecars += 1
            except OSError as exc:
                print(f"[anom_bio] {rel}: decision sidecar write failed: {exc}")

        # Score-vs-time sidecar for the app's "score vs time" plot. Written even
        # when there are zero detections so the curve is still viewable.
        if not dry_run and extras is not None:
            try:
                save_scores_local(os.path.dirname(path), [{
                    "name": os.path.basename(path),
                    "scores": extras["scores"],
                    "threshold": threshold,
                    "stft": extras["stft"],
                    "fmin": fmin, "fmax": fmax,
                    "model_name": MODEL_NAME,
                }])
                n_score_sidecars += 1
            except OSError as exc:
                print(f"[anom_bio] {rel}: score sidecar write failed: {exc}")

    if dry_run:
        print(f"[anom_bio] dry-run: sidecars skipped")
    else:
        print(f"[anom_bio] wrote {n_sidecars} decision sidecar(s) and "
              f"{n_score_sidecars} score sidecar(s) into {folder}")

    index = {
        "folder": folder,
        "model_name": MODEL_NAME,
        "band_hz": [fmin, fmax],
        "desired_delta_t": desired_delta_t,
        "threshold": threshold,
        "steps": steps,
        "min_sigma": min_sigma,
        "rule": rule_name,
        "n_files": len(outcomes),
        "n_skipped": sum(1 for o in outcomes if o.error and o.error.startswith("skipped:")),
        "n_failed": sum(1 for o in outcomes if o.error and not o.error.startswith("skipped:")),
        "n_sidecars": n_sidecars,
        "n_score_sidecars": n_score_sidecars,
        "files": [asdict(o) for o in outcomes],
    }
    with open(os.path.join(root, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)
    print(f"[anom_bio] done: {index['n_files']} files, "
          f"{index['n_skipped']} skipped, {index['n_failed']} failed "
          f"-> {root}/index.json")
    return root, outcomes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=f"Run {MODEL_NAME} (single-frame ms-echo CA) over a local "
                    f"audio folder or a single WAV; write decision sidecars "
                    f"next to each audio file.")
    p.add_argument("target",
                   help="local folder OR a single audio file to process")
    p.add_argument("--fmin", type=float, default=_DEFAULT_FMIN,
                   help=f"lower frequency band edge in Hz (default: {_DEFAULT_FMIN:.0f})")
    p.add_argument("--fmax", type=float, default=_DEFAULT_FMAX,
                   help=f"upper frequency band edge in Hz (default: {_DEFAULT_FMAX:.0f})")
    p.add_argument("--delta-t", type=float, default=_DEFAULT_DELTA_T,
                   dest="desired_delta_t", metavar="SECONDS",
                   help=f"target time resolution between frames in seconds; "
                        f"hop = round(delta_t * sample_rate) "
                        f"(default: {_DEFAULT_DELTA_T}s)")
    p.add_argument("--threshold", type=float, default=_DEFAULT_THRESHOLD,
                   help=f"detection score threshold in [0, 1] "
                        f"(default: {_DEFAULT_THRESHOLD})")
    p.add_argument("--steps", type=int, default=_DEFAULT_STEPS,
                   help=f"CA evolution steps; keep low so single-frame hits "
                        f"survive (default: {_DEFAULT_STEPS})")
    p.add_argument("--min-sigma", type=float, default=_DEFAULT_MIN_SIGMA,
                   dest="min_sigma",
                   help=f"anomaly z-score cutoff for the CA "
                        f"(default: {_DEFAULT_MIN_SIGMA})")
    p.add_argument("--rule", default=_DEFAULT_RULE, choices=RULE_PRESETS,
                   dest="rule_name",
                   help=f"CA rule chain preset (default: {_DEFAULT_RULE}). "
                        f"anomaly = WolframRule(90) + AnomalyDetectionRule; "
                        f"flux = SpectralFluxRule (best for click onsets); "
                        f"flux_anomaly = SpectralFluxRule + AnomalyDetectionRule; "
                        f"outlier = LocalOutlierRule (median + MAD); "
                        f"edge_anomaly = EdgeGatedAnomalyRule.")
    p.add_argument("--evolve-steps", type=int, default=None,
                   help="CA generations to render (default: same as --steps)")
    p.add_argument("--limit", type=int, default=None,
                   help="process at most N files under the size cap (default: all)")
    p.add_argument("--max-size-mb", type=float, default=None,
                   help="skip files larger than this many MB (default: no cap)")
    p.add_argument("--output-root", default=None,
                   help=f"explicit output folder (default: random {OUTPUT_PREFIX}<rand>)")
    p.add_argument("--no-render", dest="render", action="store_false",
                   help="skip writing CA evolution render bundles")
    p.add_argument("--dry-run", action="store_true",
                   help="score and render but do not write decision sidecars")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Script entry point. Returns a process exit code."""
    args = _parse_args(argv)
    try:
        run_bio(
            target=args.target,
            fmin=args.fmin,
            fmax=args.fmax,
            desired_delta_t=args.desired_delta_t,
            threshold=args.threshold,
            steps=args.steps,
            min_sigma=args.min_sigma,
            rule_name=args.rule_name,
            evolve_steps=args.evolve_steps,
            render=args.render,
            limit=args.limit,
            max_size_mb=args.max_size_mb,
            output_root=args.output_root,
            dry_run=args.dry_run,
        )
    except (ApiError, FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"[anom_bio] error: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
