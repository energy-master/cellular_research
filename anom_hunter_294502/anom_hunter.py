#!/usr/bin/env python3
# Vixen Intelligence c.2026

"""anom_hunter.py — run brahma_ca_03252_anomaly against a local audio folder.

Downloads the protected model from the SDK, runs it over every WAV in a
local folder, writes decision sidecars next to each file (multi-model
merge-safe), and saves CA evolution render bundles viewable in the app.

Usage::

    python anom_hunter.py /path/to/audio
    python anom_hunter.py /path/to/audio --fmin 500 --fmax 8000 --delta-t 0.005
    python anom_hunter.py /path/to/audio --limit 5 --dry-run
    python anom_hunter.py /path/to/audio --no-render          # decisions only
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

from identdynamics import (
    ApiError,
    fetch_protected,
    list_local_audio_files,
)
from hp_ca import (
    BASE_URL,
    API_KEY,
    build_pipeline,
    crop_fourier_band,
    make_client,
    new_run_id,
    render_evolution,
    write_decisions_sidecar,
)
from hp_local_big_data_ca import fourier_for_local

#: Protected model name registered in ident db.
PROTECTED_MODEL = "brahma_ca_03252_anomaly"

#: Prefix for the random per-invocation output folder.
OUTPUT_PREFIX = "anom_hunter_out"

_DEFAULT_FMIN = 1000.0
_DEFAULT_FMAX = 8000.0
_DEFAULT_DELTA_T = 0.01
_DEFAULT_THRESHOLD = 0.65
_DEFAULT_STEPS = 4


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FileOutcome:
    """Result of scoring one local file with the protected model."""

    file: str
    path: str
    out_dir: str
    run_id: str | None = None
    n_frames: int = 0
    n_detections: int = 0
    error: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_output_root() -> str:
    return f"{OUTPUT_PREFIX}{uuid.uuid4().hex[:8]}"


def _stage(rel: str, msg: str, since: float | None = None) -> float:
    now = time.perf_counter()
    extra = f" [{now - since:.1f}s]" if since is not None else ""
    print(f"[anom_hunter]   {rel}: {msg}{extra}", flush=True)
    return now




# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(path: str, folder: str, root: str, det,
                 fmin: float, fmax: float, desired_delta_t: float,
                 threshold: float, steps: int,
                 render: bool, evolve_steps: int | None,
                 ) -> tuple[FileOutcome, list | None]:
    """Score one file with the protected model; render its CA evolution.

    Args:
        path: Absolute path to the WAV file.
        folder: Scanned root folder (for relative-path keys).
        root: Output root for this invocation.
        det: Loaded protected detector (already configured).
        fmin: Lower frequency band edge in Hz.
        fmax: Upper frequency band edge in Hz.
        desired_delta_t: Target frame spacing in seconds.
        threshold: Detection score threshold.
        steps: CA evolution steps (for the render; the model uses its own
            internal step count configured via ``configure()``).
        render: Whether to write the CA evolution bundle.
        evolve_steps: Generations to render (defaults to ``steps``).

    Returns:
        ``(outcome, detections)`` — detections is ``None`` on error.
    """
    name = os.path.basename(path)
    rel = os.path.relpath(path, folder)
    stem = os.path.splitext(name)[0]
    out_dir = os.path.join(root, stem)
    outcome = FileOutcome(file=name, path=rel, out_dir=out_dir)
    size_mb = os.path.getsize(path) / 1e6

    try:
        t = _stage(rel, f"decoding + STFT ({size_mb:.0f} MB)")
        fourier = fourier_for_local(path, desired_delta_t=desired_delta_t)
        stft_info = fourier.get("stft", {})
        actual_hop = stft_info.get("hop", 0)
        actual_sr = stft_info.get("sample_rate", fourier.get("sample_rate", 0))
        actual_dt = (actual_hop / actual_sr) if actual_sr else 0.0
        t = _stage(rel, f"{fourier['frames']} frames x {fourier['bins']} bins, "
                        f"hop={actual_hop} ({actual_dt * 1000:.2f} ms/frame)", since=t)

        t = _stage(rel, f"cropping band + running CA ({steps} steps)")
        band_fourier = crop_fourier_band(fourier, fmin, fmax)
        pipeline = build_pipeline(steps=steps, threshold=threshold)
        result = pipeline.run(band_fourier)
        result.pop("_ca", None)
        detections = result["detections"]
        for d in detections:
            d["fmin"] = fmin
            d["fmax"] = fmax

        run_id = new_run_id(PROTECTED_MODEL)
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
                    "model": PROTECTED_MODEL,
                    "rule": "WolframRule(90) | AnomalyDetectionRule | GroupingRule",
                },
            )
            _stage(rel, "rendered", since=t)

        print(f"[anom_hunter] {rel}: {outcome.n_detections} detections, "
              f"run_id={run_id}", flush=True)
        return outcome, detections

    except Exception as exc:  # noqa: BLE001
        outcome.error = f"{type(exc).__name__}: {exc}"
        print(f"[anom_hunter] {rel}: ERROR {outcome.error}")
        return outcome, None


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_hunter(folder: str, base_url: str = BASE_URL, token: str = API_KEY,
               fmin: float = _DEFAULT_FMIN, fmax: float = _DEFAULT_FMAX,
               desired_delta_t: float = _DEFAULT_DELTA_T,
               threshold: float = _DEFAULT_THRESHOLD,
               steps: int = _DEFAULT_STEPS,
               render: bool = True, evolve_steps: int | None = None,
               limit: int | None = None, max_size_mb: float | None = None,
               output_root: str | None = None,
               dry_run: bool = False) -> tuple[str, list[FileOutcome]]:
    """Download the protected model and score every WAV in a local folder.

    Args:
        folder: Local audio folder to scan.
        base_url: API host.
        token: Bearer API key.
        fmin: Lower frequency band edge in Hz.
        fmax: Upper frequency band edge in Hz.
        desired_delta_t: Target time resolution between frames in seconds;
            sets STFT hop to ``round(desired_delta_t * sample_rate)``.
        threshold: Detection score threshold in [0, 1].
        steps: CA evolution steps used for the render bundle.
        render: Write a CA evolution bundle per file into ``output_root``.
        evolve_steps: Generations to render (defaults to ``steps``).
        limit: Process at most this many files under the size cap.
        max_size_mb: Skip files larger than this many MB.
        output_root: Explicit output root (random if omitted).
        dry_run: Score and render but do not write decision sidecars.

    Returns:
        ``(root, outcomes)`` — output folder and one :class:`FileOutcome`
        per file.
    """
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise NotADirectoryError(folder)

    client = make_client(base_url, token)

    print(f"[anom_hunter] downloading {PROTECTED_MODEL!r} ...")
    det = fetch_protected(client, PROTECTED_MODEL, params={
        "fmin": fmin,
        "fmax": fmax,
        "desired_delta_t": desired_delta_t,
        "threshold": threshold,
        "steps": steps,
    })
    print(f"[anom_hunter] model ready: {det.name} v{det.version}, "
          f"expects={det.expects}")

    files = list_local_audio_files(folder)
    files = [p for p in files if not os.path.basename(p).startswith("._")]

    root = output_root or _new_output_root()
    os.makedirs(root, exist_ok=True)

    band_label = f"{int(fmin)}-{int(fmax)} Hz"
    cap_bytes = int(max_size_mb * 1024 * 1024) if max_size_mb else None

    print(f"[anom_hunter] folder={folder!r} files={len(files)} -> root={root!r} "
          f"band={band_label} delta_t={desired_delta_t}s threshold={threshold}")

    outcomes: list[FileOutcome] = []
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
            print(f"[anom_hunter] {rel}: {skipped.error}")
            outcomes.append(skipped)
            continue

        if limit is not None and n_processed >= limit:
            break
        n_processed += 1
        print(f"[anom_hunter] ({n_processed}{f'/{limit}' if limit else ''}) "
              f"{rel} ({size_mb:.0f} MB)")

        outcome, dets = process_file(
            path, folder, root, det,
            fmin, fmax, desired_delta_t, threshold, steps,
            render, evolve_steps)
        outcomes.append(outcome)
        gc.collect()

        if dets and not dry_run:
            try:
                write_decisions_sidecar(
                    os.path.dirname(path), os.path.basename(path),
                    PROTECTED_MODEL, (fmin, fmax), dets)
            except OSError as exc:
                print(f"[anom_hunter] {rel}: sidecar write failed: {exc}")

    index = {
        "folder": folder,
        "model": PROTECTED_MODEL,
        "band_hz": [fmin, fmax],
        "desired_delta_t": desired_delta_t,
        "threshold": threshold,
        "n_files": len(outcomes),
        "n_skipped": sum(1 for o in outcomes if o.error and o.error.startswith("skipped:")),
        "n_failed": sum(1 for o in outcomes if o.error and not o.error.startswith("skipped:")),
        "files": [asdict(o) for o in outcomes],
    }
    with open(os.path.join(root, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)
    print(f"[anom_hunter] done: {index['n_files']} files, "
          f"{index['n_skipped']} skipped, {index['n_failed']} failed "
          f"-> {root}/index.json")
    return root, outcomes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=f"Run {PROTECTED_MODEL} (protected) over a local audio folder.")
    p.add_argument("folder", help="local folder of audio to process")
    p.add_argument("--fmin", type=float, default=_DEFAULT_FMIN,
                   help=f"lower frequency band edge in Hz (default: {_DEFAULT_FMIN:.0f})")
    p.add_argument("--fmax", type=float, default=_DEFAULT_FMAX,
                   help=f"upper frequency band edge in Hz (default: {_DEFAULT_FMAX:.0f})")
    p.add_argument("--delta-t", type=float, default=_DEFAULT_DELTA_T,
                   dest="desired_delta_t", metavar="SECONDS",
                   help=f"target time resolution between frames; "
                        f"hop = round(delta_t * sample_rate) "
                        f"(default: {_DEFAULT_DELTA_T}s)")
    p.add_argument("--threshold", type=float, default=_DEFAULT_THRESHOLD,
                   help=f"detection score threshold in [0, 1] (default: {_DEFAULT_THRESHOLD})")
    p.add_argument("--steps", type=int, default=_DEFAULT_STEPS,
                   help=f"CA evolution steps for the render bundle (default: {_DEFAULT_STEPS})")
    p.add_argument("--evolve-steps", type=int, default=None,
                   help="CA generations to render (default: same as --steps)")
    p.add_argument("--limit", type=int, default=None,
                   help="process at most N files under the size cap (default: all)")
    p.add_argument("--max-size-mb", type=float, default=None,
                   help="skip files larger than this many MB (default: no cap)")
    p.add_argument("--output-root", default=None,
                   help="explicit output folder (default: random anom_hunter_out<rand>)")
    p.add_argument("--no-render", dest="render", action="store_false",
                   help="skip writing CA evolution render bundles")
    p.add_argument("--dry-run", action="store_true",
                   help="score and render but do not write decision sidecars")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Script entry point. Returns a process exit code."""
    args = _parse_args(argv)
    try:
        run_hunter(
            folder=args.folder,
            fmin=args.fmin,
            fmax=args.fmax,
            desired_delta_t=args.desired_delta_t,
            threshold=args.threshold,
            steps=args.steps,
            evolve_steps=args.evolve_steps,
            render=args.render,
            limit=args.limit,
            max_size_mb=args.max_size_mb,
            output_root=args.output_root,
            dry_run=args.dry_run,
        )
    except (ApiError, NotADirectoryError, ValueError) as exc:
        print(f"[anom_hunter] error: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
