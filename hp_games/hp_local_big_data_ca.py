#!/usr/bin/env python3
# Vixen Intelligence c.2026

"""hp_local_big_data_ca — run the CA over a big LOCAL folder, save as a project.

Local-folder sibling of :mod:`hp_stream_ca` (which runs the ``hp`` stream). It
walks a folder of audio on disk and runs the *same* band-limited CA over every
file -- no upload of the audio -- so a large local dataset can be processed
without a server round-trip per file.

Each file gets the same per-file output as the stream runner -- a self-contained
``ca-evolution/1`` bundle under one random root folder, plus an ``index.json``::

    <output-root>/
        index.json
        <file-stem-1>/  manifest.json, frames/, evolution.mp4
        <file-stem-2>/  ...

After processing, the whole run is registered as a **Work Project** in ident db
via the SDK (``save_run_project``): the folder + each file's detection intervals
are saved so the run shows up in the app's *Work projects* list. Re-attach the
same local folder in the app and the decisions render overlaid on each file -- no
audio is uploaded, only the decisions.

Run it directly::

    python hp_local_big_data_ca.py /path/to/folder
    python hp_local_big_data_ca.py /path/to/folder --limit 5     # smoke test
    python hp_local_big_data_ca.py /path/to/folder --dry-run     # render only, no project

Note:
    The default band (:data:`hp_ca.FREQ_BAND`, 100-150 kHz) only selects bins on
    high-rate recordings; on a folder below ~300 kHz the band falls outside
    Nyquist and :func:`hp_ca.crop_fourier_band` keeps the full grid.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from dataclasses import dataclass, asdict

from identdynamics import ApiError, list_local_audio_files
from hp_ca import (
    BASE_URL,
    API_KEY,
    MODEL_NAME,
    FREQ_BAND,
    make_client,
    build_pipeline,
    crop_fourier_band,
    render_evolution,
    new_run_id,
)

#: Prefix for the random per-invocation root output folder.
OUTPUT_PREFIX = "local_ca_out"


@dataclass
class FileOutcome:
    """Result of running the CA on one local file.

    Attributes:
        file: File name (basename).
        path: Path relative to the scanned folder (the key the app re-matches).
        out_dir: Per-file output folder (a ``ca-evolution/1`` bundle).
        run_id: Random run id minted for the file.
        n_frames: Number of spectrogram frames scored.
        n_detections: Number of detections produced.
        error: Error string if the file failed, else ``None``.
    """

    file: str
    path: str
    out_dir: str
    run_id: str | None = None
    n_frames: int = 0
    n_detections: int = 0
    error: str | None = None


def new_output_root(prefix: str = OUTPUT_PREFIX) -> str:
    """Mint a random root output folder name, e.g. ``local_ca_out1f0c9ab2``."""
    return f"{prefix}{uuid.uuid4().hex[:8]}"


def fourier_for_local(path: str, fft: int = 1024, hop: int | None = None,
                      window: str = "hann") -> dict:
    """Fourier grid for a local WAV, tolerant of IEEE-float files.

    Uses the SDK's reference STFT (``fourier_for_path``) and, on a decode failure
    -- the stdlib ``wave`` module rejects IEEE-float WAVs (format 3) -- falls back
    to a SciPy decode so high-rate float recordings still load. Mirrors
    :func:`hp_ca.fourier_for_saved_file` for on-disk files.

    Args:
        path: Path to a WAV file on disk.
        fft: FFT size.
        hop: Hop size (defaults to ``fft // 4``).
        window: Window name.

    Returns:
        The standard Fourier dict (``magnitudes``/``frames``/``bins``/``stft``).
    """
    from identdynamics import fourier_for_path

    try:
        return fourier_for_path(path, fft=fft, hop=hop, window=window)
    except Exception:  # noqa: BLE001 -- fall back only on a decode failure
        import numpy as np
        from scipy.io import wavfile
        from identdynamics import stft

        sample_rate, data = wavfile.read(path)
        signal = data.astype(np.float64)
        if np.issubdtype(data.dtype, np.integer):
            signal /= float(np.iinfo(data.dtype).max)
        if signal.ndim > 1:
            signal = signal.mean(axis=1)
        hop_size = (fft >> 2) if hop is None else int(hop)
        mags, frames, bins = stft(signal, fft, hop_size, window)
        return {
            "magnitudes": mags, "frames": frames, "bins": bins,
            "sample_rate": int(sample_rate),
            "stft": {"fft": fft, "hop": hop_size, "window": window,
                     "sample_rate": int(sample_rate)},
        }


def process_file(path: str, folder: str, root: str, steps: int, threshold: float,
                 freq_band: tuple[float, float], render: bool,
                 evolve_steps: int | None) -> tuple[FileOutcome, dict | None]:
    """Run the band-limited CA on one local file; render + collect decisions.

    Args:
        path: Absolute path to the audio file.
        folder: The scanned root folder (for relative path keys).
        root: Root output folder for this invocation.
        steps: CA evolution steps.
        threshold: Detection threshold.
        freq_band: ``(fmin, fmax)`` Hz band the CA is restricted to.
        render: Whether to write the evolution bundle.
        evolve_steps: Generations to render (defaults to ``steps``).

    Returns:
        ``(outcome, per_file_entry)`` where ``per_file_entry`` is the
        ``save_run_project`` record (``{"name","path","detections"}`` with
        ``detections`` as ``(start_sec, end_sec)`` tuples), or ``None`` on error.
        Per-file failures are captured in ``outcome.error`` rather than raised.
    """
    name = os.path.basename(path)
    rel = os.path.relpath(path, folder)
    stem = os.path.splitext(name)[0]
    out_dir = os.path.join(root, stem)
    outcome = FileOutcome(file=name, path=rel, out_dir=out_dir)

    try:
        fourier = fourier_for_local(path)
        fmin, fmax = float(freq_band[0]), float(freq_band[1])
        band_fourier = crop_fourier_band(fourier, fmin, fmax)
        print(f"[hp_local] {rel}: {fourier['frames']} frames, band "
              f"{fmin:.0f}-{fmax:.0f} Hz -> {band_fourier['bins']} of "
              f"{fourier['bins']} bins")

        pipeline = build_pipeline(steps=steps, threshold=threshold)
        result = pipeline.run(band_fourier)
        result.pop("_ca", None)
        detections = result["detections"]
        for det in detections:
            det["fmin"] = fmin
            det["fmax"] = fmax

        run_id = new_run_id()
        outcome.run_id = run_id
        outcome.n_frames = len(result["scores"])
        outcome.n_detections = len(detections)

        if render:
            render_evolution(
                band_fourier,
                evolve_steps if evolve_steps is not None else steps,
                out_dir,
                source={"folder": folder, "file": rel, "run_id": run_id,
                        "band_hz": [fmin, fmax],
                        "rule": "WolframRule(90) | AnomalyDetectionRule | GroupingRule"},
            )

        # Decisions for the Work Project: detection intervals in seconds.
        entry = {
            "name": name,
            "path": rel,
            "detections": [(float(d["start_sec"]), float(d["end_sec"]))
                           for d in detections],
        }
        print(f"[hp_local] {rel}: {outcome.n_detections} detections, run_id={run_id}")
        return outcome, entry
    except (ApiError, ValueError, KeyError, OSError) as exc:
        outcome.error = f"{type(exc).__name__}: {exc}"
        print(f"[hp_local] {rel}: ERROR {outcome.error}")
        return outcome, None


def run_local(folder: str, base_url: str = BASE_URL, token: str = API_KEY,
              steps: int = 3, threshold: float = 0.45,
              freq_band: tuple[float, float] = FREQ_BAND, render: bool = True,
              evolve_steps: int | None = None, dry_run: bool = False,
              limit: int | None = None, output_root: str | None = None,
              project_name: str | None = None,
              save_project: bool = True) -> tuple[str, list[FileOutcome]]:
    """Run the CA over every audio file in a local folder and save a project.

    Renders a per-file evolution bundle for each file under a random root folder,
    writes an ``index.json``, and (unless ``dry_run`` or ``save_project`` is off)
    registers the whole run as a Work Project in ident db so the decisions are
    viewable from the project in the app.

    Args:
        folder: Local folder to scan for audio.
        base_url: API host.
        token: Bearer API key.
        steps: CA evolution steps.
        threshold: Detection threshold.
        freq_band: ``(fmin, fmax)`` Hz band the CA is restricted to.
        render: Whether to write per-file evolution bundles.
        evolve_steps: Generations to render (defaults to ``steps``).
        dry_run: If ``True``, render but do not save the project to ident db.
        limit: Process at most this many files (``None`` = all).
        output_root: Explicit root output folder (random if omitted).
        project_name: Work Project name (defaults to the folder basename).
        save_project: Whether to register the Work Project (skipped on dry runs).

    Returns:
        ``(root, outcomes)`` -- the root output folder and one
        :class:`FileOutcome` per file.

    Raises:
        NotADirectoryError: If ``folder`` is not a directory.
    """
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise NotADirectoryError(folder)
    files = list_local_audio_files(folder)
    if limit is not None:
        files = files[:limit]

    root = output_root or new_output_root()
    os.makedirs(root, exist_ok=True)
    print(f"[hp_local] folder={folder!r} files={len(files)} -> root={root!r}")

    outcomes: list[FileOutcome] = []
    per_file: list[dict] = []
    first_stft: dict | None = None
    for i, path in enumerate(files, 1):
        print(f"[hp_local] ({i}/{len(files)}) {os.path.relpath(path, folder)}")
        outcome, entry = process_file(
            path, folder, root, steps, threshold, freq_band, render, evolve_steps)
        outcomes.append(outcome)
        if entry is not None:
            per_file.append(entry)

    # Use the first decoded file's STFT for the project's display spectrogram.
    for o in outcomes:
        mani = os.path.join(o.out_dir, "manifest.json")
        if first_stft is None and os.path.exists(mani):
            try:
                first_stft = json.load(open(mani)).get("stft")
            except Exception:
                first_stft = None

    client = make_client(base_url, token)
    project = None
    if save_project and not dry_run and per_file:
        project = client.save_run_project(
            folder=folder,
            per_file=per_file,
            model_name=MODEL_NAME,
            name=project_name,
            stft=first_stft,
        )
        print(f"[hp_local] saved Work Project to ident db: {project}")
    elif dry_run:
        print(f"[hp_local] dry-run: not saving project "
              f"({len(per_file)} files would be saved)")

    index = {
        "folder": folder,
        "model_name": MODEL_NAME,
        "band_hz": [float(freq_band[0]), float(freq_band[1])],
        "n_files": len(outcomes),
        "n_failed": sum(1 for o in outcomes if o.error),
        "project": project,
        "files": [asdict(o) for o in outcomes],
    }
    with open(os.path.join(root, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)
    print(f"[hp_local] done: {index['n_files']} files, {index['n_failed']} failed "
          f"-> {root}/index.json")
    return root, outcomes


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the script entry point."""
    p = argparse.ArgumentParser(
        description="Run the hp_ca CA over a local folder and save as a Work Project.")
    p.add_argument("folder", help="local folder of audio to process")
    p.add_argument("--steps", type=int, default=3, help="CA evolution steps")
    p.add_argument("--threshold", type=float, default=0.45, help="detection threshold")
    p.add_argument("--freq-band", type=float, nargs=2, metavar=("FMIN", "FMAX"),
                   default=list(FREQ_BAND),
                   help="frequency band (Hz) to restrict the CA to "
                        "(default: %d %d)" % (int(FREQ_BAND[0]), int(FREQ_BAND[1])))
    p.add_argument("--limit", type=int, default=None,
                   help="process at most N files (default: all)")
    p.add_argument("--output-root", default=None,
                   help="explicit root output folder (default: random local_ca_out<rand>)")
    p.add_argument("--project-name", default=None,
                   help="Work Project name (default: folder basename)")
    p.add_argument("--evolve-steps", type=int, default=None,
                   help="CA generations to render per file (default: same as --steps)")
    p.add_argument("--no-render", dest="render", action="store_false",
                   help="skip writing per-file evolution bundles")
    p.add_argument("--no-save", dest="save_project", action="store_false",
                   help="skip registering the Work Project in ident db")
    p.add_argument("--dry-run", action="store_true",
                   help="render bundles but do not save the project")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Script entry point. Returns a process exit code."""
    args = _parse_args(argv)
    try:
        run_local(
            folder=args.folder,
            steps=args.steps,
            threshold=args.threshold,
            freq_band=(args.freq_band[0], args.freq_band[1]),
            render=args.render,
            evolve_steps=args.evolve_steps,
            dry_run=args.dry_run,
            limit=args.limit,
            output_root=args.output_root,
            project_name=args.project_name,
            save_project=args.save_project,
        )
    except (ApiError, NotADirectoryError, ValueError) as exc:
        print(f"[hp_local] error: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
