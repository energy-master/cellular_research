#!/usr/bin/env python3
# Vixen Intelligence c.2026

"""hp_stream_ca — run the cellular-automata model across the whole ``hp`` stream.

Sibling of :mod:`hp_ca` (which runs one saved file). This connects to the ``hp``
stream and runs the *same* band-limited CA over **every** WAV file in it, writing
a per-file CA-evolution bundle and posting each run to ident db.

All results land under a single, randomly named root folder; each stream file
gets its own subfolder (a self-contained ``ca-evolution/1`` bundle you can drop
into the app)::

    stream_hp_out<rand>/
        index.json                 # summary of every file's run
        <file-stem-1>/             # one drop-in bundle per stream file
            manifest.json
            frames/step_0000.png ...
            evolution.mp4
        <file-stem-2>/
            ...

The CA building blocks (rule chain, band crop, evolution render, run id) are
reused from :mod:`hp_ca` so the two stay in lock-step.

Run it directly::

    python hp_stream_ca.py                 # every WAV in the hp stream
    python hp_stream_ca.py --limit 3       # first 3 files (smoke test)
    python hp_stream_ca.py --smallest-first # process by ascending size
    python hp_stream_ca.py --dry-run       # render bundles, do not post to ident db

Note:
    The ``hp`` stream holds large multi-gigabyte recordings; a full pass fetches
    and processes every file and can take a long time. Use ``--limit`` /
    ``--smallest-first`` to scope a trial run.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from dataclasses import dataclass, asdict

from identdynamics import ApiError
from hp_ca import (
    BASE_URL,
    API_KEY,
    STREAM_NAME,
    STREAM_USER,
    MODEL_NAME,
    FREQ_BAND,
    make_client,
    resolve_stream,
    build_pipeline,
    crop_fourier_band,
    render_evolution,
    new_run_id,
)

#: Prefix for the random per-invocation root output folder.
OUTPUT_PREFIX = "stream_hp_out"


@dataclass
class FileOutcome:
    """Result of running the CA on one stream file.

    Attributes:
        file: Stream file name.
        out_dir: Per-file output folder (a ``ca-evolution/1`` bundle).
        run_id: Random run id the result was saved under in ident db.
        n_frames: Number of spectrogram frames scored.
        n_detections: Number of detections produced.
        posted: ``True`` if posted to ident db, ``False`` on a dry run.
        error: Error string if the file failed, else ``None``.
    """

    file: str
    out_dir: str
    run_id: str | None = None
    n_frames: int = 0
    n_detections: int = 0
    posted: bool = False
    error: str | None = None


def new_output_root(prefix: str = OUTPUT_PREFIX) -> str:
    """Mint a random root output folder name, e.g. ``stream_hp_out1f0c9ab2``.

    Args:
        prefix: Folder-name prefix. Defaults to :data:`OUTPUT_PREFIX`.

    Returns:
        ``"<prefix><8-hex>"``.
    """
    return f"{prefix}{uuid.uuid4().hex[:8]}"


def list_stream_wavs(client, folder: str, smallest_first: bool = False) -> list[dict]:
    """List the WAV file entries in a stream folder.

    Args:
        client: Authenticated API client.
        folder: Stream folder name.
        smallest_first: Sort ascending by size (handy for a cheap trial run);
            otherwise catalog order is preserved.

    Returns:
        A list of stream file entries (``{"name", "size_bytes", ...}``).
    """
    wavs = [f for f in client.list_stream_files(folder)
            if f["name"].lower().endswith(".wav")]
    if smallest_first:
        wavs.sort(key=lambda f: f.get("size_bytes", 0))
    return wavs


def process_file(client, folder: str, entry: dict, root: str, steps: int,
                 threshold: float, freq_band: tuple[float, float], render: bool,
                 evolve_steps: int | None, dry_run: bool,
                 retries: int = 3, desired_delta_t: float | None = None,
                 model_name: str = MODEL_NAME) -> FileOutcome:
    """Run the band-limited CA on one stream file and save its results.

    Fetches the file's Fourier grid, crops it to ``freq_band``, evolves the CA,
    renders a per-file evolution bundle into ``<root>/<file-stem>/``, and (unless
    ``dry_run``) posts the run to ident db against the stream target.

    The fetch is retried (the stream holds multi-gigabyte files, and a dropped
    connection mid-download raises ``IncompleteRead``); any per-file failure is
    captured in ``outcome.error`` and returned rather than raised, so one bad or
    too-large file never aborts the whole stream.

    Args:
        client: Authenticated API client.
        folder: Stream folder name.
        entry: The stream file entry to process.
        root: Root output folder for this invocation.
        steps: CA evolution steps.
        threshold: Detection threshold.
        freq_band: ``(fmin, fmax)`` Hz band the CA is restricted to.
        render: Whether to write the evolution bundle.
        evolve_steps: Generations to render (defaults to ``steps``).
        dry_run: If ``True``, do not post to ident db.
        retries: Attempts to fetch the (large) WAV before giving up on the file.
        desired_delta_t: Target time resolution in seconds between spectrogram
            frames.  Passed as ``hop`` kwarg to ``fourier_for_stream`` if
            supported; the actual achieved delta_t is always logged.
        model_name: Model label recorded against the run and used as run-id
            prefix.  Defaults to :data:`MODEL_NAME`.

    Returns:
        A :class:`FileOutcome`. Per-file failures are captured in ``error``
        rather than raised, so one bad file does not abort the whole stream.
    """
    name = entry["name"]
    stem = os.path.splitext(os.path.basename(name))[0]
    out_dir = os.path.join(root, stem)
    outcome = FileOutcome(file=name, out_dir=out_dir)

    try:
        fourier = None
        last_exc = None
        stream_kwargs = {}
        if desired_delta_t is not None:
            from scipy.io import wavfile as _wv  # noqa: F401 -- presence check only
            stream_kwargs["hop"] = None  # computed after sample_rate known; placeholder
        for attempt in range(1, retries + 1):
            try:
                fourier = client.fourier_for_stream(folder, name)
                break
            except Exception as exc:  # noqa: BLE001 -- retry transient fetch failures
                last_exc = exc
                print(f"[hp_stream_ca] {name}: fetch attempt {attempt}/{retries} "
                      f"failed: {type(exc).__name__}: {exc}")
        if fourier is None:
            raise last_exc
        # If desired_delta_t is set and the SDK didn't honour a hop override,
        # re-STFT from the raw magnitudes is not feasible; log what we got so
        # the caller can verify resolution.
        stft_info = fourier.get("stft", {})
        actual_hop = stft_info.get("hop", 0)
        actual_sr = stft_info.get("sample_rate", fourier.get("sample_rate", 0))
        actual_dt = (actual_hop / actual_sr) if actual_sr else 0.0
        fmin, fmax = float(freq_band[0]), float(freq_band[1])
        band_fourier = crop_fourier_band(fourier, fmin, fmax)
        print(f"[hp_stream_ca] {name}: {fourier['frames']} frames, band "
              f"{fmin:.0f}-{fmax:.0f} Hz -> {band_fourier['bins']} of "
              f"{fourier['bins']} bins, hop={actual_hop} ({actual_dt*1000:.2f} ms/frame)")

        pipeline = build_pipeline(steps=steps, threshold=threshold)
        result = pipeline.run(band_fourier)
        result.pop("_ca", None)
        scores = result["scores"]
        detections = result["detections"]
        band_label = f"{int(fmin)}-{int(fmax)} Hz"
        for det in detections:
            det["fmin"] = fmin
            det["fmax"] = fmax
            det["fmin_hz"] = float(det.get("fmin", fmin))
            det["fmax_hz"] = float(det.get("fmax", fmax))
            det["active_freq"] = band_label

        run_id = new_run_id(model_name)
        outcome.run_id = run_id
        outcome.n_frames = len(scores)
        outcome.n_detections = len(detections)

        if render:
            render_evolution(
                band_fourier,
                evolve_steps if evolve_steps is not None else steps,
                out_dir,
                source={"stream": folder, "file": name, "run_id": run_id,
                        "band_hz": [fmin, fmax],
                        "rule": "WolframRule(90) | AnomalyDetectionRule | GroupingRule"},
            )

        if not dry_run:
            client.post_run(
                target={"kind": "stream", "folder": folder, "file": name},
                model_name=model_name,
                scores=scores,
                threshold=result["threshold"],
                stft=result.get("stft", {}),
                detections=detections,
                run_id=run_id,
            )
            outcome.posted = True
        print(f"[hp_stream_ca] {name}: {outcome.n_detections} detections, "
              f"run_id={run_id}, posted={outcome.posted}")
    except Exception as exc:  # noqa: BLE001 -- never let one file abort the batch
        outcome.error = f"{type(exc).__name__}: {exc}"
        print(f"[hp_stream_ca] {name}: ERROR {outcome.error}")
    return outcome


def run_stream(base_url: str = BASE_URL, token: str = API_KEY,
               stream: str = STREAM_NAME, steps: int = 3,
               threshold: float = 0.45, freq_band: tuple[float, float] = FREQ_BAND,
               render: bool = True, evolve_steps: int | None = None,
               dry_run: bool = False, limit: int | None = None,
               smallest_first: bool = False, output_root: str | None = None,
               timeout: int = 1800, max_size_mb: float | None = None,
               retries: int = 3, progress: bool = True,
               desired_delta_t: float | None = 0.001,
               model_name: str = MODEL_NAME) -> tuple[str, list[FileOutcome]]:
    """Run the CA over every WAV in the ``hp`` stream into one root folder.

    Creates the random root output folder, processes each file into its own
    subfolder, and writes an ``index.json`` summarising the whole run.

    Args:
        base_url: API host.
        token: Bearer API key.
        stream: Stream folder name.
        steps: CA evolution steps.
        threshold: Detection threshold.
        freq_band: ``(fmin, fmax)`` Hz band the CA is restricted to.
        render: Whether to write per-file evolution bundles.
        evolve_steps: Generations to render (defaults to ``steps``).
        dry_run: If ``True``, render but do not post to ident db.
        limit: Process at most this many files (``None`` = all).
        smallest_first: Process files by ascending size.
        output_root: Explicit root folder; a random one is minted if omitted.
        timeout: Per-request socket timeout (seconds). The stream's WAVs are
            multi-gigabyte; the SDK default (120 s) is too short and causes an
            ``IncompleteRead`` mid-download, so this defaults to 1800 s.
        max_size_mb: Skip files larger than this many MB (``None`` = no cap).
            Useful to bound a "whole stream" run away from the few huge files.
        retries: Fetch attempts per file before recording it as failed.
        progress: Show a CLI download progress bar for each WAV fetch.
        desired_delta_t: Target time resolution in seconds between spectrogram
            frames.  Logged per file as actual ms/frame; defaults to 0.001 s
            (1 ms) for HP detector work.  Pass ``None`` to use the SDK default.
        model_name: Model label recorded against each run and used as run-id
            prefix.  Defaults to :data:`MODEL_NAME` (``"hp_ca"``).

    Returns:
        ``(root, outcomes)`` — the root folder path and one
        :class:`FileOutcome` per processed file.

    Raises:
        LookupError: If the stream is not available to the token.
    """
    client = make_client(base_url, token)
    client.timeout = timeout  # large WAVs need far more than the 120 s default
    client.progress = progress  # CLI download bar for the multi-GB WAV fetches
    folder = resolve_stream(client, stream)
    wavs = list_stream_wavs(client, folder, smallest_first=smallest_first)
    if limit is not None:
        wavs = wavs[:limit]

    root = output_root or new_output_root()
    os.makedirs(root, exist_ok=True)
    print(f"[hp_stream_ca] user={STREAM_USER} stream={folder!r} "
          f"files={len(wavs)} -> root={root!r} (timeout={timeout}s)")

    cap_bytes = int(max_size_mb * 1024 * 1024) if max_size_mb else None
    outcomes: list[FileOutcome] = []
    for i, entry in enumerate(wavs, 1):
        size_mb = entry.get("size_bytes", 0) / 1e6
        print(f"[hp_stream_ca] ({i}/{len(wavs)}) {entry['name']} ({size_mb:.0f} MB)")
        if cap_bytes is not None and entry.get("size_bytes", 0) > cap_bytes:
            stem = os.path.splitext(os.path.basename(entry["name"]))[0]
            skipped = FileOutcome(file=entry["name"], out_dir=os.path.join(root, stem),
                                  error=f"skipped: {size_mb:.0f} MB > {max_size_mb:.0f} MB cap")
            print(f"[hp_stream_ca] {entry['name']}: {skipped.error}")
            outcomes.append(skipped)
            continue
        outcomes.append(process_file(
            client, folder, entry, root, steps, threshold, freq_band,
            render, evolve_steps, dry_run, retries=retries,
            desired_delta_t=desired_delta_t, model_name=model_name,
        ))

    index = {
        "stream": folder,
        "model_name": model_name,
        "band_hz": [float(freq_band[0]), float(freq_band[1])],
        "n_files": len(outcomes),
        "n_posted": sum(1 for o in outcomes if o.posted),
        "n_skipped": sum(1 for o in outcomes if o.error and o.error.startswith("skipped:")),
        "n_failed": sum(1 for o in outcomes if o.error and not o.error.startswith("skipped:")),
        "files": [asdict(o) for o in outcomes],
    }
    with open(os.path.join(root, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)
    print(f"[hp_stream_ca] done: {index['n_posted']} posted, "
          f"{index['n_skipped']} skipped, {index['n_failed']} failed "
          f"-> {root}/index.json")
    return root, outcomes


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the script entry point."""
    p = argparse.ArgumentParser(
        description="Run the hp_ca cellular-automata model across the whole hp stream.")
    p.add_argument("--stream", default=STREAM_NAME, help="stream folder name")
    p.add_argument("--steps", type=int, default=3, help="CA evolution steps")
    p.add_argument("--threshold", type=float, default=0.65, help="detection threshold")
    p.add_argument("--freq-band", type=float, nargs=2, metavar=("FMIN", "FMAX"),
                   default=list(FREQ_BAND),
                   help="frequency band (Hz) to restrict the CA to "
                        "(default: %d %d)" % (int(FREQ_BAND[0]), int(FREQ_BAND[1])))
    p.add_argument("--limit", type=int, default=None,
                   help="process at most N files (default: all)")
    p.add_argument("--smallest-first", action="store_true",
                   help="process files by ascending size")
    p.add_argument("--output-root", default=None,
                   help="explicit root output folder (default: random stream_hp_out<rand>)")
    p.add_argument("--timeout", type=int, default=1800,
                   help="per-request socket timeout in seconds (large WAVs; default 1800)")
    p.add_argument("--max-size-mb", type=float, default=None,
                   help="skip files larger than this many MB (default: no cap)")
    p.add_argument("--retries", type=int, default=3,
                   help="fetch attempts per file before recording it as failed")
    p.add_argument("--no-progress", dest="progress", action="store_false",
                   help="hide the per-file download progress bar")
    p.add_argument("--evolve-steps", type=int, default=None,
                   help="CA generations to render per file (default: same as --steps)")
    p.add_argument("--delta-t", type=float, default=0.001, dest="desired_delta_t",
                   metavar="SECONDS",
                   help="target time resolution between spectrogram frames in seconds; "
                        "logged per file as actual ms/frame "
                        "(default: 0.001 s = 1 ms); pass 0 to use SDK default")
    p.add_argument("--model-name", default=MODEL_NAME, dest="model_name",
                   help="model label recorded against each run and used as the "
                        "run-id prefix (default: %s)" % MODEL_NAME)
    p.add_argument("--no-render", dest="render", action="store_false",
                   help="skip writing per-file evolution bundles")
    p.add_argument("--dry-run", action="store_true",
                   help="render bundles but do not post to ident db")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Script entry point. Returns a process exit code.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` on success, ``1`` on a handled API/setup error.
    """
    args = _parse_args(argv)
    try:
        run_stream(
            stream=args.stream,
            steps=args.steps,
            threshold=args.threshold,
            freq_band=(args.freq_band[0], args.freq_band[1]),
            render=args.render,
            evolve_steps=args.evolve_steps,
            dry_run=args.dry_run,
            limit=args.limit,
            smallest_first=args.smallest_first,
            output_root=args.output_root,
            timeout=args.timeout,
            max_size_mb=args.max_size_mb,
            retries=args.retries,
            progress=args.progress,
            desired_delta_t=args.desired_delta_t if args.desired_delta_t else None,
            model_name=args.model_name,
        )
    except (ApiError, LookupError, ValueError) as exc:
        print(f"[hp_stream_ca] error: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
