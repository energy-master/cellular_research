#!/usr/bin/env python3


""" Vixen Intelligence c.2026"""

"""hp_ca — first cellular-automata run against the IDent Dynamics ``hp`` stream.

This module is the minimal end-to-end framework for ``hp_ca``: the inaugural run
of a :mod:`brahma_cellular` cellular-automata (CA) model wired into the
``identdynamics`` SDK. It performs exactly one CA run over one file selected from
the ``hp`` acoustic stream and records the result back into the IDent Dynamics
database ("ident db") as a *run* with a freshly minted random run id.

Pipeline overview:
    1. Authenticate to ``https://goident.ai`` with the owning user's API token
       (the token, not the username, is what authorises the session; the
       username is retained only for provenance/logging).
    2. Resolve the ``hp`` stream folder from the account catalog.
    3. Select a single audio file from that stream (default: the smallest WAV,
       which keeps this first run tractable).
    4. Fetch the file's canonical STFT (Fourier) grid from the server.
    5. Evolve one CA pass over the spectrogram via a :class:`Pipeline`, scoring
       and labelling per-frame detections.
    6. Post the scores + detections to ident db under a random run id, attached
       to the originating stream file so it surfaces in the webapp's Local Runs.

Run it directly::

    python hp_ca.py                      # smallest WAV in the hp stream
    python hp_ca.py --file <name.wav>    # a specific stream file
    python hp_ca.py --select first       # first WAV (catalog order)
    python hp_ca.py --dry-run            # run the CA but do not post to ident db

Example:
    >>> from hp_ca import run_hp_ca
    >>> outcome = run_hp_ca()
    >>> outcome.run_id            # doctest: +SKIP
    'hp_ca-1f0c...e9'
"""

from __future__ import annotations

import argparse
import os
import uuid
from dataclasses import dataclass

from identdynamics import Client, ApiError
from brahma_cellular import (
    Pipeline,
    WolframRule,
    AnomalyDetectionRule,
    GroupingRule,
)

# --- Run configuration --------------------------------------------------------

#: IDent Dynamics API base URL (the webapp host that fronts ident db).
BASE_URL = "https://goident.ai"

#: Owning account for the ``hp`` stream. Informational only — the API key below
#: is what actually authenticates every request.
STREAM_USER = "rdtandon"

#: Bearer token for ``STREAM_USER`` (Account -> API tokens in the webapp).
#: Read from the ``IDENT_API_KEY`` environment variable so the secret never
#: lives in source / git history. Export it before running, e.g.::
#:
#:     export IDENT_API_KEY="<your-token>"
API_KEY = os.environ.get("IDENT_API_KEY", "")

#: Name of the stream folder to draw the single run file from.
STREAM_NAME = "hp"

#: Saved file (Account -> Files) to run on instead of pulling from the stream.
SAVED_FILE = "_20090817_133217_000_WTT20090817_133137.wav"

#: Model label recorded against the run in ident db.
MODEL_NAME = "hp_ca"

#: Root folder under which CA-evolution render bundles are written
#: (one ``<run_id>/`` bundle per run; drop it into the app to view).
BUNDLE_ROOT = "renders"


# --- Result container ---------------------------------------------------------


@dataclass
class RunOutcome:
    """Summary of a single ``hp_ca`` run.

    Attributes:
        run_id: The random run id under which results were saved (or would have
            been saved, on a dry run).
        stream: Stream folder the file was drawn from.
        file: Name of the stream file the CA was run on.
        model_name: Model label recorded against the run.
        n_frames: Number of spectrogram frames scored.
        n_detections: Number of labelled detection events produced.
        threshold: Detection threshold applied to the per-frame score curve.
        posted: ``True`` if the run was written to ident db, ``False`` on a dry
            run.
        response: Raw server response from ``post_run`` (``None`` on a dry run).
        bundle_dir: Path to the CA-evolution render bundle, or ``None`` if not
            rendered.
    """

    run_id: str
    stream: str
    file: str
    model_name: str
    n_frames: int
    n_detections: int
    threshold: float
    posted: bool
    response: dict | None = None
    bundle_dir: str | None = None

    def as_dict(self) -> dict:
        """Return a plain ``dict`` view of this outcome for logging/JSON."""
        return {
            "run_id": self.run_id,
            "stream": self.stream,
            "file": self.file,
            "model_name": self.model_name,
            "n_frames": self.n_frames,
            "n_detections": self.n_detections,
            "threshold": self.threshold,
            "posted": self.posted,
            "response": self.response,
            "bundle_dir": self.bundle_dir,
        }


# --- Framework steps ----------------------------------------------------------


def make_client(base_url: str = BASE_URL, token: str = API_KEY) -> Client:
    """Construct an authenticated IDent Dynamics client.

    Args:
        base_url: Webapp/API host. Defaults to :data:`BASE_URL`.
        token: Bearer API key. Defaults to :data:`API_KEY`.

    Returns:
        A ready-to-use :class:`identdynamics.Client`.

    Raises:
        ValueError: If no token is available (``IDENT_API_KEY`` unset/empty).
    """
    if not token:
        raise ValueError(
            "no API token: set the IDENT_API_KEY environment variable "
            "(export IDENT_API_KEY=<your-token>)"
        )
    return Client(base_url=base_url, token=token)


def resolve_stream(client: Client, name: str = STREAM_NAME) -> str:
    """Confirm the target stream exists on the account and return its folder.

    Args:
        client: Authenticated API client.
        name: Stream folder name to resolve. Defaults to :data:`STREAM_NAME`.

    Returns:
        The resolved stream folder name (echoes ``name`` when present).

    Raises:
        LookupError: If ``name`` is not among the account's granted streams.
    """
    catalog = client.catalog()
    available = [s.get("folder") or s.get("name") for s in catalog.get("streams", [])]
    if name not in available:
        raise LookupError(
            f"stream {name!r} not granted to this token; available: {available}"
        )
    return name


def select_file(client: Client, folder: str, select: str = "smallest",
                file: str | None = None) -> dict:
    """Select exactly one audio file from a stream folder.

    Only WAV files are considered (the Fourier helpers decode PCM WAV). The
    default ``"smallest"`` strategy keeps this first run tractable given the
    stream holds multi-gigabyte recordings.

    Args:
        client: Authenticated API client.
        folder: Stream folder to select from.
        select: Selection strategy when ``file`` is not given — ``"smallest"``
            (fewest bytes) or ``"first"`` (catalog order).
        file: Explicit file name to select; overrides ``select`` when provided.

    Returns:
        The chosen file's catalog entry (``{"name", "size_bytes", ...}``).

    Raises:
        ValueError: If ``select`` is unknown, the folder has no WAV files, or an
            explicit ``file`` is not present in the folder.
    """
    files = client.list_stream_files(folder)
    wavs = [f for f in files if f["name"].lower().endswith(".wav")]
    if not wavs:
        raise ValueError(f"stream {folder!r} has no WAV files to run on")

    if file is not None:
        for f in wavs:
            if f["name"] == file:
                return f
        raise ValueError(f"file {file!r} not found in stream {folder!r}")

    if select == "smallest":
        return min(wavs, key=lambda f: f.get("size_bytes", 0))
    if select == "first":
        return wavs[0]
    raise ValueError(f"unknown select strategy {select!r} (use smallest|first)")


def select_saved_file(client: Client, name: str = SAVED_FILE) -> dict:
    """Select one saved file (Account -> Files) by name.

    The saved-file counterpart to :func:`select_file`: instead of enumerating a
    stream folder, it looks the file up in the account's saved-files list and
    returns its catalog entry, whose ``id`` feeds ``fourier_for_file`` and the
    ``{"kind": "file", "id": ...}`` run target.

    Args:
        client: Authenticated API client.
        name: Saved file name to resolve. Defaults to :data:`SAVED_FILE`.

    Returns:
        The matching saved-file entry (``{"id", "name", ...}``).

    Raises:
        ValueError: If no saved file with ``name`` exists on the account.
    """
    for f in client.list_files():
        if f.get("name") == name:
            return f
    raise ValueError(f"saved file {name!r} not found on this account")


def fourier_for_saved_file(client: Client, file_id: int) -> dict:
    """Fourier grid for a saved file, tolerant of IEEE-float WAVs.

    ``Client.fourier_for_file`` decodes via the stdlib :mod:`wave` module, which
    only understands integer PCM and raises ``wave.Error: unknown format: 3`` on
    IEEE-float WAVs (WAVE_FORMAT_IEEE_FLOAT). This wrapper tries the SDK path
    first and, on that failure, falls back to a SciPy-based decode that handles
    float samples, then runs the *same* reference STFT with the file's canonical
    parameters so the magnitudes line up with ``fourier_for_file``.

    Args:
        client: Authenticated API client.
        file_id: Saved file id.

    Returns:
        The standard Fourier dict ``{magnitudes, frames, bins, sample_rate,
        stft}``.
    """
    try:
        return client.fourier_for_file(file_id)
    except Exception as exc:  # noqa: BLE001 — fall back only on a decode failure
        import io
        import numpy as np
        from scipy.io import wavfile
        from identdynamics import stft

        params = client.stft_params(file_id=file_id)["canonical_stft"]
        sample_rate, data = wavfile.read(io.BytesIO(client.file_wav(file_id)))
        signal = data.astype(np.float64)
        if np.issubdtype(data.dtype, np.integer):
            signal /= float(np.iinfo(data.dtype).max)  # PCM -> [-1, 1]
        if signal.ndim > 1:
            signal = signal.mean(axis=1)  # mix to mono, matching the SDK
        mags, frames, bins = stft(
            signal, params["fft_size"], params["hop_size"], params["window"]
        )
        print(f"[hp_ca] float-WAV fallback decode ({data.dtype}, {exc})")
        return {
            "magnitudes": mags,
            "frames": frames,
            "bins": bins,
            "sample_rate": int(sample_rate),
            "stft": {
                "fft": params["fft_size"], "hop": params["hop_size"],
                "window": params["window"], "sample_rate": int(sample_rate),
            },
        }


def build_rule(steps: int = 3):
    """Build the ``hp_ca`` CA rule chain.

    Pairs a Wolfram rule 90 pass (XOR — emphasises harmonic periodicity, well
    suited to tonal sonar/acoustic content) with a local z-score anomaly detector
    and connected-component grouping. Shared by the detection pipeline and the
    evolution render so the visualization is faithful to what scored detections.

    Args:
        steps: Number of CA evolution steps. Defaults to ``3``.

    Returns:
        A ``brahma_cellular`` rule chain (``CARule``).
    """
    return (
        WolframRule(rule_number=90, steps=steps)
        | AnomalyDetectionRule(min_sigma=1.5)
        | GroupingRule()
    )


def build_pipeline(steps: int = 3, threshold: float = 0.45) -> Pipeline:
    """Build the CA pipeline for the ``hp_ca`` model.

    Args:
        steps: Number of CA evolution steps. Defaults to ``3``.
        threshold: Per-frame score threshold for detections. Defaults to
            ``0.45``.

    Returns:
        A configured :class:`brahma_cellular.Pipeline`.
    """
    return Pipeline(
        rule=build_rule(steps),
        model_name=MODEL_NAME,
        steps=steps,
        threshold=threshold,
        label=True,
    )


def render_evolution(fourier: dict, steps: int, out_dir: str,
                     source: dict | None = None, fps: int = 8):
    """Render the CA grid evolution for this run into a drop-in bundle.

    Uses the SDK's :class:`identdynamics.cellular_automata.cellular_automata` to
    evolve the same rule chain over the spectrogram with history captured, and
    write a ``ca-evolution/1`` bundle (per-generation PNG frames + ``evolution.mp4``
    + ``manifest.json``). Dropping ``out_dir`` into the IDent Dynamics app
    auto-mounts the CA Evolution panel.

    Rendering is best-effort: if the optional render dependencies (matplotlib /
    Pillow, imported lazily by the SDK class) are missing, this logs and returns
    ``None`` rather than failing the run.

    Args:
        fourier: The run's Fourier grid.
        steps: CA generations to evolve and render.
        out_dir: Destination bundle folder.
        source: Provenance recorded in the manifest.
        fps: Playback frame rate for the merged video.

    Returns:
        The manifest dict on success, else ``None``.
    """
    try:
        from identdynamics.cellular_automata import cellular_automata
    except ImportError as exc:  # SDK without the CA module
        print(f"[hp_ca] evolution render skipped (import): {exc}")
        return None
    try:
        renderer = cellular_automata(cmap="viridis", fps=fps)
        manifest = renderer.render_fourier(
            fourier, build_rule(steps), steps, out_dir,
            model_name=MODEL_NAME, source=source,
        )
    except ImportError as exc:  # matplotlib / Pillow not installed
        print(f"[hp_ca] evolution render skipped (deps): {exc}")
        return None
    print(f"[hp_ca] CA evolution bundle: {out_dir} "
          f"({manifest['n_steps']} frames, video={manifest['video']})")
    return manifest


def new_run_id() -> str:
    """Mint a random run id for an ``hp_ca`` run.

    Returns:
        A unique id of the form ``"hp_ca-<uuid4-hex>"``, suitable as a distinct
        ident db run row.
    """
    return f"{MODEL_NAME}-{uuid.uuid4().hex}"


def run_hp_ca(base_url: str = BASE_URL, token: str = API_KEY,
              stream: str = STREAM_NAME, select: str = "smallest",
              file: str | None = None, steps: int = 3,
              threshold: float = 0.45, dry_run: bool = False,
              render: bool = True, evolve_steps: int | None = None,
              bundle_root: str = BUNDLE_ROOT) -> RunOutcome:
    """Execute one CA run over one ``hp`` stream file and save it to ident db.

    This is the top-level orchestrator: authenticate, resolve the stream, select
    a single file, fetch its Fourier grid, evolve one CA pipeline pass, and post
    the scored/labelled result under a freshly generated random run id.

    Args:
        base_url: API host. Defaults to :data:`BASE_URL`.
        token: Bearer API key. Defaults to :data:`API_KEY`.
        stream: Stream folder name. Defaults to :data:`STREAM_NAME`.
        select: File selection strategy (``"smallest"`` or ``"first"``) used
            when ``file`` is not supplied.
        file: Explicit stream file name to run on; overrides ``select``.
        steps: CA evolution steps.
        threshold: Detection threshold.
        dry_run: If ``True``, run the CA but do not post to ident db.
        render: If ``True``, also write a CA-evolution render bundle (drop into
            the app to watch the grid evolve). Independent of ``dry_run``.
        evolve_steps: CA generations to render for the bundle; defaults to
            ``steps`` so the visualization matches the detection run.
        bundle_root: Folder under which the ``<run_id>/`` bundle is written.

    Returns:
        A :class:`RunOutcome` describing the run and (unless ``dry_run``) the
        server response.

    Raises:
        LookupError: If the stream is not available to the token.
        ValueError: If no suitable file can be selected.
        identdynamics.ApiError: If the server rejects the fetch or post.
    """
    client = make_client(base_url, token)

    # --- Stream source (disabled) --------------------------------------------
    # Pulling the run file live from the ``hp`` stream is commented out for now;
    # we run on a saved file instead (see below).
    # folder = resolve_stream(client, stream)
    # chosen = select_file(client, folder, select=select, file=file)
    # name = chosen["name"]
    # print(f"[hp_ca] user={STREAM_USER} stream={folder!r} file={name!r} "
    #       f"({chosen.get('size_bytes', 0) / 1e6:.1f} MB)")
    # fourier = client.fourier_for_stream(folder, name)

    # --- Saved-file source (active) ------------------------------------------
    chosen = select_saved_file(client, SAVED_FILE)
    file_id = chosen["id"]
    name = chosen["name"]
    folder = "saved"  # provenance label for the RunOutcome
    print(f"[hp_ca] user={STREAM_USER} saved file id={file_id} name={name!r} "
          f"({chosen.get('wav_size_bytes', 0) / 1e6:.1f} MB)")

    fourier = fourier_for_saved_file(client, file_id)
    print(f"[hp_ca] fourier: {fourier['frames']} frames x {fourier['bins']} bins "
          f"@ {fourier['sample_rate']} Hz")

    pipeline = build_pipeline(steps=steps, threshold=threshold)
    result = pipeline.run(fourier)
    result.pop("_ca", None)  # CAState handle — not JSON-postable

    scores = result["scores"]
    detections = result["detections"]
    run_id = new_run_id()
    print(f"[hp_ca] CA run: {len(scores)} frames, {len(detections)} detections, "
          f"run_id={run_id}")

    outcome = RunOutcome(
        run_id=run_id,
        stream=folder,
        file=name,
        model_name=MODEL_NAME,
        n_frames=len(scores),
        n_detections=len(detections),
        threshold=threshold,
        posted=False,
    )

    if render:
        out_dir = os.path.join(bundle_root, run_id)
        manifest = render_evolution(
            fourier, evolve_steps if evolve_steps is not None else steps, out_dir,
            source={"stream": folder, "file": name, "run_id": run_id,
                    "rule": "WolframRule(90) | AnomalyDetectionRule | GroupingRule"},
        )
        if manifest is not None:
            outcome.bundle_dir = out_dir

    if dry_run:
        print("[hp_ca] dry-run: not posting to ident db")
        return outcome

    # target = {"kind": "stream", "folder": folder, "file": name}  # stream path
    target = {"kind": "file", "id": file_id}
    response = client.post_run(
        target=target,
        model_name=result["model_name"],
        scores=scores,
        threshold=result["threshold"],
        stft=result.get("stft", {}),
        detections=detections,
        run_id=run_id,
    )
    outcome.posted = True
    outcome.response = response
    print(f"[hp_ca] saved to ident db: {response}")
    return outcome


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the script entry point."""
    p = argparse.ArgumentParser(description="Run one hp_ca CA pass on the hp stream.")
    p.add_argument("--stream", default=STREAM_NAME, help="stream folder name")
    p.add_argument("--select", default="smallest", choices=["smallest", "first"],
                   help="file selection strategy when --file is omitted")
    p.add_argument("--file", default=None, help="explicit stream file to run on")
    p.add_argument("--steps", type=int, default=3, help="CA evolution steps")
    p.add_argument("--threshold", type=float, default=0.45, help="detection threshold")
    p.add_argument("--dry-run", action="store_true",
                   help="run the CA but do not post to ident db")
    p.add_argument("--no-render", dest="render", action="store_false",
                   help="skip writing the CA-evolution render bundle")
    p.add_argument("--evolve-steps", type=int, default=None,
                   help="CA generations to render (default: same as --steps)")
    p.add_argument("--bundle-root", default=BUNDLE_ROOT,
                   help="folder to write the <run_id>/ render bundle under")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Script entry point. Returns a process exit code.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` on success, ``1`` on a handled API/selection error.
    """
    args = _parse_args(argv)
    try:
        run_hp_ca(
            stream=args.stream,
            select=args.select,
            file=args.file,
            steps=args.steps,
            threshold=args.threshold,
            dry_run=args.dry_run,
            render=args.render,
            evolve_steps=args.evolve_steps,
            bundle_root=args.bundle_root,
        )
    except (ApiError, LookupError, ValueError) as exc:
        print(f"[hp_ca] error: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
