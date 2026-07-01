#!/usr/bin/env python3
# Vixen Intelligence c.2026

"""anom_ca_pub.py — ProtectedDetector source for the hp anomaly CA model.

This file implements the identdynamics ProtectedDetector contract and is the
source submitted to the brahma build server (tools/build_protected_model.py),
which obfuscates it with PyArmor and packages it for distribution to rdtandon.

Once built and registered, rdtandon downloads and runs it with::

    from identdynamics import Client, fetch_protected, score_fourier

    client = Client("https://goident.ai", token="<rdtandon-token>")
    det = fetch_protected(client, "hp_ca")
    fourier = client.fourier_for_file(file_id)          # or fourier_for_path(...)
    scores = score_fourier(det, fourier)

Runtime parameters can be tuned per-run without a rebuild::

    det = fetch_protected(client, "hp_ca",
                          params={"fmin": 80000, "fmax": 160000, "threshold": 0.5})

Contract surface required by the SDK loader
-------------------------------------------
    detector(manifest=None) -> AnomalyCADetector    top-level factory function
    .name              str
    .version           str
    .default_threshold float    suggested decision threshold in [0, 1]
    .expects           dict     STFT hints for the consumer
    .score(grid)       -> float32[frames] in [0, 1]
    .configure(**params)        late-binding for per-run tuning
"""

from __future__ import annotations

import numpy as np
from brahma_cellular import (
    Pipeline,
    WolframRule,
    AnomalyDetectionRule,
    GroupingRule,
)

_NAME = "brahma_ca_03252_anomaly"
_VERSION = "1.0.0"
_DEFAULT_THRESHOLD = 0.65
_DEFAULT_STEPS = 4
_DEFAULT_MIN_SIGMA = 1.5
_DEFAULT_FMIN = 1000.0   # Hz — lower edge of the target HP band
_DEFAULT_FMAX = 8000.0   # Hz — upper edge
_DESIRED_DELTA_T = 0.01    # 1 ms target frame spacing (set hop = round(0.001 * sr))


class AnomalyCADetector:
    """Anomaly cellular-automata detector conforming to the ProtectedDetector contract.

    Runs a Wolfram Rule-90 / z-score anomaly / grouping CA pipeline over the
    supplied Fourier grid, restricted to the configured frequency band.  Per-frame
    scores in [0, 1] are returned; thresholding and detection labelling are left
    to the consumer (identdynamics.score_fourier / pipeline helpers).

    Runtime parameters (via configure or fetch_protected params=…):
        fmin          float   lower band edge in Hz  (default 100 000)
        fmax          float   upper band edge in Hz  (default 150 000)
        steps         int     CA evolution steps      (default 3)
        min_sigma     float   anomaly z-score cutoff  (default 1.5)
        threshold     float   detection threshold     (default 0.45)
    """

    name: str = _NAME
    version: str = _VERSION
    default_threshold: float = _DEFAULT_THRESHOLD

    def __init__(self, manifest: dict | None = None) -> None:
        self._fmin: float = _DEFAULT_FMIN
        self._fmax: float = _DEFAULT_FMAX
        self._desired_delta_t: float = _DESIRED_DELTA_T
        self._steps: int = _DEFAULT_STEPS
        self._min_sigma: float = _DEFAULT_MIN_SIGMA
        self._threshold: float = _DEFAULT_THRESHOLD

        # Apply any params baked into the bundle manifest at build time.
        params = (manifest or {}).get("params", {})
        if params:
            self._apply(params)

    @property
    def expects(self) -> dict:
        """STFT hints for the consumer, reflecting the current configuration.

        ``desired_delta_t`` tells the consumer what hop to use:
        ``hop = round(desired_delta_t * sample_rate)``.
        ``fmin`` / ``fmax`` indicate the active band so the consumer can
        optionally pre-crop the grid before calling :meth:`score`.
        """
        return {
            "window": "hann",
            "desired_delta_t": self._desired_delta_t,
            "fmin": self._fmin,
            "fmax": self._fmax,
        }

    def configure(self, **params) -> None:
        """Late-bind runtime parameters.

        Called automatically by fetch_protected / configure_detector when the
        consumer passes params=…. Also callable directly to re-tune between files.

        Accepted keys:
            fmin (float): lower frequency band edge in Hz.
            fmax (float): upper frequency band edge in Hz.
            desired_delta_t (float): target frame spacing in seconds; the
                consumer should set ``hop = round(desired_delta_t * sample_rate)``
                when computing the Fourier grid.
            steps (int): CA evolution steps.
            min_sigma (float): anomaly z-score cutoff.
            threshold (float): detection threshold in [0, 1].
        """
        self._apply(params)

    def _apply(self, params: dict) -> None:
        if "fmin" in params:
            self._fmin = float(params["fmin"])
        if "fmax" in params:
            self._fmax = float(params["fmax"])
        if "desired_delta_t" in params:
            self._desired_delta_t = float(params["desired_delta_t"])
        if "steps" in params:
            self._steps = int(params["steps"])
        if "min_sigma" in params:
            self._min_sigma = float(params["min_sigma"])
        if "threshold" in params:
            self._threshold = float(params["threshold"])

    # ------------------------------------------------------------------
    # ProtectedDetector contract
    # ------------------------------------------------------------------

    def score(self, grid) -> np.ndarray:
        """Run the anomaly CA and return ``float32[frames]`` scores in ``[0, 1]``.

        Args:
            grid: A :class:`identdynamics.FrameGrid` (magnitudes, frames, bins,
                sample_rate, fft, hop, window).

        Returns:
            ``float32`` array of length ``grid.frames``.
        """
        fourier = {
            "magnitudes": np.asarray(grid.magnitudes, dtype=np.float64),
            "frames": int(grid.frames),
            "bins": int(grid.bins),
            "sample_rate": float(grid.sample_rate),
            "stft": {
                "fft": int(grid.fft),
                "hop": int(grid.hop),
                "window": str(grid.window),
                "sample_rate": float(grid.sample_rate),
            },
        }

        band = _crop_band(fourier, self._fmin, self._fmax)

        pipeline = Pipeline(
            rule=(
                WolframRule(rule_number=90, steps=self._steps)
                | AnomalyDetectionRule(min_sigma=self._min_sigma)
                | GroupingRule()
            ),
            model_name=self.name,
            threshold=self._threshold,
        )
        result = pipeline.run(band)
        return np.clip(np.asarray(result["scores"], dtype=np.float32), 0.0, 1.0)


def _crop_band(fourier: dict, fmin: float, fmax: float) -> dict:
    """Slice the Fourier grid to the [fmin, fmax] Hz band.

    Mirrors hp_ca.crop_fourier_band but is inlined here so the published
    package has no dependency on internal scripts.
    """
    frames = int(fourier["frames"])
    bins = int(fourier["bins"])
    stft = fourier.get("stft", {})
    fft_size = int(stft.get("fft", 1024))
    sr = float(stft.get("sample_rate") or fourier.get("sample_rate") or 1)

    bin_hz = np.arange(bins) * (sr / fft_size)
    mask = (bin_hz >= fmin) & (bin_hz <= fmax)

    if not mask.any():
        return dict(fourier, bin_offset=0)

    lo = int(mask.argmax())
    hi = int(len(mask) - mask[::-1].argmax())
    grid = np.asarray(fourier["magnitudes"], dtype=np.float64).reshape(frames, bins)
    cropped = np.ascontiguousarray(grid[:, lo:hi])

    out = dict(fourier)
    out["magnitudes"] = cropped
    out["bins"] = cropped.shape[1]
    out["bin_offset"] = lo
    return out


def detector(manifest: dict | None = None) -> AnomalyCADetector:
    """Top-level factory required by the identdynamics ProtectedDetector contract.

    The brahma build server calls this to verify the contract before obfuscating.
    At run time the SDK loader calls it to instantiate the detector.

    Args:
        manifest: Optional config dict from the bundle's ``manifest.json``
            (merged with any runtime ``params`` passed to ``fetch_protected``).

    Returns:
        A ready-to-use :class:`AnomalyCADetector`.
    """
    return AnomalyCADetector(manifest)
