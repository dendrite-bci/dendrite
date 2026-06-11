"""Regression-based EOG (ocular) artifact correction for online EEG.

Regresses dedicated EOG reference channels out of the EEG:

    EEG_clean = EEG - B @ EOG_ref

This is the Gratton & Coles (1983) / Schlögl et al. (2007) method — the standard
real-time approach when EOG electrodes are recorded. Two details make it work on
real (monopolar, common-referenced) montages:

* The regression is fit/applied on **CAR-referenced, phase-matched** signals: the
  EOG reference is referenced to the EEG common-average (cancelling the shared
  reference that otherwise makes monopolar EOG correlate with EEG across all bands)
  and band-limited to the ocular range, so ``B @ EOG`` cannot touch mu/β.
* ``B`` is estimated **adaptively** — running covariance accumulators are updated
  per chunk and ``B`` is refit periodically — so it converges to a stable fit over
  the first minutes. The accumulators have no forgetting factor, so the fit settles
  to the session-wide least squares rather than tracking later drift; a short fixed
  window, by contrast, doesn't even generalise across the whole session.
"""

import numpy as np

_RIDGE_LAMBDA = 1e-6       # relative ridge term for numerical stability


def _solve_ridge(cross: np.ndarray, gram: np.ndarray) -> np.ndarray | None:
    """Ridge least squares: ``B = cross @ inv(gram + λI)`` (λ scaled by tr(gram))."""
    ridge = _RIDGE_LAMBDA * np.trace(gram) / max(gram.shape[0], 1)
    try:
        return cross @ np.linalg.inv(gram + ridge * np.eye(gram.shape[0]))
    except np.linalg.LinAlgError:
        return None


class AdaptiveEOGRegression:
    """Streaming EOG→EEG regression with periodic refit.

    Maintains the regression's sufficient statistics — ``G = ref·refᵀ`` and
    ``C = eeg·refᵀ`` — as running covariance accumulators, refits
    ``B = C·(G+λI)⁻¹`` every ``refit_s`` once ``min_fit_s`` of data has been
    seen, and subtracts ``B·ref`` from the EEG. Inputs are the already-processed
    (CAR + band-passed) EEG and the EEG-referenced, ocular-band reference.

    The accumulators are never decayed, so ``B`` converges to the session-wide
    least-squares fit and then holds steady — it does not chase later electrode
    drift (that would need an exponential forgetting factor).

    Cheap: ``G`` is ``n_ref x n_ref`` and the refit is one small inverse.
    """

    def __init__(
        self,
        n_eeg: int,
        n_eog: int,
        sample_rate: float,
        *,
        min_fit_s: float = 30.0,
        refit_s: float = 2.0,
    ) -> None:
        self.n_eeg = n_eeg
        self.n_eog = n_eog
        self._min_fit = min_fit_s * sample_rate
        self._refit_interval = refit_s * sample_rate
        self.reset()

    def reset(self) -> None:
        self._G = np.zeros((self.n_eog, self.n_eog))
        self._C = np.zeros((self.n_eeg, self.n_eog))
        self._n = 0.0
        self._since_refit = 0.0
        self.B: np.ndarray | None = None

    def update_and_apply(
        self, eeg_low: np.ndarray, eeg_high: np.ndarray, ref: np.ndarray,
    ) -> np.ndarray:
        """Regress ocular out of the phase-matched low band, recombine with the high
        band: returns ``(eeg_low - B·ref) + eeg_high``.

        ``eeg_low`` and ``ref`` are filtered to the SAME ocular band ``[lowcut, 6]``,
        so the regression and the subtraction share the same causal-filter phase —
        without this the estimate is time-shifted and *adds* a second blink. ``eeg_high``
        (``[6, highcut]``) carries mu/β and passes through untouched. Before the first
        refit, ``B`` is None and the output is just ``eeg_low + eeg_high``. No-op (returns
        ``eeg_low + eeg_high``) on shape mismatch; does not modify the inputs.
        """
        recombined = eeg_low + eeg_high
        if (
            eeg_low.shape[0] != self.n_eeg or eeg_high.shape[0] != self.n_eeg
            or ref.shape[0] != self.n_eog or eeg_low.shape[1] != ref.shape[1]
            or eeg_high.shape[1] != ref.shape[1] or eeg_low.shape[1] == 0
        ):
            return recombined

        t = ref.shape[1]
        # Raw second moments, not mean-centred covariances: both inputs are
        # high-passed (lowcut ≥ 0.5 Hz) and CAR'd, so E[ref] ≈ E[eeg_low] ≈ 0 and
        # the omitted centring terms are negligible — no explicit centring needed.
        self._G += ref @ ref.T
        self._C += eeg_low @ ref.T
        self._n += t
        self._since_refit += t

        if self._n >= self._min_fit and self._since_refit >= self._refit_interval:
            b = _solve_ridge(self._C, self._G)
            if b is not None:
                self.B = b
            self._since_refit = 0.0

        if self.B is not None:
            return (eeg_low - self.B @ ref) + eeg_high
        return recombined
