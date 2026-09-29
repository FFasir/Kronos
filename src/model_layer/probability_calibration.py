"""Probability calibration utilities for realized Kronos forecasts."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
from sklearn.isotonic import IsotonicRegression


def normal_probability_up(pred_close: float, last_close: float, pred_std: float) -> float:
    """Estimate an uncalibrated probability from forecast sample dispersion."""
    if np.isfinite(pred_std) and pred_std > 1e-8:
        value = NormalDist().cdf((pred_close - last_close) / pred_std)
        return float(np.clip(value, 0.001, 0.999))
    return 0.5


def apply_probability_calibration(values, artifact: dict) -> np.ndarray:
    """Apply a JSON-serializable isotonic artifact or the identity mapping."""
    probabilities = np.clip(np.asarray(values, dtype=float), 0.001, 0.999)
    if not artifact.get("enabled") or artifact.get("method") != "isotonic":
        return probabilities
    x = np.asarray(artifact["x_thresholds"], dtype=float)
    y = np.asarray(artifact["y_thresholds"], dtype=float)
    if len(x) < 2 or len(x) != len(y):
        raise ValueError("invalid isotonic calibration thresholds")
    return np.clip(np.interp(probabilities, x, y), 0.001, 0.999)


def probability_metrics(probabilities, outcomes, bins: int = 10) -> dict:
    """Return Brier, log loss, ECE, and threshold accuracy."""
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 0.001, 0.999)
    outcomes = np.asarray(outcomes, dtype=int)
    if probabilities.shape != outcomes.shape or probabilities.ndim != 1:
        raise ValueError("probabilities and outcomes must be matching one-dimensional arrays")
    if not len(probabilities):
        raise ValueError("at least one probability is required")
    brier = float(np.mean(np.square(probabilities - outcomes)))
    log_loss = float(-np.mean(
        outcomes * np.log(probabilities) + (1 - outcomes) * np.log(1 - probabilities)
    ))
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probabilities >= edges[index]) & (probabilities <= edges[index + 1])
        else:
            mask = (probabilities >= edges[index]) & (probabilities < edges[index + 1])
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(probabilities[mask].mean()) - float(outcomes[mask].mean())
            )
    return {
        "brier": brier,
        "log_loss": log_loss,
        "ece": float(ece),
        "directional_accuracy": float(np.mean((probabilities >= 0.5) == outcomes)),
        "mean_probability": float(probabilities.mean()),
        "observed_up_rate": float(outcomes.mean()),
    }


def fit_probability_calibrator(
    probabilities,
    outcomes,
    target_dates,
    fit_fraction: float = 0.7,
) -> dict:
    """Fit isotonic calibration and enable it only after chronological validation."""
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 0.001, 0.999)
    outcomes = np.asarray(outcomes, dtype=int)
    target_dates = np.asarray(target_dates, dtype=str)
    if not (len(probabilities) == len(outcomes) == len(target_dates)):
        raise ValueError("calibration inputs must have equal lengths")
    if len(probabilities) < 50:
        raise ValueError("at least 50 calibration observations are required")
    if not 0.5 <= fit_fraction < 1.0:
        raise ValueError("fit_fraction must be in [0.5, 1.0)")

    unique_dates = np.unique(target_dates)
    if len(unique_dates) < 2:
        raise ValueError("calibration requires at least two target dates")
    split_index = min(max(int(len(unique_dates) * fit_fraction), 1), len(unique_dates) - 1)
    fit_dates = set(unique_dates[:split_index])
    fit_mask = np.array([item in fit_dates for item in target_dates])
    holdout_mask = ~fit_mask
    if len(np.unique(outcomes[fit_mask])) < 2 or not holdout_mask.any():
        raise ValueError("chronological calibration split must contain both classes and a holdout")

    candidate = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
    candidate.fit(probabilities[fit_mask], outcomes[fit_mask])
    raw_holdout = probability_metrics(probabilities[holdout_mask], outcomes[holdout_mask])
    calibrated_holdout_values = candidate.predict(probabilities[holdout_mask])
    calibrated_holdout = probability_metrics(calibrated_holdout_values, outcomes[holdout_mask])
    enabled = (
        calibrated_holdout["brier"] < raw_holdout["brier"]
        and calibrated_holdout["log_loss"] <= raw_holdout["log_loss"] + 0.01
    )

    # Once the method passes the chronological holdout gate, refit on all
    # pre-target validation outcomes for deployment into the next month.
    deployment = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
    deployment.fit(probabilities, outcomes)
    return {
        "version": 1,
        "enabled": bool(enabled),
        "method": "isotonic" if enabled else "identity",
        "fit_fraction": fit_fraction,
        "fit_count": int(fit_mask.sum()),
        "holdout_count": int(holdout_mask.sum()),
        "split_target_date": str(unique_dates[split_index]),
        "x_thresholds": deployment.X_thresholds_.tolist(),
        "y_thresholds": deployment.y_thresholds_.tolist(),
        "holdout_metrics": {
            "raw": raw_holdout,
            "calibrated_candidate": calibrated_holdout,
        },
        "all_validation_raw_metrics": probability_metrics(probabilities, outcomes),
    }
