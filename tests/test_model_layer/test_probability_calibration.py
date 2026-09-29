import numpy as np

from src.model_layer.probability_calibration import (
    apply_probability_calibration,
    fit_probability_calibrator,
    probability_metrics,
)


def test_isotonic_calibration_requires_holdout_improvement_before_enablement():
    probabilities = []
    outcomes = []
    dates = []
    for day in range(10):
        probabilities.extend([0.01] * 5 + [0.99] * 5)
        outcomes.extend([0, 0, 0, 0, 1, 0, 0, 1, 1, 1])
        dates.extend([f"2025-08-{day + 1:02d}"] * 10)

    artifact = fit_probability_calibrator(probabilities, outcomes, dates)
    calibrated = apply_probability_calibration(probabilities, artifact)

    assert artifact["enabled"] is True
    assert artifact["method"] == "isotonic"
    assert probability_metrics(calibrated, outcomes)["brier"] < probability_metrics(
        probabilities, outcomes
    )["brier"]
    assert np.all((calibrated >= 0.001) & (calibrated <= 0.999))


def test_disabled_calibration_is_identity():
    values = np.array([0.1, 0.5, 0.9])
    result = apply_probability_calibration(values, {"enabled": False, "method": "identity"})
    assert np.allclose(values, result)
