import sys
from pathlib import Path

import torch
import torch.nn.functional as F


FINETUNE_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "Kronos" / "finetune"
if str(FINETUNE_ROOT) not in sys.path:
    sys.path.insert(0, str(FINETUNE_ROOT))

from dataset import balanced_direction_sample
from utils.training_utils import temporal_weighted_cross_entropy


def test_direction_sampling_is_balanced_unique_and_deterministic():
    up = list(range(0, 80))
    down = list(range(80, 100))
    first = balanced_direction_sample(up, down, 30, seed=42)
    second = balanced_direction_sample(up, down, 30, seed=42)

    assert first == second
    assert len(first) == len(set(first)) == 30
    assert sum(item in up for item in first) == 15
    assert sum(item in down for item in first) == 15


def test_forecast_weight_increases_penalty_for_future_errors():
    targets = torch.zeros((1, 4), dtype=torch.long)
    logits = torch.tensor([[[5.0, -5.0], [5.0, -5.0], [-5.0, 5.0], [-5.0, 5.0]]])

    unweighted, _, _ = temporal_weighted_cross_entropy(
        logits, logits, targets, targets, forecast_start=2, forecast_weight=1.0
    )
    weighted, _, _ = temporal_weighted_cross_entropy(
        logits, logits, targets, targets, forecast_start=2, forecast_weight=4.0
    )

    expected = F.cross_entropy(logits.transpose(1, 2), targets)
    assert torch.allclose(unweighted, expected)
    assert weighted > unweighted
