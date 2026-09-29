import os
import random
import datetime
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F


def setup_ddp():
    """
    Initializes the distributed data parallel environment.

    This function relies on environment variables set by `torchrun` or a similar
    launcher. It initializes the process group and sets the CUDA device for the
    current process.

    Returns:
        tuple: A tuple containing (rank, world_size, local_rank).
    """
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available.")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", device_id=device)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(
        f"[DDP Setup] Global Rank: {rank}/{world_size}, "
        f"Local Rank (GPU): {local_rank} on device {torch.cuda.current_device()}"
    )
    return rank, world_size, local_rank


def cleanup_ddp():
    """Cleans up the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int, rank: int = 0):
    """
    Sets the random seed for reproducibility across all relevant libraries.

    Args:
        seed (int): The base seed value.
        rank (int): The process rank, used to ensure different processes have
                    different seeds, which can be important for data loading.
    """
    actual_seed = seed + rank
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)
        # The two lines below can impact performance, so they are often
        # reserved for final experiments where reproducibility is critical.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_model_size(model: torch.nn.Module) -> str:
    """
    Calculates the number of trainable parameters in a PyTorch model and returns
    it as a human-readable string.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        str: A string representing the model size (e.g., "175.0B", "7.1M", "50.5K").
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if total_params >= 1e9:
        return f"{total_params / 1e9:.1f}B"  # Billions
    elif total_params >= 1e6:
        return f"{total_params / 1e6:.1f}M"  # Millions
    else:
        return f"{total_params / 1e3:.1f}K"  # Thousands


def reduce_tensor(tensor: torch.Tensor, world_size: int, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """
    Reduces a tensor's value across all processes in a distributed setup.

    Args:
        tensor (torch.Tensor): The tensor to be reduced.
        world_size (int): The total number of processes.
        op (dist.ReduceOp, optional): The reduction operation (SUM, AVG, etc.).
                                      Defaults to dist.ReduceOp.SUM.

    Returns:
        torch.Tensor: The reduced tensor, which will be identical on all processes.
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=op)
    # Note: `dist.ReduceOp.AVG` is available in newer torch versions.
    # For compatibility, manual division is sometimes used after a SUM.
    if op == dist.ReduceOp.AVG:
        rt /= world_size
    return rt


def format_time(seconds: float) -> str:
    """
    Formats a duration in seconds into a human-readable H:M:S string.

    Args:
        seconds (float): The total seconds.

    Returns:
        str: The formatted time string (e.g., "0:15:32").
    """
    return str(datetime.timedelta(seconds=int(seconds)))


def temporal_weighted_cross_entropy(
    s1_logits: torch.Tensor,
    s2_logits: torch.Tensor,
    s1_targets: torch.Tensor,
    s2_targets: torch.Tensor,
    forecast_start: int,
    forecast_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute dual-head CE while emphasizing forecast-region transitions."""
    if forecast_weight <= 0:
        raise ValueError("forecast_weight must be positive")
    if s1_logits.shape[:2] != s1_targets.shape or s2_logits.shape[:2] != s2_targets.shape:
        raise ValueError("logits and targets must share batch and sequence dimensions")

    sequence_length = s1_targets.size(1)
    forecast_start = min(max(int(forecast_start), 0), sequence_length)
    weights = torch.ones(sequence_length, device=s1_logits.device, dtype=s1_logits.dtype)
    weights[forecast_start:] = forecast_weight
    weights = weights.unsqueeze(0).expand(s1_targets.size(0), -1)
    denominator = weights.sum().clamp_min(1.0)

    s1_per_token = F.cross_entropy(
        s1_logits.transpose(1, 2), s1_targets, reduction="none"
    )
    s2_per_token = F.cross_entropy(
        s2_logits.transpose(1, 2), s2_targets, reduction="none"
    )
    s1_loss = (s1_per_token * weights).sum() / denominator
    s2_loss = (s2_per_token * weights).sum() / denominator
    return (s1_loss + s2_loss) / 2, s1_loss, s2_loss



def update_early_stopping(
    best_loss: float,
    epochs_without_improvement: int,
    current_loss: float,
    min_delta: float,
    patience: int,
) -> tuple[bool, float, int, bool]:
    """Update early-stop state and report improvement/stop decisions."""
    improved = current_loss < best_loss - min_delta
    if improved:
        best_loss = current_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
    should_stop = patience > 0 and epochs_without_improvement >= patience
    return improved, best_loss, epochs_without_improvement, should_stop
