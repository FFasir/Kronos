import os
import sys
import json
import time
import random
import numpy as np
from time import gmtime, strftime
import torch.distributed as dist
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import comet_ml

# Ensure project root is in path
sys.path.append('../')
from config import Config
from dataset import QlibDataset
from model.kronos import KronosTokenizer, Kronos
# Import shared utilities
from utils.training_utils import (
    setup_ddp,
    cleanup_ddp,
    set_seed,
    get_model_size,
    format_time,
    update_early_stopping,
    temporal_weighted_cross_entropy,
)


def create_dataloaders(config: dict, rank: int, world_size: int):
    """
    Creates and returns distributed dataloaders for training and validation.

    Args:
        config (dict): A dictionary of configuration parameters.
        rank (int): The global rank of the current process.
        world_size (int): The total number of processes.

    Returns:
        tuple: (train_loader, val_loader, train_dataset, valid_dataset).
    """
    print(f"[Rank {rank}] Creating distributed dataloaders...")
    train_dataset = QlibDataset('train')
    valid_dataset = QlibDataset('val')
    print(f"[Rank {rank}] Train dataset size: {len(train_dataset)}, Validation dataset size: {len(valid_dataset)}")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], sampler=train_sampler,
        num_workers=config.get('num_workers', 2), pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        valid_dataset, batch_size=config['batch_size'], sampler=val_sampler,
        num_workers=config.get('num_workers', 2), pin_memory=True, drop_last=False
    )
    return train_loader, val_loader, train_dataset, valid_dataset


def train_model(model, tokenizer, device, config, save_dir, logger, rank, world_size):
    """
    The main training and validation loop for the predictor.
    """
    start_time = time.time()
    if rank == 0:
        effective_bs = config['batch_size'] * world_size
        print(f"Effective BATCHSIZE per GPU: {config['batch_size']}, Total: {effective_bs}")

    train_loader, val_loader, train_dataset, valid_dataset = create_dataloaders(config, rank, world_size)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['predictor_learning_rate'],
        betas=(config['adam_beta1'], config['adam_beta2']),
        weight_decay=config['adam_weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config['predictor_learning_rate'],
        steps_per_epoch=len(train_loader), epochs=config['epochs'],
        pct_start=0.03, div_factor=10
    )

    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    val_loss_history = []
    unweighted_val_loss_history = []
    completed_epochs = 0
    dt_result = {}
    batch_idx_global = 0
    start_epoch = 0
    training_state_path = os.path.join(save_dir, 'checkpoints', 'training_state.pt')
    if config.get('resume_training') and os.path.isfile(training_state_path):
        state = torch.load(training_state_path, map_location=device, weights_only=False)
        model.module.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        start_epoch = int(state['completed_epochs'])
        completed_epochs = start_epoch
        best_val_loss = float(state['best_val_loss'])
        best_epoch = int(state['best_epoch'])
        epochs_without_improvement = int(state['epochs_without_improvement'])
        val_loss_history = list(state['val_loss_history'])
        unweighted_val_loss_history = list(state.get('unweighted_val_loss_history', []))
        random.setstate(state['python_rng_state'])
        np.random.set_state(state['numpy_rng_state'])
        torch.set_rng_state(state['torch_rng_state'].cpu())
        if torch.cuda.is_available() and state.get('cuda_rng_states'):
            torch.cuda.set_rng_state_all([item.cpu() for item in state['cuda_rng_states']])
        batch_idx_global = start_epoch * len(train_loader)
        if rank == 0:
            print(f"Resuming predictor training from epoch {start_epoch + 1}")

    for epoch_idx in range(start_epoch, config['epochs']):
        epoch_start_time = time.time()
        model.train()
        train_loader.sampler.set_epoch(epoch_idx)

        train_dataset.set_epoch_seed(epoch_idx)
        valid_dataset.set_epoch_seed(0)

        for i, (batch_x, batch_x_stamp) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

            # Tokenize input data on-the-fly
            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

            # Prepare inputs and targets for the language model
            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

            # Forward pass and loss calculation
            logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
            loss, s1_loss, s2_loss = temporal_weighted_cross_entropy(
                logits[0], logits[1], token_out[0], token_out[1],
                forecast_start=config['lookback_window'] - 1,
                forecast_weight=config['forecast_loss_weight'],
            )

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            scheduler.step()

            # Logging (Master Process Only)
            if rank == 0 and (batch_idx_global + 1) % config['log_interval'] == 0:
                lr = optimizer.param_groups[0]['lr']
                print(
                    f"[Rank {rank}, Epoch {epoch_idx + 1}/{config['epochs']}, Step {i + 1}/{len(train_loader)}] "
                    f"LR {lr:.6f}, Loss: {loss.item():.4f}"
                )
            if rank == 0 and logger:
                lr = optimizer.param_groups[0]['lr']
                logger.log_metric('train_predictor_loss_batch', loss.item(), step=batch_idx_global)
                logger.log_metric('train_S1_loss_each_batch', s1_loss.item(), step=batch_idx_global)
                logger.log_metric('train_S2_loss_each_batch', s2_loss.item(), step=batch_idx_global)
                logger.log_metric('predictor_learning_rate', lr, step=batch_idx_global)

            batch_idx_global += 1

        # --- Validation Loop ---
        model.eval()
        tot_val_loss_sum_rank = 0.0
        tot_unweighted_val_loss_sum_rank = 0.0
        val_samples_processed_rank = 0
        with torch.no_grad():
            for batch_x, batch_x_stamp in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

                logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                val_loss, _, _ = temporal_weighted_cross_entropy(
                    logits[0], logits[1], token_out[0], token_out[1],
                    forecast_start=config['lookback_window'] - 1,
                    forecast_weight=config['forecast_loss_weight'],
                )
                unweighted_val_loss, _, _ = temporal_weighted_cross_entropy(
                    logits[0], logits[1], token_out[0], token_out[1],
                    forecast_start=config['lookback_window'] - 1,
                    forecast_weight=1.0,
                )

                tot_val_loss_sum_rank += val_loss.item() * batch_x.size(0)
                tot_unweighted_val_loss_sum_rank += unweighted_val_loss.item() * batch_x.size(0)
                val_samples_processed_rank += batch_x.size(0)

        # Reduce validation metrics
        val_loss_sum_tensor = torch.tensor(tot_val_loss_sum_rank, device=device)
        unweighted_val_loss_sum_tensor = torch.tensor(
            tot_unweighted_val_loss_sum_rank, device=device
        )
        val_batches_tensor = torch.tensor(val_samples_processed_rank, device=device)
        dist.all_reduce(val_loss_sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(unweighted_val_loss_sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_batches_tensor, op=dist.ReduceOp.SUM)

        avg_val_loss = val_loss_sum_tensor.item() / val_batches_tensor.item() if val_batches_tensor.item() > 0 else 0
        avg_unweighted_val_loss = (
            unweighted_val_loss_sum_tensor.item() / val_batches_tensor.item()
            if val_batches_tensor.item() > 0 else 0
        )

        # --- End of Epoch Summary & Checkpointing (Master Process Only) ---
        stop_requested = False
        if rank == 0:
            completed_epochs = epoch_idx + 1
            val_loss_history.append(avg_val_loss)
            unweighted_val_loss_history.append(avg_unweighted_val_loss)
            print(f"\n--- Epoch {epoch_idx + 1}/{config['epochs']} Summary ---")
            print(f"Validation Loss (weighted): {avg_val_loss:.4f}")
            print(f"Validation Loss (unweighted): {avg_unweighted_val_loss:.4f}")
            print(f"Time This Epoch: {format_time(time.time() - epoch_start_time)}")
            print(f"Total Time Elapsed: {format_time(time.time() - start_time)}\n")
            if logger:
                logger.log_metric('val_predictor_loss_epoch', avg_val_loss, epoch=epoch_idx)

            improved, best_val_loss, epochs_without_improvement, stop_requested = update_early_stopping(
                best_val_loss,
                epochs_without_improvement,
                avg_val_loss,
                config['early_stopping_min_delta'],
                config['early_stopping_patience'],
            )
            if improved:
                best_epoch = epoch_idx + 1
                save_path = f"{save_dir}/checkpoints/best_model"
                model.module.save_pretrained(save_path)
                print(f"Best model saved to {save_path} (Val Loss: {best_val_loss:.4f})")

            patience = config['early_stopping_patience']
            if stop_requested:
                print(
                    f"Early stopping after epoch {epoch_idx + 1}: no validation "
                    f"improvement for {patience} epochs."
                )

            if config.get('resume_training'):
                torch.save({
                    'completed_epochs': completed_epochs,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_epoch': best_epoch,
                    'epochs_without_improvement': epochs_without_improvement,
                    'val_loss_history': val_loss_history,
                    'unweighted_val_loss_history': unweighted_val_loss_history,
                    'python_rng_state': random.getstate(),
                    'numpy_rng_state': np.random.get_state(),
                    'torch_rng_state': torch.get_rng_state(),
                    'cuda_rng_states': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                }, training_state_path)

        stop_tensor = torch.tensor(int(stop_requested), device=device)
        dist.broadcast(stop_tensor, src=0)
        if stop_tensor.item():
            break

    dt_result.update({
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'epochs_completed': completed_epochs,
        'stopped_early': completed_epochs < config['epochs'],
        'val_loss_history': val_loss_history,
        'unweighted_val_loss_history': unweighted_val_loss_history,
        'available_train_windows': len(train_dataset.indices),
        'available_val_windows': len(valid_dataset.indices),
        'train_samples_per_epoch': len(train_dataset),
        'val_samples_per_epoch': len(valid_dataset),
        'sampling_strategy': (
            'deterministic_balanced_direction_without_replacement'
            if config['balance_train_directions']
            else 'deterministic_without_replacement'
        ),
        'available_up_windows': len(train_dataset.up_indices),
        'available_down_windows': len(train_dataset.down_indices),
    })
    return dt_result


def main(config: dict):
    """Main function to orchestrate the DDP training process."""
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    set_seed(config['seed'], rank)

    save_dir = os.path.join(config['save_path'], config['predictor_save_folder_name'])

    # Logger and summary setup (master process only)
    comet_logger, master_summary = None, {}
    if rank == 0:
        os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
        master_summary = {
            'start_time': strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
            'save_directory': save_dir,
            'world_size': world_size,
            'experiment_id': config['experiment_id'],
            'training_config': {
                key: str(config[key])
                for key in (
                    'dataset_path', 'pretrained_predictor_path',
                    'finetuned_tokenizer_path', 'lookback_window',
                    'predict_window', 'epochs', 'batch_size', 'n_train_iter',
                    'n_val_iter', 'num_workers', 'predictor_learning_rate',
                    'early_stopping_patience', 'early_stopping_min_delta', 'seed',
                    'forecast_loss_weight', 'balance_train_directions'
                )
            },
        }
        if config['use_comet']:
            comet_logger = comet_ml.Experiment(
                api_key=config['comet_config']['api_key'],
                project_name=config['comet_config']['project_name'],
                workspace=config['comet_config']['workspace'],
            )
            comet_logger.add_tag(config['comet_tag'])
            comet_logger.set_name(config['comet_name'])
            comet_logger.log_parameters(config)
            print("Comet Logger Initialized.")

    dist.barrier()

    # Model Initialization
    tokenizer = KronosTokenizer.from_pretrained(config['finetuned_tokenizer_path'])
    tokenizer.eval().to(device)

    model = Kronos.from_pretrained(config['pretrained_predictor_path'])
    model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    if rank == 0:
        print(f"Predictor Model Size: {get_model_size(model.module)}")

    # Start Training
    dt_result = train_model(
        model, tokenizer, device, config, save_dir, comet_logger, rank, world_size
    )

    if rank == 0:
        master_summary['final_result'] = dt_result
        with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
            json.dump(master_summary, f, indent=4)
        print('Training finished. Summary file saved.')
        if comet_logger: comet_logger.end()

    cleanup_ddp()


if __name__ == '__main__':
    # Usage: torchrun --standalone --nproc_per_node=NUM_GPUS train_predictor.py
    if "WORLD_SIZE" not in os.environ:
        raise RuntimeError("This script must be launched with `torchrun`.")

    config_instance = Config()
    main(config_instance.__dict__)
