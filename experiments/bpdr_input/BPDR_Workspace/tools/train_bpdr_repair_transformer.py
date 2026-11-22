"""Training script for BPDR Pitch-Repair Transformer (3 matched arms: frame_only, augmentation_only, bpdr).

Evaluates the declared research hypothesis:
Disentangled boundary conditioning enables continuous pitch error correction
without step glitches or degradation under false boundary splits, merges, or jitter.

Run: ..\\.venv\\Scripts\\python.exe tools\\train_bpdr_repair_transformer.py
"""
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tnf

WS = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(WS / 'source'))
from bpdr_input.studio.repair_transformer import BPDRRepairTransformer

DATA_DIR = WS / 'results' / 'studio' / 'r1' / 'paired_evaluator_dataset'
OUT_DIR = WS / 'results' / 'studio' / 'r1'
CHECKPOINT_PATH = OUT_DIR / 'bpdr_repair_transformer.pt'
BENCHMARK_PATH = OUT_DIR / 'bpdr_repair_benchmark.json'

SEEDS = (339, 340, 341)
ARMS = ('frame_only', 'augmentation_only', 'bpdr')
EPOCHS = 18
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4


def load_dataset_split(name='train'):
    npz_path = DATA_DIR / f"{name}_data.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset split not found: {npz_path}")
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}


def build_feature_tensor(split_data, arm='bpdr', corrupt_boundary=False, rng=None):
    """Construct 8-channel input tensor from split arrays.
    
    Channels:
    0: sung_pitch (cents / 100)
    1: ref_pitch (cents / 100)
    2: pitch_diff (cents / 100)
    3: ok_mask (0 or 1)
    4: confidence (0 to 1)
    5: energy (normalized dB / 100)
    6: onset (0 to 1)
    7: prov_boundary (0 to 1)
    """
    sung = split_data['sung_pitch'] / 100.0
    ref = split_data['ref_centered'] / 100.0
    diff = sung - ref
    ok = split_data['ok_mask']
    conf = split_data['conf']
    energy = np.clip(split_data['energy'] / 100.0, -1.0, 1.0)
    onset = split_data['onset']
    boundary = split_data['prov_boundary'].copy()

    if corrupt_boundary and rng is not None:
        # Corrupt boundary by adding noise, dropping peaks, or jitter
        B_, T_ = boundary.shape
        for i in range(B_):
            mode = rng.choice(['split', 'merge', 'jitter'])
            if mode == 'split':
                # Double the boundary peaks
                noise = rng.uniform(0.0, 0.6, size=T_) * (rng.random(T_) > 0.85)
                boundary[i] = np.clip(boundary[i] + noise, 0.0, 1.0)
            elif mode == 'merge':
                # Suppress 50% of boundary peaks
                drop_mask = (rng.random(T_) > 0.5)
                boundary[i] = boundary[i] * drop_mask
            elif mode == 'jitter':
                # Roll boundary randomly by +- 1..3 frames
                shift = int(rng.integers(-3, 4))
                boundary[i] = np.roll(boundary[i], shift)

    if arm == 'frame_only':
        # Zero out boundary channel
        boundary = np.zeros_like(boundary)

    feats = np.stack([sung, ref, diff, ok, conf, energy, onset, boundary], axis=-1)
    targets = split_data['target_corr']
    masks = split_data['ok_mask']
    return feats.astype(np.float32), targets.astype(np.float32), masks.astype(np.float32)


def evaluate_model(model, val_feats, val_targets, val_masks, corrupt=False, rng=None):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(val_feats, dtype=torch.float32)
        target = torch.tensor(val_targets, dtype=torch.float32)
        mask = torch.tensor(val_masks, dtype=torch.float32)

        pred = model(x)
        err = (pred - target).abs() * mask
        mae_cents = float(err.sum() / mask.sum().clamp_min(1.0))

        # Measure smoothness: 1st difference of predictions on voiced segments
        diff1 = (pred[:, 1:] - pred[:, :-1]).abs() * mask[:, 1:] * mask[:, :-1]
        smoothness_step = float(diff1.mean())
        
        # Identity / vibrato preservation error: on samples where target == 0
        clean_mask = (target.abs() < 1e-4) & (mask > 0.5)
        if clean_mask.sum() > 0:
            vibrato_preservation_mae = float(((pred).abs() * clean_mask).sum() / clean_mask.sum())
        else:
            vibrato_preservation_mae = 0.0

    return {
        'mae_cents': round(mae_cents, 2),
        'smoothness_step_cents': round(smoothness_step, 2),
        'vibrato_preservation_mae': round(vibrato_preservation_mae, 2),
    }


def train_single_arm(arm='bpdr', seed=339, train_data=None, dev_data=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = BPDRRepairTransformer(nin=8, d=64, nhead=4, nlayers=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    huber = nn.SmoothL1Loss(reduction='none')

    train_x, train_y, train_m = build_feature_tensor(train_data, arm=arm, corrupt_boundary=False)
    dev_x_clean, dev_y, dev_m = build_feature_tensor(dev_data, arm=arm, corrupt_boundary=False)
    dev_x_corrupt, _, _ = build_feature_tensor(dev_data, arm=arm, corrupt_boundary=True, rng=np.random.default_rng(42))

    N_train = len(train_x)
    indices = np.arange(N_train)
    
    best_dev_mae = float('inf')
    best_weights = None

    print(f"\n--- Training Arm: [{arm.upper()}] (Seed {seed}) ---")
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        rng.shuffle(indices)
        total_loss = 0.0
        n_batches = 0

        for b_start in range(0, N_train, BATCH_SIZE):
            b_idx = indices[b_start:b_start + BATCH_SIZE]
            x_nom = torch.tensor(train_x[b_idx], dtype=torch.float32)
            y = torch.tensor(train_y[b_idx], dtype=torch.float32)
            m = torch.tensor(train_m[b_idx], dtype=torch.float32)

            pred_nom = model(x_nom)
            loss_nom = (huber(pred_nom, y) * m).sum() / m.sum().clamp_min(1.0)
            loss = loss_nom

            # Boundary augmentation / corruption
            if arm in ('augmentation_only', 'bpdr'):
                x_aug = x_nom.clone()
                # Apply random boundary split/merge/jitter to channel 7
                aug_type = rng.choice(['drop', 'noise', 'jitter'])
                if aug_type == 'noise':
                    x_aug[:, :, 7] = torch.clamp(x_aug[:, :, 7] + torch.rand_like(x_aug[:, :, 7]) * 0.4, 0.0, 1.0)
                elif aug_type == 'drop':
                    x_aug[:, :, 7] = x_aug[:, :, 7] * (torch.rand_like(x_aug[:, :, 7]) > 0.5).float()
                elif aug_type == 'jitter':
                    x_aug[:, :, 7] = torch.roll(x_aug[:, :, 7], shifts=int(rng.integers(-2, 3)), dims=1)

                pred_aug = model(x_aug)
                loss_aug = (huber(pred_aug, y) * m).sum() / m.sum().clamp_min(1.0)
                loss = 0.5 * loss + 0.5 * loss_aug

                # BPDR Disentangled Loss: invariance penalty between nominal and corrupted boundary predictions
                if arm == 'bpdr':
                    disentangle_loss = ((pred_nom - pred_aug).abs() * m).sum() / m.sum().clamp_min(1.0)
                    # Smoothness penalty
                    smooth_loss = ((pred_nom[:, 1:] - pred_nom[:, :-1]).abs() * m[:, 1:] * m[:, :-1]).mean()
                    loss = loss + 0.5 * disentangle_loss + 0.02 * smooth_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        scheduler.step()

        # Dev evaluation
        dev_res_clean = evaluate_model(model, dev_x_clean, dev_y, dev_m)
        dev_res_corrupt = evaluate_model(model, dev_x_corrupt, dev_y, dev_m)
        
        if dev_res_clean['mae_cents'] < best_dev_mae:
            best_dev_mae = dev_res_clean['mae_cents']
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 6 == 0 or epoch == EPOCHS:
            print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {total_loss/n_batches:.3f} | Dev Clean MAE: {dev_res_clean['mae_cents']:.2f}c | Dev Corrupt MAE: {dev_res_corrupt['mae_cents']:.2f}c | Vib Pres: {dev_res_clean['vibrato_preservation_mae']:.2f}c")

    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.1f}s. Best Dev MAE: {best_dev_mae:.2f} cents.")

    # Load best weights
    model.load_state_dict(best_weights)
    final_clean = evaluate_model(model, dev_x_clean, dev_y, dev_m)
    final_corrupt = evaluate_model(model, dev_x_corrupt, dev_y, dev_m)

    return model, {
        'arm': arm,
        'seed': seed,
        'best_dev_mae_clean': final_clean['mae_cents'],
        'dev_mae_corrupt': final_corrupt['mae_cents'],
        'corruption_drop_cents': round(final_corrupt['mae_cents'] - final_clean['mae_cents'], 2),
        'smoothness_step_cents': final_clean['smoothness_step_cents'],
        'vibrato_preservation_mae': final_clean['vibrato_preservation_mae'],
        'elapsed_s': round(elapsed, 1),
    }


def run_training_experiment():
    print("Loading paired dataset splits...")
    train_data = load_dataset_split('train')
    dev_data = load_dataset_split('dev')
    print(f"Train samples: {len(train_data['sung_pitch'])}, Dev samples: {len(dev_data['sung_pitch'])}")

    results = []
    best_bpdr_model = None
    best_bpdr_mae = float('inf')

    for arm in ARMS:
        for seed in SEEDS:
            model, res = train_single_arm(arm=arm, seed=seed, train_data=train_data, dev_data=dev_data)
            results.append(res)
            if arm == 'bpdr' and res['best_dev_mae_clean'] < best_bpdr_mae:
                best_bpdr_mae = res['best_dev_mae_clean']
                best_bpdr_model = model

    # Save best BPDR model checkpoint
    if best_bpdr_model is not None:
        torch.save(best_bpdr_model.state_dict(), CHECKPOINT_PATH)
        print(f"\nSaved Best BPDR Model Checkpoint to: {CHECKPOINT_PATH} (Dev MAE: {best_bpdr_mae:.2f} cents)")

    # Aggregate 3-arm benchmark summary
    summary_by_arm = {}
    for arm in ARMS:
        arm_res = [r for r in results if r['arm'] == arm]
        summary_by_arm[arm] = {
            'mean_clean_mae': round(float(np.mean([r['best_dev_mae_clean'] for r in arm_res])), 2),
            'mean_corrupt_mae': round(float(np.mean([r['dev_mae_corrupt'] for r in arm_res])), 2),
            'mean_corruption_drop': round(float(np.mean([r['corruption_drop_cents'] for r in arm_res])), 2),
            'mean_smoothness_step': round(float(np.mean([r['smoothness_step_cents'] for r in arm_res])), 2),
            'mean_vibrato_preservation_mae': round(float(np.mean([r['vibrato_preservation_mae'] for r in arm_res])), 2),
        }

    benchmark = {
        'status': 'SUCCESS',
        'checkpoint': str(CHECKPOINT_PATH),
        'seeds': list(SEEDS),
        'summary_by_arm': summary_by_arm,
        'detailed_runs': results,
    }

    with open(BENCHMARK_PATH, 'w', encoding='utf-8') as f:
        json.dump(benchmark, f, indent=2)

    print("\n================== 3-ARM EXPERIMENTAL BENCHMARK SUMMARY ==================")
    print(f"{'Arm':<20}{'Clean MAE':<14}{'Corrupt MAE':<14}{'Drop':<10}{'Step Cents':<12}{'Vib Pres':<10}")
    print("-" * 78)
    for arm, s in summary_by_arm.items():
        print(f"{arm:<20}{s['mean_clean_mae']:<14.2f}{s['mean_corrupt_mae']:<14.2f}{s['mean_corruption_drop']:<10.2f}{s['mean_smoothness_step']:<12.2f}{s['mean_vibrato_preservation_mae']:<10.2f}")
    print("-" * 78)
    print(f"Benchmark results written to {BENCHMARK_PATH}")
    return benchmark


if __name__ == '__main__':
    run_training_experiment()
