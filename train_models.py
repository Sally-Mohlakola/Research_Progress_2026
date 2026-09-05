#!/usr/bin/env python3
"""
train_models.py - Train Model_M and Model_T on RDM data for diamond neural rendering.

RDM structure from gather_rdm.py:
    rdm_t: (θi_bins, φi_bins, θo_bins, φo_bins, 3)  # Transmission
    rdm_r: (θi_bins, φi_bins, θo_bins, φo_bins, 3)  # Reflection  
    rdm_m: (θi_bins, φi_bins, θo_bins, φo_bins, 3)  # Multi-scatter
    count_t: (θi_bins, φi_bins)                      # Sample count per incoming dir
    x: (θi_bins, φi_bins, θo_bins, φo_bins, 4)      # Bin centers (θi, φi, θo, φo)
    sa: (θo_bins, φo_bins)                           # Solid angles
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from config import device, variant
import mitsuba as mi
import drjit as dr

# Set variant before anything else
mi.set_variant(variant)

# Import your models
from neural.base_model import Model_M, Model_T
from config.parameters import get_diamond_parameters

# Disable matplotlib to avoid GLIBCXX error
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Train neural models on RDM data")
    parser.add_argument("--checkpoint_name", type=str, required=True,
                       help="Checkpoint name (must match gather_rdm.py)")
    parser.add_argument("--diamond_name", type=str, default='round_diamond_gia',
                       help="Diamond name from config/parameters.py")
    parser.add_argument("--epochs_m", type=int, default=10000,
                       help="Number of epochs for Model_M")
    parser.add_argument("--epochs_t", type=int, default=5000,
                       help="Number of epochs for Model_T")
    parser.add_argument("--lr_m", type=float, default=0.001,
                       help="Learning rate for Model_M")
    parser.add_argument("--lr_t", type=float, default=0.001,
                       help="Learning rate for Model_T")
    parser.add_argument("--batch_size", type=int, default=4096,
                       help="Batch size for training")
    parser.add_argument("--no_plot", action='store_true',
                       help="Skip plotting loss curves")
    parser.add_argument("--drop_zero_bins", action='store_true',
                       help="Train Model_M only on bins that carry energy, discarding "
                            "measured-dark ones. Reproduces pre-fix checkpoints "
                            "(run_09, run_10); the model then never sees a dark example.")
    return parser.parse_args()


def load_rdm_data(checkpoint_dir):
    """Load RDM data from npz file."""
    rdm_path = os.path.join(checkpoint_dir, 'rdm.npz')
    if not os.path.exists(rdm_path):
        raise FileNotFoundError(f"RDM file not found: {rdm_path}")
    
    data = np.load(rdm_path)
    
    # Check what keys are available
    print(f"RDM keys: {list(data.keys())}")
    
    # Extract data (handle different naming conventions)
    rdm_t = data.get('rdm_t')
    rdm_r = data.get('rdm_r')
    rdm_m = data.get('rdm_m')
    count_t = data.get('count_t')
    count_r = data.get('count_r')
    count_m = data.get('count_m')
    x = data.get('x')  # Bin centers
    sa = data.get('sa')  # Solid angles
    
    # If x is not available, compute bin centers from shapes
    if x is None:
        print("  'x' not found in RDM, computing bin centers from shapes...")
        theta_bins = rdm_m.shape[0] * 2  # θi_bins = theta_bins/2
        phi_bins = rdm_m.shape[1]
        
        theta_i_centers = np.linspace(0, np.pi/2, rdm_m.shape[0], endpoint=False) + (np.pi/2 / rdm_m.shape[0]) / 2
        phi_i_centers = np.linspace(-np.pi, np.pi, rdm_m.shape[1], endpoint=False) + (2*np.pi / rdm_m.shape[1]) / 2
        theta_o_centers = np.linspace(0, np.pi, rdm_m.shape[2], endpoint=False) + (np.pi / rdm_m.shape[2]) / 2
        phi_o_centers = np.linspace(-np.pi, np.pi, rdm_m.shape[3], endpoint=False) + (2*np.pi / rdm_m.shape[3]) / 2
        
        theta_i, phi_i, theta_o, phi_o = np.meshgrid(
            theta_i_centers, phi_i_centers, theta_o_centers, phi_o_centers, indexing='ij'
        )
        x = np.stack([theta_i, phi_i, theta_o, phi_o], axis=-1)
    
    # If sa is not available, compute solid angles
    if sa is None:
        print("  'sa' not found in RDM, computing solid angles...")
        theta_edges = np.linspace(0, np.pi, rdm_m.shape[2] + 1)
        phi_edges = np.linspace(-np.pi, np.pi, rdm_m.shape[3] + 1)
        sa = np.zeros((rdm_m.shape[2], rdm_m.shape[3]))
        for i in range(rdm_m.shape[2]):
            for j in range(rdm_m.shape[3]):
                sa[i, j] = (np.cos(theta_edges[i]) - np.cos(theta_edges[i+1])) * (phi_edges[j+1] - phi_edges[j])
    
    # Print shapes
    print(f"  rdm_t shape: {rdm_t.shape if rdm_t is not None else 'None'}")
    print(f"  rdm_r shape: {rdm_r.shape if rdm_r is not None else 'None'}")
    print(f"  rdm_m shape: {rdm_m.shape if rdm_m is not None else 'None'}")
    print(f"  x shape: {x.shape if x is not None else 'None'}")
    print(f"  sa shape: {sa.shape if sa is not None else 'None'}")
    
    return rdm_t, rdm_r, rdm_m, count_t, count_r, count_m, x, sa


def prepare_training_data(rdm_m, x, sa, count_i=None, drop_zero_bins=False):
    """
    Prepare training data for Model_M.

    Input: (θi, φi, θo, φo) → wi (3D), wo (3D) → RGB

    A zero in rdm_m is ambiguous, and the two cases must be treated
    differently:

      * the incoming-direction bin was sampled and no energy left in this
        outgoing direction. That is a *measurement* -- the RDM really is
        dark there -- and it belongs in the training set. Diamond
        appearance is mostly contrast between lit and unlit directions, so
        a model shown only the lit ones has been asked to learn half a
        function and has no way to represent the other half.

      * the incoming-direction bin was never sampled at all. Then
        compute_histogram_4d divided a zero sum by a zero count and mapped
        the resulting NaN to 0, so the array reads 0 without anything
        having been measured. Training on those teaches the network
        darkness that was never observed.

    The array alone cannot tell them apart; `count_i` (samples per incoming
    bin, saved as count_t/count_r/count_m -- they are one array) can, and is
    the only correct mask. If it is missing, every bin is assumed observed.

    drop_zero_bins=True restores the old `rgb.sum() > 1e-6` filter, which
    kept only lit bins, for reproducing pre-fix checkpoints.

    Returns:
        X: (N, 6) - [wi_x, wi_y, wi_z, wo_x, wo_y, wo_z]
        Y: (N, 3) - [R, G, B]
    """
    theta_i_centers = x[:, :, :, :, 0]
    phi_i_centers = x[:, :, :, :, 1]
    theta_o_centers = x[:, :, :, :, 2]
    phi_o_centers = x[:, :, :, :, 3]
    
    if count_i is None:
        print("  [warn] no sample counts in this RDM; assuming every incoming "
              "bin was observed")
        observed = np.ones(rdm_m.shape[:2], dtype=bool)
    else:
        observed = np.asarray(count_i) > 0

    X = []
    Y = []
    n_dark = 0
    n_lit = 0

    for ti in range(rdm_m.shape[0]):
        for pi in range(rdm_m.shape[1]):
            if not observed[ti, pi]:
                continue
            for to in range(rdm_m.shape[2]):
                for po in range(rdm_m.shape[3]):
                    rgb = rdm_m[ti, pi, to, po]
                    if rgb.sum() <= 1e-6:
                        n_dark += 1
                        if drop_zero_bins:
                            continue
                    else:
                        n_lit += 1

                    # Convert angles to direction vectors
                    theta_i = theta_i_centers[ti, pi, to, po]
                    phi_i = phi_i_centers[ti, pi, to, po]
                    theta_o = theta_o_centers[ti, pi, to, po]
                    phi_o = phi_o_centers[ti, pi, to, po]

                    wi = np.array([
                        np.sin(theta_i) * np.cos(phi_i),
                        np.sin(theta_i) * np.sin(phi_i),
                        np.cos(theta_i)
                    ], dtype=np.float32)

                    wo = np.array([
                        np.sin(theta_o) * np.cos(phi_o),
                        np.sin(theta_o) * np.sin(phi_o),
                        np.cos(theta_o)
                    ], dtype=np.float32)

                    X.append(np.concatenate([wi, wo]))
                    Y.append(rgb)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    total_bins = int(np.prod(rdm_m.shape[:4]))
    unobserved = total_bins - int(observed.sum()) * int(np.prod(rdm_m.shape[2:4]))
    print(f"  Training data: {len(X)} samples of {total_bins} bins "
          f"({100.0 * len(X) / total_bins:.1f}%)")
    print(f"    lit          : {n_lit}")
    print(f"    measured dark: {n_dark}"
          + ("  (excluded: --drop_zero_bins)" if drop_zero_bins else ""))
    print(f"    unobserved   : {unobserved}  (excluded; no samples in that "
          f"incoming bin)")
    print(f"  X shape: {X.shape}, Y shape: {Y.shape}")
    print(f"  Y range: [{Y.min():.4f}, {Y.max():.4f}]")

    return X, Y


def prepare_transmittance_data(rdm_t, rdm_r, rdm_m, x):
    """
    Prepare training data for Model_T.
    
    Input: (θi, φi) → wi (3D) → transmittance (scalar)
    """
    theta_i_centers = x[:, :, 0, 0, 0]
    phi_i_centers = x[:, :, 0, 0, 1]
    
    X = []
    Y = []
    
    for ti in range(rdm_t.shape[0]):
        for pi in range(rdm_t.shape[1]):
            # Sum over outgoing directions for each component
            total_t = rdm_t[ti, pi, :, :].sum()
            total_r = rdm_r[ti, pi, :, :].sum()
            total_m = rdm_m[ti, pi, :, :].sum()
            total = total_t + total_r + total_m

            # Unlike Model_M's mask, this guard is not the 2.2 defect: the
            # target here is a *ratio* total_t/total, which is genuinely
            # undefined when nothing left the stone in this incoming bin
            # (never sampled, or every path trapped by TIR). There is no
            # "measured dark" case to rescue -- do not relax it to match.
            if total > 1e-6:
                # Transmittance = total_t / total
                transmittance = total_t / total
                
                theta_i = theta_i_centers[ti, pi]
                phi_i = phi_i_centers[ti, pi]
                
                wi = np.array([
                    np.sin(theta_i) * np.cos(phi_i),
                    np.sin(theta_i) * np.sin(phi_i),
                    np.cos(theta_i)
                ], dtype=np.float32)
                
                X.append(wi)
                Y.append([transmittance])
    
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    
    print(f"  Transmittance data: {len(X)} samples")
    print(f"  X shape: {X.shape}, Y shape: {Y.shape}")
    print(f"  Y range: [{Y.min():.4f}, {Y.max():.4f}]")
    
    return X, Y


def train_model(model, X, Y, epochs, lr, batch_size, model_name, checkpoint_dir):
    """Generic training function."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    print(f"  Samples: {len(X)}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Batch size: {batch_size}")
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Convert to tensors
    X_tensor = torch.tensor(X, device=device)
    Y_tensor = torch.tensor(Y, device=device)
    
    history = []
    
    for epoch in range(epochs):
        # Shuffle data
        perm = torch.randperm(len(X_tensor))
        
        epoch_loss = 0.0
        num_batches = 0
        
        for i in range(0, len(X_tensor), batch_size):
            batch_idx = perm[i:i+batch_size]
            x_batch = X_tensor[batch_idx]
            y_batch = Y_tensor[batch_idx]
            
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        history.append(avg_loss)
        
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.7f}")
    
    # Save model
    model_path = os.path.join(checkpoint_dir, f'{model_name}.pth')
    torch.save(model.state_dict(), model_path)
    print(f"  ✓ Saved {model_name} to {model_path}")
    
    return history


def plot_loss(history_m, history_t, output_dir):
    """Plot loss curves."""
    plt.figure(figsize=(12, 5))
    
    if history_m:
        plt.subplot(1, 2, 1)
        plt.plot(history_m)
        plt.xlabel('Epoch (x100)')
        plt.ylabel('Loss')
        plt.title('Model_M Loss')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
    
    if history_t:
        plt.subplot(1, 2, 2)
        plt.plot(history_t)
        plt.xlabel('Epoch (x100)')
        plt.ylabel('Loss')
        plt.title('Model_T Loss')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_loss.png'), dpi=150)
    print(f"  ✓ Saved loss plot to {output_dir}/training_loss.png")


def main():
    args = parse_args()
    
    checkpoint_dir = os.path.join('checkpoints', args.checkpoint_name)
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    print(f"✓ Training on RDM from: {checkpoint_dir}")
    print(f"✓ Device: {device}")
    
    # Load RDM data
    print("\nLoading RDM data...")
    rdm_t, rdm_r, rdm_m, count_t, count_r, count_m, x, sa = load_rdm_data(checkpoint_dir)
    
    if rdm_m is None:
        raise ValueError("rdm_m not found in RDM data!")
    
    # Prepare training data for Model_M
    print("\nPreparing Model_M training data...")
    X_m, Y_m = prepare_training_data(rdm_m, x, sa, count_m, args.drop_zero_bins)

    if len(X_m) == 0:
        raise ValueError("No usable bins found in rdm_m!")
    
    # Prepare training data for Model_T
    print("\nPreparing Model_T training data...")
    if rdm_t is not None and rdm_r is not None:
        X_t, Y_t = prepare_transmittance_data(rdm_t, rdm_r, rdm_m, x)
    else:
        print("  ⚠️ rdm_t or rdm_r missing, skipping Model_T")
        X_t, Y_t = None, None
    
    # Train Model_M
    model_m = Model_M()
    history_m = train_model(
        model_m, X_m, Y_m,
        epochs=args.epochs_m,
        lr=args.lr_m,
        batch_size=args.batch_size,
        model_name='model_m',
        checkpoint_dir=checkpoint_dir
    )
    
    # Train Model_T (if data available)
    history_t = []
    if X_t is not None and len(X_t) > 0:
        model_t = Model_T()
        history_t = train_model(
            model_t, X_t, Y_t,
            epochs=args.epochs_t,
            lr=args.lr_t,
            batch_size=args.batch_size,
            model_name='model_t',
            checkpoint_dir=checkpoint_dir
        )
    else:
        print("  ⚠️ Skipping Model_T training (no transmittance data)")
    
    # Plot loss curves
    if not args.no_plot:
        print("\nPlotting loss curves...")
        plot_loss(history_m, history_t, checkpoint_dir)
    
    print(f"\n{'='*60}")
    print("✓ TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Model_M: {len(X_m)} samples, loss: {history_m[-1]:.7f}")
    if history_t:
        print(f"  Model_T: {len(X_t)} samples, loss: {history_t[-1]:.7f}")
    print(f"  Models saved to: {checkpoint_dir}")
    print(f"  Loss plot: {checkpoint_dir}/training_loss.png")


if __name__ == "__main__":
    main()