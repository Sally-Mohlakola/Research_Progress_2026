#!/usr/bin/env python3
"""
Regenerate RDM with stratified sampling for improved coverage,
especially at grazing angles.

This script uses the improved stratified sampling method to create
a new RDM with much better fill rate and consistent coverage across
all angles.
"""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.parameters import get_diamond_parameters
from utils.rdm import compute_rdm
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description="Regenerate RDM with stratified sampling for better grazing angle coverage"
    )
    
    parser.add_argument(
        "--checkpoint_name",
        default="my_diamond_run_stratified",
        help="Checkpoint name for the improved RDM"
    )
    
    parser.add_argument(
        "--diamond_name",
        default="default",
        help="Diamond preset name from config/parameters.py"
    )
    
    parser.add_argument(
        "--theta_bins",
        type=int,
        default=18,
        help="Theta resolution (default: 18 for faster collection)"
    )
    
    parser.add_argument(
        "--phi_bins",
        type=int,
        default=36,
        help="Phi resolution (default: 36 for faster collection)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8192,
        help="Samples per batch (default: 8192)"
    )
    
    parser.add_argument(
        "--num_batches",
        type=int,
        default=2048,
        help="Number of batches (default: 2048 for ~16M samples)"
    )
    
    parser.add_argument(
        "--max_depth",
        type=int,
        default=64,
        help="Max path depth (default: 64)"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("RDM REGENERATION WITH STRATIFIED SAMPLING")
    print("="*70)
    print(f"Checkpoint:     {args.checkpoint_name}")
    print(f"Diamond:        {args.diamond_name}")
    print(f"Sampling:       stratified (improved)")
    print(f"Resolution:     θ={args.theta_bins}, φ={args.phi_bins}")
    print(f"Total samples:  {args.batch_size * args.num_batches:,}")
    print("="*70)
    print()
    
    # Create output directory
    output_dir = os.path.join('checkpoints', args.checkpoint_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get diamond parameters
    diamond_kwargs = get_diamond_parameters(args.diamond_name)
    
    print("Starting RDM collection with stratified sampling...")
    print("This will provide much better coverage of grazing angles.")
    print()
    
    # Compute RDM with stratified sampling
    rdm_t, rdm_r, rdm_m, count_t, count_r, count_m, x, sa = compute_rdm(
        theta_bins=args.theta_bins,
        phi_bins=args.phi_bins,
        max_depth=args.max_depth,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        diamond_kwargs=diamond_kwargs,
        sampling_method='stratified',  # Use improved stratified sampling
    )
    
    # Convert to numpy
    rdm_t, rdm_r, rdm_m, count_t, count_r, count_m, x, sa = [
        arr.numpy() for arr in [rdm_t, rdm_r, rdm_m, count_t, count_r, count_m, x, sa]
    ]
    
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    
    # Analyze fill rates
    rdm_m_fill = (rdm_m.sum(axis=-1) > 0).sum() / (rdm_m.size / 3) * 100
    rdm_t_fill = (rdm_t.sum(axis=-1) > 0).sum() / (rdm_t.size / 3) * 100
    rdm_r_fill = (rdm_r.sum(axis=-1) > 0).sum() / (rdm_r.size / 3) * 100
    
    print(f"RDM_M fill rate: {rdm_m_fill:.2f}%")
    print(f"RDM_T fill rate: {rdm_t_fill:.2f}%")
    print(f"RDM_R fill rate: {rdm_r_fill:.2f}%")
    print()
    
    # Analyze grazing angles
    theta_i_bins = rdm_m.shape[0]
    print("Grazing angle coverage (last 3 θ_i bins):")
    for i in range(max(0, theta_i_bins-3), theta_i_bins):
        theta_i_deg = i * (90/theta_i_bins) + (90/(2*theta_i_bins))
        slice_data = rdm_m[i, :, :, :, :]
        non_zero = (slice_data.sum(axis=-1) > 0).sum()
        total = slice_data.size // 3
        fill = (non_zero / total) * 100
        print(f"  θ_i={theta_i_deg:.1f}°: {fill:.2f}% fill ({non_zero}/{total} bins)")
    
    print()
    print("="*70)
    
    # Save results
    save_path = os.path.join(output_dir, "rdm.npz")
    np.savez(
        save_path,
        rdm_t=rdm_t, rdm_r=rdm_r, rdm_m=rdm_m,
        count_t=count_t, count_r=count_r, count_m=count_m,
        x=x, sa=sa,
        diamond_name=args.diamond_name,
        theta_bins=args.theta_bins,
        phi_bins=args.phi_bins,
        sampling_method='stratified',
    )
    
    print(f"✓ Saved RDM to: {save_path}")
    print()
    print("NEXT STEPS:")
    print("1. Retrain neural models with the improved RDM:")
    print(f"   python train_models.py --checkpoint_name {args.checkpoint_name}")
    print()
    print("2. Re-render with the new models:")
    print(f"   python eval.py --checkpoint_dir checkpoints/{args.checkpoint_name} \\")
    print(f"      --output_dir renders/{args.checkpoint_name}")
    print()
    print("="*70)

if __name__ == "__main__":
    main()
