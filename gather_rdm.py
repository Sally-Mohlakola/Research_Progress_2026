from config.parameters import get_diamond_parameters
from utils.rdm import compute_rdm

import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="Gather RDM for a diamond.")

parser.add_argument("--checkpoint_name", help="Checkpoint name to store outputs", required=True)
parser.add_argument("--diamond_name", help="Name of the diamond preset defined in config/parameters.py",
                     required=True)
parser.add_argument("--batch_size", type=int, help="Samples per batch for RDM collection", default=1024 * 8)
parser.add_argument("--num_batches", type=int, help="Number of batches for RDM collection", default=1024)
parser.add_argument("--theta_bins", type=int, help="Theta resolution of the RDM", default=18)
parser.add_argument("--phi_bins", type=int, help="Phi resolution of the RDM", default=36)
parser.add_argument("--max_depth", type=int, help="Max path bounces before Russian-roulette/truncation",
                     default=64)
parser.add_argument("--sampling_method", type=str, help="Sampling method: uniform, cos_theta, or stratified",
                     default="stratified", choices=["uniform", "cos_theta", "stratified"])
parser.add_argument("--no_aim_jitter", action="store_true",
                     help="Aim every gather ray at the stone's centre instead of sweeping its "
                          "projected disc. Reproduces pre-fix checkpoints (run_09, run_10); "
                          "leaves most (facet, incidence angle) bins unreachable.")

args = parser.parse_args()

output_dir = os.path.join('checkpoints/', args.checkpoint_name)
os.makedirs(output_dir, exist_ok=True)

diamond_kwargs = get_diamond_parameters(args.diamond_name)

rdm_t, rdm_r, rdm_m, count_t, count_r, count_m, x, sa = compute_rdm(
    theta_bins=args.theta_bins,
    phi_bins=args.phi_bins,
    max_depth=args.max_depth,
    num_batches=args.num_batches,
    batch_size=args.batch_size,
    diamond_kwargs=diamond_kwargs,
    sampling_method=args.sampling_method,
    aim_jitter=not args.no_aim_jitter,
)

rdm_t, rdm_r, rdm_m, count_t, count_r, count_m, x, sa = [
    arr.numpy() for arr in [rdm_t, rdm_r, rdm_m, count_t, count_r, count_m, x, sa]
]

print("Saving to", output_dir)

np.savez(
    os.path.join(output_dir, "rdm.npz"),
    rdm_t=rdm_t, rdm_r=rdm_r, rdm_m=rdm_m,
    count_t=count_t, count_r=count_r, count_m=count_m,
    x=x, sa=sa,
    diamond_name=args.diamond_name,
    theta_bins=args.theta_bins,
    phi_bins=args.phi_bins,
    aim_jitter=not args.no_aim_jitter,
)



print("Done")