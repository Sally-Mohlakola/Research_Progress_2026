"""
export_rdm_json.py - convert a saved rdm.npz into a compact JSON file for
the interactive HTML viewer (view_rdm.html).

The viewer needs, for each (theta_i, phi_i) incoming-direction bin, the
full (theta_o, phi_o) outgoing distribution for transmittance/reflectance/
multi-scatter. We collapse RGB -> a single scalar (mean of channels) here,
since the interactive view shows energy density, not color, and for this
non-spectral dielectric all three channels carry the same value anyway.

Usage:
    python export_rdm_json.py --checkpoint_name test_run_v2
"""

import argparse
import json
import os

import numpy as np

def export(checkpoint_name):
    npz_path = os.path.join('checkpoints',checkpoint_name, "rdm.npz")
    data = np.load(npz_path, allow_pickle=True)

    rdm_t, rdm_r, rdm_m = data["rdm_t"], data["rdm_r"], data["rdm_m"]
    x = data["x"]  # (theta_i, phi_i, theta_o, phi_o, 4): [theta_i, phi_i, theta_o, phi_o]

    theta_i_bins, phi_i_bins, theta_o_bins, phi_o_bins = rdm_t.shape[:4]

    theta_i_centers = np.degrees(x[:, 0, 0, 0, 0]).tolist()
    phi_i_centers = np.degrees(x[0, :, 0, 0, 1]).tolist()
    theta_o_centers = np.degrees(x[0, 0, :, 0, 2]).tolist()
    phi_o_centers = np.degrees(x[0, 0, 0, :, 3]).tolist()

    # Collapse RGB -> scalar energy density (sum of channels).
    # FIX: mean across RGB, not sum -- for this non-spectral dielectric all
    # three channels carry the identical scalar value (no dispersion is
    # modeled at this IOR), so summing tripled the true energy value.
    # Verified directly: mi.Spectrum(0.5).x == .y == .z == 0.5 in this
    # variant. Mean recovers the correct per-channel value and stays
    # correct even if a future spectral variant gives genuinely different
    # R/G/B throughput.
    lum_t = rdm_t.mean(axis=-1)  # (theta_i, phi_i, theta_o, phi_o)
    lum_r = rdm_r.mean(axis=-1)
    lum_m = rdm_m.mean(axis=-1)

    global_max = float(max(lum_t.max(), lum_r.max(), lum_m.max(), 1e-9))

    out = {
        "diamond_name": str(data["diamond_name"]) if "diamond_name" in data else None,
        "theta_i_bins": theta_i_bins,
        "phi_i_bins": phi_i_bins,
        "theta_o_bins": theta_o_bins,
        "phi_o_bins": phi_o_bins,
        "theta_i_centers_deg": theta_i_centers,
        "phi_i_centers_deg": phi_i_centers,
        "theta_o_centers_deg": theta_o_centers,
        "phi_o_centers_deg": phi_o_centers,
        "global_max": global_max,
        # flat row-major (theta_i, phi_i, theta_o, phi_o) lists -- the
        # viewer reshapes/indexes these itself.
        "transmittance": lum_t.flatten().tolist(),
        "reflectance": lum_r.flatten().tolist(),
        "multiscatter": lum_m.flatten().tolist(),
    }

    out_dir = os.path.join("checkpoints/",args.checkpoint_name)
    out_path = os.path.join(out_dir, "rdm_viewer_data.json")
    with open(out_path, "w") as f:
        json.dump(out, f)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"[saved] {out_path} ({size_kb:.1f} KB)")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export rdm.npz to JSON for the HTML viewer.")
    parser.add_argument("--checkpoint_name", required=True)
    args = parser.parse_args()
    export(args.checkpoint_name)