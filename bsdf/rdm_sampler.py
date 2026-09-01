"""
rdm_sampler.py - Turns a collected RDM histogram (rdm.py's compute_rdm output)
into an actual importance sampler over outgoing directions, indexed by
incoming direction.

This is what NeuralDiamond.sample_from_rdm()/pdf() expect as `self.rdm_sampler`:
    sample_outgoing(theta_i, phi_i, u1, u2) -> theta_o, phi_o, pdf   (per-ray, drjit)
    pdf_outgoing(theta_i, phi_i, theta_o, phi_o) -> pdf              (per-ray, drjit)

Design: for each incoming-direction bin (theta_i, phi_i), the outgoing
directions form a discrete distribution over (theta_o, phi_o) bins, weighted
by the *energy* in that bin (T + M histograms, luminance-summed, weighted by
solid angle to get a proper probability mass rather than a density). We build
a Walker alias table per incoming bin at load time (numpy, done once) so that
sampling at render time is O(1) per ray. pdf_outgoing looks up the same
per-bin probability mass and divides by the outgoing bin's solid angle to
report a density, matching what NeuralDiamond.pdf() / the MC estimator expect.

If you saved the RDM tensors under different array names than compute_rdm()
returns, adjust `load_rdm_arrays` below.
"""

import numpy as np
import mitsuba as mi
import drjit as dr


def load_rdm_arrays(npz_path):
    """
    Load the arrays compute_rdm() produced. Expects an .npz with (at least):
        rdm_t, rdm_m          : (theta_i_bins, phi_i_bins, theta_o_bins, phi_o_bins, 3)
        count_t (or count_i)  : (theta_i_bins, phi_i_bins)
        solid_angles          : (theta_o_bins, phi_o_bins)  -- note: same grid
                                  used for both incoming/outgoing theta ranges
                                  in solid_angle_grid(theta_bins, phi_bins); the
                                  incoming half is solid_angles[:theta_i_bins].
    If your save used different keys, edit the lookups below rather than the
    rest of this file.
    """
    data = np.load(npz_path)

    def pick(*names):
        for n in names:
            if n in data:
                return data[n]
        raise KeyError(f"None of {names} found in {npz_path}. "
                        f"Available keys: {list(data.keys())}")

    rdm_t = pick("rdm_t")
    rdm_m = pick("rdm_m")
    solid_angles = pick("solid_angles")
    count_i = pick("count_t", "count_i", "count_m")  # identical across T/R/M, see rdm.py
    return rdm_t, rdm_m, solid_angles, count_i


class RDMAliasSampler:
    """
    theta_i_bins, phi_i_bins: incoming-direction grid (theta_i in [0, pi/2])
    theta_o_bins, phi_o_bins: outgoing-direction grid (theta_o in [0, pi])
    """

    def __init__(self, rdm_t, rdm_m, solid_angles, count_i, min_count=1.0):
        ti_bins, pi_bins, to_bins, po_bins, _ = rdm_t.shape
        self.theta_i_bins = ti_bins
        self.phi_i_bins = pi_bins
        self.theta_o_bins = to_bins
        self.phi_o_bins = po_bins

        solid_angles_out = solid_angles  # (to_bins, po_bins), full outgoing grid

        # Energy = luminance of T + M (what the BSDF actually needs good
        # samples for -- R is delta/analytic and doesn't go through this
        # sampler). Weight by outgoing solid angle to turn a per-steradian
        # histogram value into a probability *mass* per bin, since that's
        # what a discrete sampler needs to draw from.
        luminance = (0.2126 * (rdm_t[..., 0] + rdm_m[..., 0])
                     + 0.7152 * (rdm_t[..., 1] + rdm_m[..., 1])
                     + 0.0722 * (rdm_t[..., 2] + rdm_m[..., 2]))
        mass = np.clip(luminance, 0.0, None) * solid_angles_out[None, None, :, :]

        # Incoming bins with too few collected samples have unreliable
        # histogram values (see rdm.py discussion) -- fall back to a
        # uniform outgoing distribution for those rather than trusting
        # noisy near-zero energy estimates.
        unreliable = count_i < min_count  # (ti_bins, pi_bins)
        uniform_mass = solid_angles_out[None, None, :, :] * np.ones((ti_bins, pi_bins, 1, 1))
        mass = np.where(unreliable[:, :, None, None], uniform_mass, mass)

        flat_mass = mass.reshape(ti_bins * pi_bins, to_bins * po_bins)
        row_sum = flat_mass.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum <= 0, 1.0, row_sum)
        pmf = flat_mass / row_sum  # normalized probability mass per incoming bin

        self.pmf = pmf.reshape(ti_bins, pi_bins, to_bins, po_bins).astype(np.float32)
        self.solid_angles_out = solid_angles_out.astype(np.float32)

        n_outcomes = to_bins * po_bins
        self.n_outcomes = n_outcomes
        prob_table = np.zeros((ti_bins * pi_bins, n_outcomes), dtype=np.float32)
        alias_table = np.zeros((ti_bins * pi_bins, n_outcomes), dtype=np.int64)

        for row in range(ti_bins * pi_bins):
            prob_table[row], alias_table[row] = self._build_alias_row(pmf[row], n_outcomes)

        # Flatten to Dr.Jit arrays for per-ray gather at render time.
        self.mi_prob_table = mi.Float(prob_table.flatten())
        self.mi_alias_table = mi.UInt32(alias_table.flatten().astype(np.uint32))
        self.mi_pmf = mi.Float(self.pmf.flatten())
        self.mi_solid_angles_out = mi.Float(self.solid_angles_out.flatten())

    @staticmethod
    def _build_alias_row(p, n):
        """Standard Walker alias-table construction for one discrete distribution."""
        p = p * n  # scale so mean is 1
        prob = np.zeros(n, dtype=np.float32)
        alias = np.zeros(n, dtype=np.int64)
        small, large = [], []
        for i, pi in enumerate(p):
            (small if pi < 1.0 else large).append(i)
        while small and large:
            s = small.pop()
            l = large.pop()
            prob[s] = p[s]
            alias[s] = l
            p[l] = p[l] - (1.0 - p[s])
            (small if p[l] < 1.0 else large).append(l)
        for i in large:
            prob[i] = 1.0
        for i in small:
            prob[i] = 1.0
        return prob, alias

    # ------------------------------------------------------------------
    # Binning helpers (must match compute_histogram_4d's binning exactly)
    # ------------------------------------------------------------------

    def _incoming_bin(self, theta_i, phi_i):
        ti_idx = dr.clip(mi.UInt32(theta_i / (dr.pi / 2) * self.theta_i_bins),
                          0, self.theta_i_bins - 1)
        pi_idx = dr.clip(mi.UInt32((phi_i + dr.pi) / dr.two_pi * self.phi_i_bins),
                          0, self.phi_i_bins - 1)
        return ti_idx, pi_idx

    def _outgoing_bin(self, theta_o, phi_o):
        to_idx = dr.clip(mi.UInt32(theta_o / dr.pi * self.theta_o_bins),
                          0, self.theta_o_bins - 1)
        po_idx = dr.clip(mi.UInt32((phi_o + dr.pi) / dr.two_pi * self.phi_o_bins),
                          0, self.phi_o_bins - 1)
        return to_idx, po_idx

    def _bin_center(self, idx, n_bins, lo, hi):
        return lo + (mi.Float(idx) + 0.5) * (hi - lo) / n_bins

    # ------------------------------------------------------------------
    # Public sampler interface (called from NeuralDiamond)
    # ------------------------------------------------------------------

    def sample_outgoing(self, theta_i, phi_i, u1, u2):
        """
        Per-ray alias-method sample of an outgoing direction, conditioned on
        the (binned) incoming direction. Returns bin-center theta_o, phi_o
        (jittered within the bin) and the *density* pdf for that direction.
        """
        ti_idx, pi_idx = self._incoming_bin(theta_i, phi_i)
        row = ti_idx * self.phi_i_bins + pi_idx

        # Standard alias sampling: u1 selects a candidate outcome uniformly,
        # u2 decides whether to keep it or take its alias.
        u1_scaled = u1 * self.n_outcomes
        outcome = dr.clip(mi.UInt32(u1_scaled), 0, self.n_outcomes - 1)
        frac = u1_scaled - mi.Float(outcome)

        table_idx = row * self.n_outcomes + outcome
        keep_prob = dr.gather(mi.Float, self.mi_prob_table, table_idx)
        alias_outcome = dr.gather(mi.UInt32, self.mi_alias_table, table_idx)

        final_outcome = dr.select(frac < keep_prob, outcome, alias_outcome)

        to_idx = final_outcome // self.phi_o_bins
        po_idx = final_outcome % self.phi_o_bins

        # Jitter within the chosen bin so we don't only ever emit bin centers.
        theta_o = self._bin_center(to_idx, self.theta_o_bins, 0.0, dr.pi) \
            + (u2 - 0.5) * (dr.pi / self.theta_o_bins)
        phi_o = self._bin_center(po_idx, self.phi_o_bins, -dr.pi, dr.pi) \
            + (u2 - 0.5) * (dr.two_pi / self.phi_o_bins)
        theta_o = dr.clip(theta_o, 0.0, dr.pi)

        pdf = self._pdf_from_bins(ti_idx, pi_idx, to_idx, po_idx)
        return theta_o, phi_o, pdf

    def pdf_outgoing(self, theta_i, phi_i, theta_o, phi_o):
        ti_idx, pi_idx = self._incoming_bin(theta_i, phi_i)
        to_idx, po_idx = self._outgoing_bin(theta_o, phi_o)
        return self._pdf_from_bins(ti_idx, pi_idx, to_idx, po_idx)

    def _pdf_from_bins(self, ti_idx, pi_idx, to_idx, po_idx):
        pmf_idx = ((ti_idx * self.phi_i_bins + pi_idx) * self.theta_o_bins + to_idx) \
            * self.phi_o_bins + po_idx
        mass = dr.gather(mi.Float, self.mi_pmf, pmf_idx)

        sa_idx = to_idx * self.phi_o_bins + po_idx
        solid_angle = dr.gather(mi.Float, self.mi_solid_angles_out, sa_idx)
        solid_angle = dr.maximum(solid_angle, 1e-8)

        return mass / solid_angle


def build_rdm_sampler(checkpoint_dir, filename="rdm_data.npz", min_count=1.0):
    """
    Convenience loader: looks for `<checkpoint_dir>/<filename>`, builds and
    returns an RDMAliasSampler, or None (with a printed reason) if the file
    isn't there -- so callers can decide whether to fall back explicitly
    rather than doing so silently.
    """
    import os
    path = os.path.join(checkpoint_dir, filename)
    if not os.path.exists(path):
        print(f"  ⚠ RDM data not found at {path} -- "
              f"direction sampling will fall back to uniform hemisphere")
        return None
    rdm_t, rdm_m, solid_angles, count_i = load_rdm_arrays(path)
    sampler = RDMAliasSampler(rdm_t, rdm_m, solid_angles, count_i, min_count=min_count)
    print(f"✓ RDM alias sampler built from {path} "
          f"({sampler.theta_i_bins}x{sampler.phi_i_bins} incoming bins, "
          f"{sampler.theta_o_bins}x{sampler.phi_o_bins} outgoing bins)")
    return sampler