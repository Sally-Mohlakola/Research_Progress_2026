# Facet-Conditioned Neural Appearance Model for Faceted Gems

Working design document. Everything here is a proposal to be argued with, except the
items marked **[measured]**, which are results already obtained in this project.

---

## 1. The problem

This project adapts Soh & Montazeri 2024, *Neural Appearance Model for Cloth Rendering*
(EGSR, CGF 43-4), from yarn to a round-brilliant diamond. The pipeline gathers a Radiance
Distribution Map (RDM) by tracing light through the stone, splits the result into direct
transmission, direct reflection and multi-scattering (`S = S_T + S_R + S_M`, their Eq. 6),
and fits small networks to the components.

The port is numerically healthy. **[measured]** After fixing the batch-averaging and
count-aliasing defects, total albedo is 1.0047 and `rdm_r` throughput per ray is 1.000,
so energy is conserved and the gather is trustworthy.

The renders are nevertheless wrong in a specific way: they are *flat*. The target
appearance — the kaleidoscope, where facets are visible through other facets in sharp
bright and dark wedges — is absent, and no amount of retraining has recovered it.

**[measured]** Linear-EXR statistics over the stone region:

| render | median | mean | black px | p99/median |
|---|---|---|---|---|
| test_14 | 0.00223 | 0.0993 | 40.0% | — |
| test_15 | 0.0328 | 0.0861 | 19.8% | — |
| ground truth (`--no_neural`) | 0.0742 | 0.3554 | **0.0%** | 28.0 |

The ground truth has high contrast and no black facets. The neural renders are dim,
low-contrast and pockmarked.

## 2. Diagnosis

Two independent causes, both structural rather than incidental.

**2.1 The same-point assumption is violated.** The paper's model is a BSDF
`f(omega_i, omega_o)`. It has no position argument and explicitly assumes light leaves at
the point where it entered. **[measured]** On this project's actual mesh, light exits a
median of **1.23 stone radii** from where it entered; 66% of rays exit more than a full
radius away. The assumption is not merely approximate here, it is maximally violated.

**2.2 The parameterization deletes facet identity.** In `utils/rdm.py:312`, incidence is
binned as a *local* angle against the hit facet's own frame. In cloth this is correct and
is the source of the method's compactness: fibers are interchangeable, so factoring out
local orientation loses nothing. On a diamond it means a table facet and a pavilion facet
at the same local incidence angle fall in the same bin and are averaged together.

Cause 2.2 is the one worth attacking. The flatness is not an under-trained network or an
under-resolved grid — it is the parameterization behaving exactly as designed. The model
is being asked to reproduce a spatial pattern from a representation built to discard
spatial information.

The deeper reason the transfer fails: aggregation suits cloth because yarn microstructure
is statistically uniform and sub-pixel, so averaging hundreds of random fibers is
invisible. A diamond's facets are few, ordered and deterministic. Its kaleidoscope is
signal, not noise, and averaging deletes precisely the target appearance.

## 3. Proposed method

Condition the appearance model on **facet identity**, so each facet learns its own
outgoing distribution instead of being averaged into a shared one.

Replace

```
f_M(omega_i, omega_o)
```

with

```
f_M(omega_i, omega_o, k),    k = facet index at the shading point
```

The RDM gains a leading axis, from `[theta_i, phi_i, theta_o, phi_o]` of shape
`(4, 16, 8, 16)` to `[k, theta_i, phi_i, theta_o, phi_o]`.

`Model_M` gains a facet input. With a learned embedding `e: k -> R^8`, the network goes
from `6-21-21-21-3` to `14-21-21-21-3`, its input being `[omega_i, omega_o, e(k)]`. The
embedding is what makes this better than K independent tables: the network shares
statistics across geometrically similar facets and specialises only where they actually
differ, which is the same compactness argument the original paper makes, applied along a
new axis.

### Why this is cheap rather than a research gamble

The conditioning variable is free at both ends.

- **Gather.** `utils/rdm.py` already holds the first-hit `si`; `si.prim_index` is the
  entry facet. `ravel_index` already takes a shape tuple, so this is an added axis, not a
  rewrite.
- **Render.** In `bsdf/neural_bsdf.py`, `eval` (line 284), `sample` (line 230) and `pdf`
  (line 296) all receive `si`. Same `si.prim_index`. No interface change, no new data
  structure, no change to how Mitsuba drives the BSDF. It remains a BSDF.

The analytic reflection lobe (`eval_r`, `pdf_r`, `sample_r`, already implemented) is
unaffected and stays as it is.

## 4. Choosing K

Three candidates, in increasing sophistication. This is the main open decision.

**(a) Raw triangles, K = 72.** Simplest. But the RDM becomes 589,824 cells; at the
current ~20M gathered rays that is roughly 34 samples per cell, which is too sparse. Would
need a substantially longer gather.

**(b) Coplanar groups, K ~ 33.** Cluster triangles by normal so each logical facet (table,
star, kite, upper girdle, pavilion main) is one class. Roughly halves the cell count and
matches how a gemmologist would describe the stone.

**(c) Symmetry classes, K ~ 9.** A round brilliant has 8-fold rotational symmetry about
the vertical axis. Facets related by that symmetry are identical up to a rotation of
`phi`. Fold the facet index by the symmetry group and rotate `omega_i` and `omega_o`
correspondingly. This multiplies the data per class by about 8 while *losing nothing*,
because the facets being merged are genuinely equivalent.

Option (c) also gives the paper its cleanest sentence: **aggregate over the symmetry
group, not over structurally distinct facets.** That is the precise statement of what the
original method got wrong in this domain — it is not that aggregation is invalid, it is
that it was applied along the wrong equivalence relation.

Note that folding by symmetry does not force a symmetric image: the model is conditioned
on `omega_i` in the folded frame, so asymmetric illumination still produces asymmetric
appearance.

## 5. Conditioning variable: identity or geometry?

Two variants, worth deciding deliberately.

**(i) Discrete identity.** `nn.Embedding(K, 8)` on the facet index. Simple, fits fast,
specialises freely. Does not transfer to a different cut — the embedding is meaningless
for a facet the model never saw.

**(ii) Continuous geometric descriptor.** Condition instead on the facet's own geometry:
its normal in the stone-local frame, plus (say) the height of its centroid. The model then
learns a function of *what kind of facet this is*, not *which facet this is*, and can be
evaluated on a cut it was never trained on.

Variant (ii) is the stronger contribution — it turns the method from a fitted lookup for
one stone into an appearance model for faceted gems in general — but it is harder to fit
and needs training data spanning several cuts to demonstrate the claim. Suggested plan:
build (i) first as the stepping stone and the correctness check, and treat (ii) as the
headline result if time allows.

## 6. Implementation order

1. `utils/rdm.py` — capture `prim_index` at first hit, add the facet axis to
   `compute_histogram_4d` and the count array. Upstream dependency; nothing else can be
   tested until this lands.
2. `bsdf/rdm_sampler.py` — alias tables become per `(k, theta_i, phi_i)` rather than per
   `(theta_i, phi_i)`. Row count rises from 64 to 64K; memory is not a concern at these
   sizes.
3. `neural/base_model.py` — widen `Model_M` input, add the embedding.
4. `train_models.py` — carry the facet index through `prepare_training_data`.
5. `bsdf/neural_bsdf.py` — thread `si.prim_index` into `eval_model_m` and
   `sample_from_rdm`.

## 7. Evaluation

### Ablation ladder

| | configuration |
|---|---|
| A0 | path-traced ground truth (`--no_neural`) |
| A1 | unconditioned neural (current state; the baseline that fails) |
| A2 | A1 + analytic `S_R` (already implemented) |
| A3 | A2 + facet-conditioned `S_M` — **the method** |
| A4 | A3 + symmetry folding |
| A5 | A4 + within-facet position — stretch, see §9 |

### The control experiment that must be run

A resolution sweep on the unconditioned model (A1) at 3-4 RDM resolutions, plotting error
against A0. The current grid is 8,192 cells against the paper's ~8,019,000, so the first
objection any examiner will raise is that the failure is simply undersampling. If error
**plateaus** well above zero as resolution rises, that objection is closed and the failure
is established as structural. Without this plot the central claim of the write-up is
contestable; with it, it is not.

### Metrics

- RMSE and SSIM (or FLIP) against A0.
- Black-pixel fraction and p99/median contrast ratio — already tracked, and a reasonable
  proxy for whether the kaleidoscope survives, since aggregation flattens contrast.
- **Per-facet mean radiance correlation.** Compute mean radiance per facet in A0 and in
  each render, then report the correlation. This measures directly and only the thing the
  method is supposed to add, and it will separate A3 from A1 far more legibly than a
  whole-image metric.

## 8. Limitations to state plainly

Conditioning on the shading facet gives **per-facet** variation — facets differing in
brightness and colour according to their real internal path structure. That is a genuine
spatial signal the current model cannot represent at all.

It does **not** give a true refracted image of the environment seen through the stone. The
model still does not know the entry point when it is evaluated at the exit, so the
see-through effect remains out of reach for a single-point BSDF. Naming this boundary
precisely is worth marks rather than costing them; the honest claim is *restores
facet-level spatial variation*, not *solves gem rendering*.

Also worth conceding: the original paper's 23x speed and 300x memory wins came from cloth
having hundreds of fibers per yarn across millions of yarns. This stone is 72 triangles
with an analytic dielectric, which Mitsuba renders natively. The contribution here is
representational, not a performance result, and should not be dressed up as one.

## 9. Open questions to workshop

1. **K.** Raw triangles, coplanar groups, or symmetry classes? (§4 argues for (c), but the
   folding transform needs to be written and verified.)
2. **Conditioning variable.** Discrete embedding or continuous geometric descriptor? (§5)
3. **Frames.** Once facet identity is an explicit input, is the local-frame `theta_i` still
   the right parameterization, or should directions move to the stone frame? The network
   can in principle recover one from the other, but one of the two is likely far easier to
   fit, and this has not been tested.
4. **Scope of conditioning.** Does `S_T` need the facet input too, or only `S_M`?
5. **Sampler.** Do the alias tables genuinely need to be per-facet, or is a single shared
   proposal distribution adequate, with the `value/pdf` ratio absorbing the difference?
   Shared tables would be much cheaper; the cost is variance, and it is unclear how much.
6. **Within-facet detail (A5).** Adding `si.uv` would give variation *inside* a facet, not
   just between facets. Blocked: `ground_truth/brilliant_geometry.py:235` assigns the
   identical UV template `[[0,0],[1,0],[0,1]]` to every face, so UVs are degenerate and
   real ones must be generated first. Do not start here.
7. **Two-point extension.** Is there a tractable middle ground between a BSDF and a full
   BSSRDF — conditioning on entry facet *and* exit facet — that stays compatible with how
   Mitsuba invokes a BSDF? Probably not without a custom integrator, but worth ten minutes
   of thought before it is ruled out.

## 10. Positioning

The framing that makes this a contribution rather than a reimplementation:

> Aggregated neural appearance models assume the aggregated microstructure is
> statistically uniform. We show this assumption fails for faceted gems, quantify the
> failure, and propose facet-conditioned aggregation, which restores spatial variation
> while retaining the compactness of the neural representation.

Every element of that sentence is backed by something in this document: the failure is
measured (§2), the quantification is the 1.23-radii study and the resolution plateau (§7),
and the proposal is §3 with an ablation ladder that isolates its effect.
