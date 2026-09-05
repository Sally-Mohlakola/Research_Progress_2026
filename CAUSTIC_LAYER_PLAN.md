# Adding a High-Frequency Specular Layer on Top of the RDM

Working design document for the "caustics on top of the RDM base" step.

Everything marked **[measured]** is a result already obtained in this project, with the
frame pair that produced it. Everything else is a proposal to be argued with. Items marked
**[verify]** are claims not yet checked against the current Mitsuba 3.9 API and that should
be confirmed before they are planned around.

Baseline for every number below: `renders/diamond_test_10/frames/frame_0000.exr` (neural)
against `renders/gt_test_13/frames/frame_0000.exr` (analytic ground truth), 768x768,
exposure 45.06 applied identically to both.

---

## 1. The framing is wrong, and that is good news

The intuition was: the RDM supplies base shading, so add a caustics pass on top. The first
half is roughly right (Section 3). The second half names the wrong phenomenon.

**[measured]** Where the missing energy actually lives:

| | stone p99/median | ground p99/median | ground-plane energy |
|---|---|---|---|
| `gt_test_13` (analytic GT) | **32.4** | 1.67 | 220.8 |
| `diamond_test_10` (neural) | 15.3 | 1.54 | 212.9 |

The ground plane is featureless in the ground truth as well, and the neural render already
reproduces its energy to within 3.6%. **There is no cast caustic in this scene to recover.**
The whole deficit sits on the stone itself.

What is missing is *direct-view internal specular transport*: camera rays that enter the
crown, totally internally reflect off pavilion facets, and exit toward the eye. In path
notation these are `E S+ L` paths — specular chains seen directly from the eye.

This rules out the standard caustic toolkit. Photon mapping / SPPM, Manifold Next Event
Estimation (Hanika et al. 2015) and Specular Manifold Sampling (Zeltner et al. 2020) all
exist to connect *light sources to diffuse receivers* through specular chains, i.e.
`L S+ D E` paths. That is not the missing term here. An ordinary path tracer already
samples `E S+ L` correctly — which is exactly why the analytic render has this structure
and the neural render does not.

**The RDM BSDF is what destroys these paths. Nothing needs to be added to recover them;
something needs to stop removing them.**

The measurement above is the justification for saying so in the methods section, and it
saves implementing a photon mapper.

---

## 2. Prerequisite: the base is not yet sound enough to build on

A residual layer is only meaningful on top of a correct base. Three defects say the base is
not there yet, and all three are cheap relative to Section 4.

### 2.1 Half the crown is dead **[measured]**

On `round_diamond_gia` at the eval camera, 32 of 64 facets are first-hit visible (8 star,
16 bezel, 8 table; the 24 pavilion and 8 culet facets are occluded by the crown). Of those
32, **12 are pinned at a near-constant floor**, and the separator is geometric:

| group | n | r(nn, ref) | brightness | analytic dynamic range | neural dynamic range |
|---|---|---|---|---|---|
| normal tilts up (elev >= 0) | 20 | +0.265 | 0.258x | 7.7x | 62x |
| normal tilts down (elev < 0) | 12 | +0.129 | **0.031x** | 16.9x | **1.4x** |

Every facet whose normal tilts below the horizon is dead: a 1.4x spread across facets where
the analytic reference varies 16.9x. They are not dark because the reference is dark —
analytic black fraction is 0.0% on every crown facet. `diamond_test_10` shows the same floor
(~0.00012–0.0004 on bezels f16–f18 and stars st02/st04/st06).

Root cause: `utils/rdm.py:401-406`. `collect_rdm` aims every gather ray at the origin:

```python
origin_radius = 3.0 * bounding_radius
o = mi.Point3f(mi.Float(dirs_np[:,0]) * origin_radius, ...)
d = mi.Vector3f(-mi.Float(dirs_np[:,0]), ...)
```

Aiming at a single point caps the achievable local incidence angle on each facet, leaving 23
of 64 incident bins empty (reproduced: 41/64 occupied). Downward-tilted facets are precisely
the ones whose (wi, wo) configurations are never gathered.

Aiming every ray at one point also means each sampled direction produces exactly *one* ray —
the line from `3R*u` through the centre — so a facet is only ever entered by the narrow cone
of directions it subtends from the centre, and its local incidence angle is pinned near its
own normal.

**Fixed.** `collect_rdm` now offsets each origin by a uniform sample on the disc of radius
`bounding_radius` perpendicular to the beam (`sample_aim_offsets`, `utils/rdm.py`), sweeping
the stone with a parallel beam over its full projected area. Ray *directions* are unchanged,
so whichever `--sampling_method` is in force still produces the incoming distribution it was
chosen for. `--no_aim_jitter` restores the old behaviour for reproducing run_09/run_10.

**[measured]** 200K rays, `round_diamond_gia`, theta_bins 8 / phi_bins 16:

| | centre-aimed | disc-jittered |
|---|---|---|
| incident bins occupied | 41/64 | **64/64** |
| occupancy by theta_i band | 16, 14, 11, **0** | 16, 16, 16, 16 |
| outgoing cells non-zero | 1179/24576 | **21723/24576** |
| samples per bin (min/median/max) | 0 / 1342 / 15901 | 775 / 1986 / 4064 |

The grazing band (theta_i in [67.5, 90]) was **completely empty** before — 0 of 16 bins —
which is the direct cause of the dead down-tilted facets, since those can only be lit at
grazing incidence from the crown side.

31.1% of jittered rays miss the stone by design (the disc is the bounding circle, the
silhouette is smaller). They are excluded from the histogram *counts* as well as zeroed in
throughput, via a new `active` argument to `compute_histogram_4d` — counting them would have
diluted every bin mean toward black in proportion to the miss rate.

Still to do: re-gather and retrain, then re-measure the Section 2.1 dead-facet table. The
numbers above are coverage, not appearance.

### 2.2 Darkness is never trained **[measured]**

`train_models.py:132`:

```python
if rgb.sum() > 1e-6:  # Non-zero bin
```

This drops every zero bin, so only 944 of 8192 bins (11.5%) reach the optimiser and the
network never sees a dark example. A model that cannot represent darkness cannot represent
the contrast the residual layer is supposed to sit against.

**Fixed.** A zero in `rdm_m` has two meanings and the old filter conflated them. Either the
incoming bin *was* sampled and no energy left in that outgoing direction — a measurement of
darkness, which belongs in the training set — or the incoming bin was never sampled, in
which case `compute_histogram_4d` divided a zero sum by a zero count and mapped the NaN to
0, so the array reads 0 without anything having been observed. The array cannot tell them
apart; `count_i` can, and was already in `rdm.npz` unused by the trainer.
`prepare_training_data` now takes it and keeps every bin under an observed incoming
direction, dark or lit, while still excluding the unobserved ones. `--drop_zero_bins`
restores the old filter.

`prepare_transmittance_data`'s similar-looking `total > 1e-6` guard is deliberately left
alone: its target is the ratio `total_t / total`, which really is undefined when nothing
left the stone. There is no measured-dark case to rescue there.

**[measured]** Composition, on run_09's existing RDM (8192 bins):

| | bins |
|---|---|
| lit | 949 |
| measured dark — was silently dropped | 4299 |
| unobserved — correctly still excluded | 2944 |

11.6% of the domain reaches the optimiser before, 64.1% after. Not 100%, because the 2944
unobserved bins are exactly the 23 empty incoming bins of Section 2.1 times 128 outgoing
bins — the two defects are quantitatively consistent, and **2.2's benefit is capped by 2.1
until the stone is re-gathered.**

**[measured]** Effect on the trained model. Model_M, 5000 epochs, lr 1e-3, batch 4096 —
run_09's own training command — evaluated over all 5248 usable bins:

| | old filter | fix 2.2 |
|---|---|---|
| dark bins predicted < 1e-3 | 24.9% | **85.6%** |
| dark bins predicted < 1e-2 | 55.0% | 92.9% |
| dark-bin prediction, median | 0.0065 | ~0 |
| dark-bin prediction, p90 | 0.209 | **0.0038** |
| dark-bin prediction, max | 87.7 | 1.59 |
| dark/lit mean ratio | 0.153 | **0.016** |
| log-space r on lit bins | +0.4704 | +0.4739 |
| RMSE on lit bins | 0.747 | 0.919 |

Darkness is now represented — a 10x improvement in dark/lit separation — at no cost to
lit-bin *correlation* and a 23% cost in lit-bin RMSE. That trade is the right way round for
a stone whose appearance is mostly contrast. Note the old model's 87.7 maximum on bins it
was never shown: unconstrained extrapolation into the dark part of the domain, which is the
mechanism by which the missing training data leaked into the render as a grey floor.

One caveat worth recording, because it cost a wrong conclusion once: at 400 epochs the fix
appears to *hurt* (dark/lit ratio 0.38 -> 0.78), because the extra 4299 near-zero targets
slow the fit globally under MSE on raw radiance before they sharpen it. The effect only
inverts with enough epochs. Any future ablation of this flag must run to convergence.

Still open, and not changed here because it is a modelling decision rather than a defect:
`train_model` uses `nn.MSELoss` on raw radiance while lit targets span 0.0006 to 7.07, so
the loss is dominated by a handful of bright bins. A relative or log-space loss is the
obvious candidate and would change the ablation against Soh & Montazeri, so it should be
argued for explicitly rather than slipped in.

### 2.3 The recorded commands gather and evaluate on different geometry **[measured]**

From `checkpoints/run_09/commands.md`:

```
python gather_rdm.py  --checkpoint_name run_09 --diamond_name round_diamond_gia ...
python eval.py        --checkpoint_name run_09 --diamond_name round_brilliant_sharp_culet ...
```

`round_diamond_gia` has 64 faces (culet radius 0.02); `round_brilliant_sharp_culet` has 48
(culet radius 0). The RDM is indexed by first-hit facet frame, so the bins do not correspond
between gather and render. Fixing this costs one word in the command.

Also in the same file: `run_10` was gathered with `--batch_size 1000 --num_batches 20`
(20K rays) against `run_09`'s `--batch_size 1000000` (20M rays). **[measured]** run_10
occupies 182 of 8192 outgoing bins against run_09's 949, and its Model_M is uncorrelated
with run_09's (r = -0.004). run_10 should not be used as a baseline.

**Recommendation: close 2.1–2.3 before starting Section 4.** They are small edits, and every
metric in Section 7 is currently measured against a base with half its crown missing.

---

## 3. What the RDM does and does not currently supply

**[measured]** `diamond_test_10` against `gt_test_13`, 32 visible crown facets:

| facets | n | r (facet means) | r (facet medians) | neural/GT mean |
|---|---|---|---|---|
| bezel | 16 | +0.698 | **+0.892** | 0.226x |
| table | 8 | -0.110 | -0.142 | 1.520x |
| star | 8 | -0.278 | +0.065 | 0.330x |

So the claim "the RDM supplies the base shading" holds in a specific and defensible sense —
**aggregate per-facet tone, most clearly on the bezels** — and fails in another.

**[measured]** Spatially, the base shading does not match at all:

- low-pass r(neural, GT) = **+0.047 / +0.060 / +0.087** at blur radii 8 / 16 / 32 px
- per-pixel r inside the stone = **+0.014**

**[measured]** And the energy is distributed the wrong way:

| | top 0.1% of pixels | top 1% | mean/median |
|---|---|---|---|
| `gt_test_13` (GT) | 9.7% of stone energy | **24.8%** | 4.79 |
| `diamond_test_10` | 2.5% | **9.1%** | 2.46 |

High-pass energy in the neural render is 0.36–0.50x the ground truth's. The neural render has
a *higher* median but a *lower* mean than the GT (1.43x vs 0.56x) — the signature of taking
roughly the right total energy and spreading it smoothly instead of concentrating it into
sparse bright structure.

This is the aggregated-BSDF formulation behaving exactly as designed. Soh & Montazeri's method
integrates over the outgoing lobe on purpose; that integration is what produces the per-facet
tone and what removes the see-through detail. They are the same operation. Caustics are
therefore not a separable feature that can be bolted on — which is what Section 4 has to
address structurally.

---

## 4. Making "on top of" well-posed

### 4.1 Why naive addition is wrong

The RDM already contains *all* internal transport, aggregated. Adding an explicit specular
layer on top double-counts every path the RDM already integrated. Any correct version of this
plan must first *remove* something from the RDM.

### 4.2 The decomposition

Split the transport by internal bounce depth `k`:

```
L  =  L_specular( <= k internal bounces, traced explicitly )
   +  L_rdm     (  > k internal bounces, learned          )
```

The explicit term restores the sharp `E S+ L` structure; the learned term keeps the neural
speedup for the long, decorrelated tail, which is the regime where an aggregate model is
actually appropriate.

**The critical requirement: the RDM must be re-gathered conditioned on internal bounce count
> k.** This is a filter in `gather_rdm.py` — count internal interactions along each gathered
path and reject those with `bounces <= k`. Without this conditioning the sum above is wrong
and the render will be too bright by roughly the short-path energy. This is the single most
important design point in this document.

`k = 2` or `k = 3` is the proposed starting target. The bright splinters in the ground truth
come from short TIR chains; beyond three bounces the distribution is much closer to the smooth
lobe the network already fits well. `k` should be chosen empirically by sweeping it and reading
the Section 7 metrics.

Stated this way the contribution is a residual decomposition, not a patch: *the aggregated
model captures the >k-bounce residual, while the high-frequency term is recovered
analytically.* That is a stronger claim for the paper than "we added caustics".

---

## 5. Three implementation routes, by cost

### Route A — depth-aware BSDF (the version worth writing up)

`NeuralDiamond` becomes depth-conditional: at path depth < k it behaves as a smooth dielectric
(Fresnel + Snell + total internal reflection); at depth >= k it samples the residual RDM.

Complication: **[verify]** Mitsuba's `BSDFContext` does not appear to carry path depth, so the
depth must be tracked in a custom integrator that selects the lobe, or threaded through some
other channel. Confirm against the 3.9 API before planning around this.

Effort: moderate. This is the real contribution.

### Route B — two-pass composite (the fallback)

Render pass 1 analytically with `max_depth = k+1`, render pass 2 with the neural residual RDM,
and add the two images. Touches no BSDF code and produces a figure in days.

This is not a renderer — it cannot handle a stone that is partly occluded or interreflecting
with its environment — but as a proof that the decomposition in 4.2 recovers the missing
structure it is a legitimate ablation, and a sound fallback if time runs short.

### Route C — path-space regularization

Kaplanyan & Dachsbacher 2013: roughen specular interactions slightly so that next-event
estimation can connect through them, with the bias vanishing as roughness goes to zero.

Section 1 says this is not needed for the current scene. It is the correct tool if cast
caustics are added later, e.g. for a scene where the stone sits on a brightly lit surface
rather than the current dark plane.

---

## 6. This is also the route to fire

**The RDM can never produce dispersion.** It bins radiance into RGB (`train_models.py` operates
on `rgb`), so wavelength-dependent refraction is averaged away before the network sees it. No
amount of retraining recovers spectral fire from an RGB-binned aggregate.

Spectral dispersion can only come from the explicitly traced short paths of Section 4.2, where
each sampled wavelength refracts at its own IOR. Under `llvm_ad_spectral` the surrounding
machinery is already in place; what is needed is a wavelength-dependent IOR (a Sellmeier fit
for diamond) on the specular lobe.

**[verify]** Mitsuba's stock `dielectric` takes a scalar `int_ior` and does not appear to expose
a spectral one, so this likely requires a small custom BSDF. Confirm before scheduling it.

This matters for scope: the depth-split architecture serves *both* stated appearance goals —
the kaleidoscopic see-through look and the fire — from one change. Worth saying explicitly in
the write-up.

---

## 7. Validation

Reuse the metrics from this comparison so the write-up has a before/after table on the same
frame pair.

| metric | GT | current | target |
|---|---|---|---|
| stone p99/median | 32.4 | 15.3 | > 25 |
| top-1% energy share | 24.8% | 9.1% | > 20% |
| high-pass energy ratio | 1.0 | 0.36–0.50 | > 0.8 |
| low-pass r (base shading) | 1.0 | +0.06 | > 0.7 |
| bezel per-facet median r | 1.0 | +0.89 | do not regress |

Two notes on reading these:

- **The last row is a guardrail.** Aggregate facet tone is the one thing currently working; the
  specular layer must not cost it.
- **Establish the reliability ceiling before quoting any correlation.** **[measured]** Two
  independent analytic renders of the same scene at 64spp agree at r = 0.9996 with a median
  per-facet difference of 2.1%, so at this sample count reference noise is not a meaningful
  attenuator and correlations can be read at face value. Re-measure this if the sample count
  drops.

---

## 8. Prior work to engage with

- **Guy & Soler 2004**, *Graphics Gems Revisited: Fast and Physically-Based Rendering of
  Gemstones* (SIGGRAPH). The closest prior art: real-time gemstone rendering with precomputed
  internal reflection paths and dispersion on fixed convex facet geometry. Very close to the
  architecture proposed here and should be engaged with directly.
- **Soh & Montazeri 2024**, *Neural Appearance Model for Cloth Rendering* (EGSR, CGF 43-4). The
  method being adapted; Section 3 is the argument about where the adaptation breaks.
- **Walter et al. 2007**, *Microfacet Models for Refraction through Rough Surfaces*. The
  dielectric model for the explicit lobe.
- **Kaplanyan & Dachsbacher 2013**, *Path Space Regularization*. Route C.
- **Jakob & Marschner 2012** (Manifold Exploration), **Hanika et al. 2015** (Manifold Next Event
  Estimation), **Zeltner et al. 2020** (Specular Manifold Sampling). Cite as the standard caustic
  machinery that the measurement in Section 1 rules out — a deliberate, evidenced scoping
  decision rather than an omission.

---

## 9. Process note

`commands.md` records no command for either `diamond_test_10` or `gt_test_13`, and both recorded
`eval.py` lines write to `renders/diamond_test_14` with `--no_neural`, which would put an
*analytic* render into a directory named `diamond_test_*`.

Two provenance problems have already cost time in this project: the gather/render geometry
mismatch in 2.3, and a `gt_test_13` whose facet structure identifies as a round brilliant (n=8)
despite having been launched with `--diamond_name princess`. **[measured]** Silhouette coverage
is 95.2% for the round presets against 87.7% for princess, and facet-boundary edge score 1.228
against 1.067.

Writing the exact invocation into each output directory at render time would remove this whole
class of doubt before the write-up.
