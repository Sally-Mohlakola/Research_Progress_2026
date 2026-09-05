# Analytic / Neural Split: `r17_base` and `diamond_test_27`

Measurement report. Every number marked **[measured]** comes from renders in
this repository; nothing here is estimated.

## Method

The split is measured by ablation. Each configuration is rendered twice with
identical settings, seeds and geometry, differing only in `--clamp_value`:

* `--clamp_value 10` — normal render.
* `--clamp_value 0` — the activation in `eval.py` becomes
  `min(exp(min(x, 20)), 0)`, which is identically zero, so `Model_M`
  contributes nothing. Every analytic component is untouched.

The difference between the two images is the neural contribution; the
`clamp 0` image is the analytic contribution. Each configuration is compared
against its own analytic reference (`--no_neural`), rendered at the same
resolution and camera, so no cross-resolution comparison is involved.

The stone mask is the brightest 3% of the analytic reference after a 3-pixel
box blur; "ground" is everything else that receives light.

| render | configuration | checkpoint | resolution | reference |
|---|---|---|---|---|
| `r17_base` | aggregate-only (`--no_explicit_entry`) | run_17 | 768² | `gt_test_22` |
| `diamond_test_27` | Route A (explicit entry) | run_13 | 728² | `gt_test_27` |

Note that `--no_explicit_entry` removes the *analytic* dielectric, so it is
the **more** neural configuration, not the less.

---

## 1. The split **[measured]**

### `r17_base` — aggregate-only

| region | total | analytic | neural | neural share |
|---|---|---|---|---|
| whole image | 1687.95 | 382.65 | 1305.30 | **77.3%** |
| stone | 107.12 | 18.28 | 88.84 | **82.9%** |
| ground | 1580.83 | 364.37 | 1216.46 | 77.0% |

### `diamond_test_27` — Route A

| region | total | analytic | neural | neural share |
|---|---|---|---|---|
| whole image | 423.96 | 394.94 | 29.02 | **6.8%** |
| stone | 54.54 | 52.90 | 1.65 | **3.0%** |
| ground | 369.41 | 342.04 | 27.37 | 7.4% |

The two configurations sit at opposite ends. In the aggregate-only baseline
the network carries 83% of the light reaching the visible stone. Under
Route A it carries 3%, because `sample_entry` — an analytic dispersive
dielectric with Fresnel, Snell, total internal reflection and dispersion —
has already produced the image before the residual is consulted.

---

## 2. Energy conservation **[measured]**

Against each configuration's own analytic reference:

| | whole image | stone | ground |
|---|---|---|---|
| `r17_base` | **0.941x** | **0.163x** | **1.391x** |
| `diamond_test_27` | **0.261x** | 0.091x | 0.360x |

These are two different failures and they matter differently.

**The aggregate conserves energy and misdirects it.** `r17_base` returns 94%
of the reference's total light — the RDM is energy-conserving by
construction (gathered albedos: `rdm_t` 0.084, `rdm_r` 0.227, `rdm_m` 0.680,
total 0.991) and that survives into the render. What does not survive is
*where* the light goes:

| | onto the stone | onto the ground |
|---|---|---|
| neural energy in `r17_base` | **6.7%** | 93.3% |
| analytic reference | 36.6% | 63.4% |

A 5.5x deficit on the visible stone, with the surplus landing on the floor.
At an 8x16 outgoing grid each bin spans roughly 22 x 22 degrees, so a
few-degree exit lobe is smeared across some 500 square degrees, most of which
does not point back at the viewer.

**Route A does not conserve energy at all.** 0.261x overall is not a
redistribution problem, it is missing transport, and its cause is known:
`utils/rdm.py` classifies `select_m = escaped & (depth >= 3)`, and Model_M
trains on `rdm_m` alone, while Route A's explicit term covers only the entry
interaction. The direct enter-and-exit path at depth 2 — which belongs to
`rdm_t` — is represented by neither term. The measured 0.26x is that gap.

---

## 3. Visual effect **[measured]**

All figures on the stone, normalised to each configuration's own analytic
reference. `highpass` is high-frequency energy as a fraction of the
reference's; `pixel r` is per-pixel correlation with the reference.

### `r17_base` — aggregate-only

| | brightness | p99/median | top-1% | highpass | pixel r |
|---|---|---|---|---|---|
| analytic reference | 1.000 | 22.20 | 15.4% | 1.000 | +1.0000 |
| analytic part only | 0.028 | 47.71 | 14.2% | 0.028 | −0.0024 |
| neural part only | 0.135 | 7.14 | 7.3% | 0.063 | +0.0066 |
| full render | 0.163 | 7.81 | 7.1% | 0.072 | +0.0044 |

The neural term supplies most of the brightness (0.135 of 0.163) and all of
the flatness. Its contrast is 7.14 against the reference's 22.20, and adding
it to the analytic lobe *lowers* the render's contrast from 47.71 to 7.81.
Aggregation is not merely failing to add structure — it is actively
smoothing away the structure the analytic lobe had.

### `diamond_test_27` — Route A

| | brightness | p99/median | top-1% | highpass | pixel r |
|---|---|---|---|---|---|
| analytic reference | 1.000 | 22.84 | 16.1% | 1.000 | +1.0000 |
| analytic part only | 0.089 | 549.38 | 26.6% | 0.069 | +0.0577 |
| neural part only | 0.003 | 4.80 | 5.6% | 0.002 | +0.0301 |
| full render | 0.091 | 302.64 | 25.8% | 0.070 | +0.0580 |

Here the neural term is a rounding error on the stone — 0.003 brightness
against the analytic part's 0.089 — and the render inherits the explicit
term's character: far too contrasty (302.64 against 22.84) because it
consists of a few bright specular hits on an otherwise black stone.

---

## 4. What this says

The two configurations fail in exactly opposite ways, and each has what the
other lacks:

| | energy | placement |
|---|---|---|
| aggregate-only | conserved (0.941x) | wrong — 6.7% onto the stone against 36.6% |
| Route A | lost (0.261x) | best available — pixel r +0.058, contrast far too high |

This is the empirical case for the bounce-count decomposition, stated in
measurements rather than argument. The aggregate holds the energy but cannot
say where it goes; explicit transport says exactly where it goes but only for
the paths it traces. Neither is a diamond on its own.

It also sharpens the research claim. "Pure RDM-based surface aggregation
systematically under-represents non-local specular paths" is confirmed, and
the mechanism is now specific: **the under-representation is directional, not
energetic.** The aggregate delivers 94% of the light and puts a sixth of the
correct share of it on the stone. That is a stronger and more defensible
sentence than an energy-loss claim, because the energy loss is measurably not
what happens.

## 5. What the neural term is *not* responsible for: colour **[measured]**

It is easy to look at `diamond_test_27` — a black stone with a few lit facets,
one blue and the others gold-grey — and attribute the colour to the learned
term, since the analytic dielectric reads as "the boring part". The ablation
says otherwise, and the margin is not close.

The RDM cannot represent colour at all. The gather scene uses an achromatic
`dielectric`, so all three channels of every bin hold identical throughput:

```
rdm_m per-bin channel spread (max-min)/max : mean 0.000e+00, max 0.000e+00
```

Measured over the visibly lit facets (pixels above 5x the stone median, 3925
px, about a quarter of the stone):

| region | total | analytic | neural | neural share |
|---|---|---|---|---|
| all lit facets | 51.988 | 51.492 | 0.496 | **0.95%** |
| blue patch (1929 px) | 8.983 | 8.712 | 0.272 | 3.02% |
| warm patch (1970 px) | 42.961 | 42.739 | 0.222 | 0.52% |

And the casts themselves survive with the network switched off. Mean RGB,
box-blurred at radius 4 so per-pixel noise is not driving the result:

| | R | G | B | ratio |
|---|---|---|---|---|
| blue patch, full render | 0.00399 | 0.00446 | 0.00694 | B/R 1.76 |
| blue patch, analytic only | 0.00385 | 0.00432 | 0.00678 | B/R 1.76 |
| blue patch, neural only | 0.00014 | 0.00014 | 0.00016 | B/R 1.14 |
| warm patch, full render | 0.02607 | 0.01905 | 0.01569 | R/B 1.66 |
| warm patch, analytic only | 0.02594 | 0.01894 | 0.01558 | R/B 1.66 |
| warm patch, neural only | 0.00014 | 0.00011 | 0.00011 | R/B 1.27 |

The analytic-only render carries the identical blue and identical warm cast to
three significant figures, while the neural layer sits 25x to 180x below it
and is close to neutral.

**Fire is an analytic capability of this renderer.** `sample_entry` refracts
with `hero_eta`, committing each path to one wavelength through the Sellmeier
fit in `bsdf/dispersion.py`; different facet orientations deliver different
surviving wavelengths to the camera. The residual can only inherit whichever
wavelength its path already carries, which is why it leans faintly blue inside
the blue patch and faintly warm inside the warm one without being able to
create either.

This is worth stating explicitly in the write-up rather than left ambiguous.
"The neural appearance model reproduces the stone's fire" would be an
overclaim, and the RGB-binned RDM makes it one that cannot be repaired by
training — only by adding a wavelength axis to the gather.

*Methodological note:* an earlier version of this measurement, taken on raw
per-pixel values, reported the neural layer as the *most saturated* in the
image (0.794 against the analytic part's 0.333) and pointed to the opposite
conclusion. That was an artefact: at spp 8 a dim single-hero-wavelength sample
is saturated almost by construction. Blurring before measuring removes it.
Any colour attribution at this sample count should be treated as unreliable —
figures for publication want spp 128 or above.

## 6. Caveats

* The two configurations use different checkpoints (run_17 and run_13) and
  resolutions (768² and 728²). Each is compared only against its own matched
  reference, so the *shares* and *ratios* are sound, but the raw energy
  totals are not comparable between the two tables.
* Both checkpoints use the default `Model_M` (width 21, no input encoding),
  which is the largest network `neural/drjit_wrapper.py` can evaluate. A
  frequency-encoded 128-wide network fits the same RDM with 4.7% median
  relative error against this one's 55.5%, so every neural figure here is a
  floor rather than the method's ceiling.
* `Model_T` is trained in every run and never called by the BSDF. It
  contributes 0% in both configurations.
* Single frame, spp 8, one seed per condition. The energy ratios are stable
  quantities but the per-pixel correlations are noisy at this sample count.
