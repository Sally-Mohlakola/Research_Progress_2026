# Implementation Plan — One Month Build, One Month Write

Companion to `FACET_CONDITIONED_BSDF.md`, which holds the *method*. This file holds the
*schedule and the experiment design*.

Target: 90%+. That target, not the four weeks, is what drives the changes below. A
faithful port plus a diagnosis plus a working fix on one stone lands in the high
seventies to mid eighties. The thing that moves it past ninety is **generalisation
evidence** — a model that transfers to a cut it was never trained on.

**Code freeze: end of month 1.** Month 2 is analysis, figures and writing, all of which
can be done from archived EXRs without touching the renderer. Protect the freeze.

---

## 0. STOP — a confound to fix on day one

**Every gather in `commands.md` used a different stone from every render.**

```
gather_rdm.py  --diamond_name round_diamond_gia            # culet_radius = 0.02
eval.py        --diamond_name round_brilliant_sharp_culet  # culet_radius = 0.00
```

There are two independent preset tables that do not share a single name:

- `config/parameters.py` → `DIAMOND_VARIANTS`: `round_diamond_gia`,
  `round_diamond_deep`, `round_diamond_sharp_culet`
- `eval.py` → `DIAMOND_PRESETS`: `round_brilliant_sharp_culet`,
  `round_brilliant_culet`, `princess`

So the RDM was fitted on a stone with a small flat culet facet and then rendered on a
stone with a sharp point culet. The meshes differ near the tip: the sharp-culet variant
drops the culet facet entirely and reshapes the adjoining pavilion facets.

**Consequences.** Every render statistic recorded so far — test_14, test_15, and the
ground-truth comparison — carries this mismatch. Some of the discrepancy attributed to
aggregation may be geometry mismatch instead. This does not overturn the diagnosis (the
1.23-radii result is independent of it, and a culet facet cannot explain a flat render),
but a table of numbers produced under a silent geometry mismatch will not survive an
examiner, and it must not appear in the write-up.

**Fix, before anything else:**

1. Delete `DIAMOND_PRESETS` from `eval.py` and import `get_diamond_parameters` from
   `config/parameters.py`. One table, one source of truth. Port `princess` into it.
2. Record the resolved parameter dict into `rdm.npz` at gather time, and have `eval.py`
   assert the render geometry matches what the checkpoint was gathered on. A mismatch
   should be a hard error, not a silent difference.
3. Re-run one baseline gather+render pair on matched geometry to confirm nothing else
   was resting on the mismatch.

This costs under a day and removes an objection that would otherwise sit under every
result in the paper.

---

## 1. The thesis that earns 90+

Not *"aggregated neural appearance models fail on faceted gems"* — that is a negative
result about someone else's method, and a hostile reader will note that the source paper
already lists the same-point assumption as a limitation.

Instead:

> Aggregated neural appearance models assume the aggregated structure is statistically
> uniform. For faceted gems it is not, and we show the resulting error is **not
> recoverable by increasing resolution**. We propose conditioning on facet *geometry*
> rather than facet identity, which restores spatial variation and — because the
> conditioning variable is a property of the facet rather than an index into one stone —
> **transfers to cuts the model was never trained on**.

Three claims, each with an experiment:

| claim | experiment |
|---|---|
| The failure is structural, not undersampling | Resolution sweep, error plateaus above zero |
| Facet conditioning restores spatial variation | Ablation ladder + per-facet correlation |
| The model generalises across cuts | **Train on a family of cuts, test on held-out cuts** |

The third is the one that lifts the mark, and it is the one the previous version of this
plan treated as a stretch goal. It is now the centrepiece.

---

## 2. Design revision: condition on geometry, not identity

`FACET_CONDITIONED_BSDF.md` §5 offered two variants — a learned per-facet embedding (i),
or a continuous geometric descriptor (ii) — and suggested building (i) first. **Build
(ii) directly.** Three reasons:

1. **It is not harder.** A learned `nn.Embedding(K, 8)` is replaced by four numbers
   already available from the mesh. There is no embedding table to size, initialise or
   debug.
2. **The K question dissolves.** With a continuous descriptor there is no grouping
   decision at all: every triangle carries its own descriptor, and data sharing across
   similar facets happens through the network's own smoothness rather than through an
   equivalence relation you had to choose and defend. The entire §4 debate — 72 vs 33 vs
   9 — disappears, and with it the K sweep.
3. **An index cannot transfer.** Embedding row 17 is meaningless for a cut the model
   never saw. A facet normal is meaningful for any cut. Generalisation is only possible
   under (ii).

**Descriptor.** Per triangle, in the stone-local frame: the facet normal (3 values) and
the centroid height normalised by girdle radius (1 value). Four extra inputs, so `Model_M`
goes `6-21-21-21-3` → `10-21-21-21-3`. Consider adding the centroid's radial distance if
the table and culet prove hard to separate; keep it under six.

**Sampler.** The alias tables are the one place the raw per-triangle histogram is too
sparse to use directly. Build them on coplanar-grouped facets (K ≈ 33) while the network
consumes the continuous descriptor. This is safe: the proposal distribution affects
variance only, not correctness, because the `value/pdf` ratio corrects for it. State that
justification explicitly in the write-up — it is the sort of detail that signals you
understand your own estimator.

---

## 3. Month 1 — build

### Week 1 — unblock, then instrument

- **Day 1:** the preset fix in §0.
- **Day 1:** start the **resolution sweep**. It runs unattended, and it is the control
  that closes the undersampling objection. Trimmed to three points to make room:

```
python gather_rdm.py --checkpoint_name sweep_08 --diamond_name <matched> \
    --batch_size 1000000 --num_batches 20 --theta_bins 8  --phi_bins 16 --max_depth 64
python gather_rdm.py --checkpoint_name sweep_16 ... --theta_bins 16 --phi_bins 32 ...
python gather_rdm.py --checkpoint_name sweep_24 ... --theta_bins 24 --phi_bins 48 --num_batches 80
```

  Cells: 8,192 / 131,072 / 663,552. The top point is sparsest at a flat ray budget, so it
  gets 4× the rays — otherwise a rise there reads as variance, not signal, and the plot
  proves nothing.

- **Rest of week:** the facet axis and descriptor in the gather.
  1. `utils/rdm.py` — capture `si_first.prim_index` in `trace_path`; add a leading facet
     axis to `compute_histogram_4d` and the count array. `ravel_index` already takes a
     shape tuple, so this is an added dimension, not a rewrite.
  2. Emit a per-triangle descriptor table (normal, centroid height) into `rdm.npz`
     alongside the histogram.
  3. `bsdf/rdm_sampler.py` — alias tables per `(coplanar group, theta_i, phi_i)`.

**Deliverable:** sweep running; one facet-conditioned RDM gathered on matched geometry.

### Week 2 — the model works on one stone

- `neural/base_model.py` — `Model_M` input 6 → 10.
- `train_models.py` — carry the descriptor through `prepare_training_data`.
- `bsdf/neural_bsdf.py` — thread `si.prim_index` into `eval_model_m` and
  `sample_from_rdm`. `eval`, `sample` and `pdf` already receive `si`, so no interface
  changes are needed.
- Render the ablation ladder:

| | configuration |
|---|---|
| A0 | path-traced ground truth (`--no_neural`) |
| A1 | unconditioned neural — the baseline that fails |
| A2 | A1 + analytic `S_R` |
| A3 | A2 + geometry-conditioned `S_M` — **the method** |
| — | A0 with `--no_dispersion` — the fire control |

Raise spp well above the 8 used in earlier test frames: hero-wavelength sampling discards
three of four wavelength samples on entry, so spectral renders carry roughly 4× the
colour variance of the old RGB ones.

**Deliverable:** A3 beats A1 on matched geometry, with numbers.

### Week 3 — the generalisation study

This is the week that earns the mark. The geometry generator is fully parametric, so a
cross-cut study costs gathers, not code.

**Training family** (4 gathers): `crown_angle_deg` ∈ {30, 34.5, 37} crossed with
`num_main_facets` ∈ {6, 8}, choosing four combinations.

**Held-out test** (2 gathers, never trained on):
- `num_main_facets = 4` — the princess-like cut, a genuinely different facet topology.
- An unseen crown angle on 8 main facets — interpolation within the family.

Train `Model_M` once on the pooled training family. Then evaluate on both held-out cuts
**with no retraining**. Report per-facet correlation and RMSE against each cut's own path-
traced ground truth.

Add the two comparisons that make it an experiment rather than a demo:

- **Oracle:** a model trained directly on the held-out cut. Upper bound.
- **Naive:** the round-brilliant-only model applied to the held-out cut. Lower bound.

Transfer sitting between naive and oracle, closer to oracle, is the result. Transfer
matching naive is also publishable — it says the descriptor is insufficient and names
what is missing.

**Deliverable:** a transfer table with held-out cuts.

### Week 4 — variance, archive, freeze

- **Three seeds per configuration**, minimum. Report mean and spread. Single-run numbers
  invite the question of whether your A1/A3 separation exceeds run-to-run noise, and that
  question is fatal if you cannot answer it. This is cheap and non-negotiable.
- Archive every EXR with the exact command that produced it.
- Buffer for whatever slipped.
- **Code freeze.**

---

## 4. Month 2 — write

No new code. Figures regenerate from archived EXRs.

Figures to produce:

1. Resolution sweep: RMSE vs cell count, plateau annotated. *(claim 1)*
2. Ablation strip: A0 / A1 / A3, same crop, same exposure. *(claim 2)*
3. Per-facet scatter: ground truth vs model, A1 and A3 overlaid. *(claim 2)*
4. **Transfer table and strip: held-out cuts, naive / transfer / oracle.** *(claim 3)*
5. Fire: A0 with and without dispersion, same crop.
6. The 1.23-radii entry/exit histogram.

### Metrics

- RMSE and SSIM (or FLIP) against A0 over the stone region.
- Black-pixel fraction and p99/median contrast ratio — already tracked, and a fair proxy
  for whether the kaleidoscope survives, since aggregation flattens contrast.
- **Per-facet mean radiance correlation** — the important one, because it measures only
  what the method adds. Get the segmentation from Mitsuba's `aov` integrator:

```python
{'type': 'aov', 'aovs': 'pi:prim_index', 'integrator': {'type': 'path', 'max_depth': 24}}
```

  Segment A0 and the render under test by facet index, take mean radiance per facet, and
  report Pearson correlation across facets plus the scatter. The shape of the failure is
  more informative than the coefficient.

### Framing

Lead with the method working. For an implementation project, opening with "here is a
method that fails and here is why" reads as an account of a method that did not work.
Order: method → it works → it transfers → diagnosis as motivation → the plateau as proof
the naive alternative cannot be rescued. Same content; the difference between a project
that explains a failure and one that fixes it.

State the boundary precisely in limitations: the method restores facet-level spatial
variation; it does **not** recover the refracted see-through image, and it does **not**
recover fire — and the reason is known and stated in both cases.

---

## 5. Related work

Four of these change what the paper *is*. Verify venues and years at source.

**Kuznetsov et al., NeuMIP: Multi-Resolution Neural Materials (2021)** — the key
citation. A neural material conditioned on continuous **uv position**, with a learned
*offset* module specifically for light entering and leaving at different points: the
1.23-radii problem, named and addressed. It licenses the framing — *NeuMIP conditions on
continuous position for stochastic surfaces; we condition on facet geometry for
deterministic faceted macrostructure, and unlike a per-material fit, ours transfers
across cuts.* The 2022 follow-up on curved surfaces is worth a line.

**Dana et al., BTF (1999)** — a BTF is f(x, omega_i, omega_o). Your model is a *discrete
BTF over facets*, and saying so connects it to twenty-five years of work.

**Guy & Soler, Graphics Gems Revisited (2004)** — the gemstone rendering paper. Exploits
exactly what you diagnosed: a gem has few facets and deterministic paths, so structure is
traced rather than aggregated. Closest prior work; not citing it is a visible gap.

**Zhao et al., downsampling scattering parameters (2016)** and **Loubet & Neyret,
microflake downsampling (2018)** — *when is aggregation valid?* in the volumetric
setting. They give you vocabulary to state the claim as a principle rather than a diamond
anecdote.

Cheaper but worth having: **Walter et al. 2007** (rough-dielectric microfacet model,
already used via GGX in `eval_r`); **Wilkie et al. 2014** (hero wavelength spectral
sampling, the algorithm running in `bsdf/dispersive_dielectric.py`); **Jakob & Marschner
2012** and **Zeltner et al. 2020** (specular chains and SDS paths through faceted glass
are the canonical hard case for path tracing — your motivation, from authoritative
sources).

---

## 6. Risk register and cut list

Cut from the bottom. Everything above the line still makes a coherent paper.

| priority | item | if cut |
|---|---|---|
| 1 | Preset/geometry fix | Do not cut. Every number is suspect without it |
| 2 | Resolution sweep | Do not cut. Claim 1 becomes contestable |
| 3 | A3 on matched geometry | Do not cut. This is the method |
| 4 | Per-facet correlation | Do not cut. Cheap; measures the actual claim |
| 5 | **Cross-cut transfer** | Mark drops to low-to-mid eighties. Cut only if week 3 is lost |
| 6 | Three seeds | Report single runs and concede it in limitations |
| 7 | Oracle/naive bounds | Report transfer numbers bare; weaker but still a result |
| 8 | SSIM/FLIP | RMSE plus per-facet correlation is enough |

Known risks:

- **Descriptor insufficient for transfer.** Genuinely possible. It is still a result:
  report the gap and name what the descriptor misses. Do not hide it.
- **Six gathers too slow.** Earlier runs did 100 × 9M rays, so 20M per cut should be
  comfortable — but time one before committing to six.
- **Spectral renders too noisy at final quality.** Use `--no_dispersion` for the ablation
  ladder and keep dispersion for the dedicated fire figure only. The ladder is about
  spatial structure, so nothing is lost.
- **Model_T checkpoint mismatch.** `eval.py` degrades to analytic Fresnel with a warning
  rather than failing. Read the console output; do not assume it loaded.

---

## 7. Corrections to `FACET_CONDITIONED_BSDF.md`

Two overclaims to fix before any of it reaches the write-up:

- "total albedo is 1.0047 … **so energy is conserved and the gather is trustworthy**" —
  energy consistency is necessary, not sufficient, and says nothing about directional
  correctness. Weaken to "the gather is energy-consistent."
- "**No amount of retraining** has recovered it" — anecdotal. Delete it; the resolution
  sweep carries that claim properly.

Also fill or drop the empty `p99/median` cells in the results table, and re-run those
statistics once the geometry mismatch in §0 is fixed.
