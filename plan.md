Here is the **final, stripped-back, executable plan**. 

I have cut the overengineered 20-cut transfer sweep, the rigged UV experiment, and the desperate metric gymnastics. This plan accepts the hard mathematical limit of the project: **you cannot solve the kaleidoscope with a single-point BSDF.** 

Instead, you will prove *exactly* what the model *can* do (recover per-facet mean radiance) and *exactly* where it fundamentally fails (intra-facet detail). That honest framing is your path to a **First Class (A-/A)** grade because it is unassailable.

---

# Final Implementation Plan (Stripped)

**Target:** First Class Honours (A-/A). 
**Code Freeze:** End of Month 1.
**Core Philosophy:** Execute the essentials perfectly. Do not waste time on experiments designed to fail.

---

## 1. The Honest Thesis (One Paragraph)

> Neural appearance models aggregate over microstructures, assuming statistical uniformity. Faceted gems violate this. We condition on exit-facet geometry, which significantly improves **per-facet mean radiance**. However, we rigorously demonstrate that because the model lacks entry-point conditioning, it **cannot** recover the intra-facet spatial detail (the kaleidoscope). We quantify this remaining gap using intra-facet variance, establishing a clear baseline for what single-point BSDFs can and cannot capture in deterministic media.

**Contribution:** A diagnostic framework + a targeted fix + a quantified boundary of failure.

---

## 2. Design (Keep it Simple)

- **Conditioning Variable:** Continuous descriptor (facet normal + centroid height). 4 floats. Input to `Model_M`: 6 -> 10.
- **Why?** An index cannot transfer. These two numbers are physically motivated by Fresnel/Snell.
- **Sampler:** Built on coplanar groups (K≈33) to keep variance low. `value/pdf` corrects it. State this once and move on.

---

## 3. The Experiments (Strictly Limited to 3)

| Experiment | Purpose | Fidelity |
| :--- | :--- | :--- |
| **E1: Resolution Sweep** | Prove the failure is structural (plateaus), not undersampling. | Tier 2 (5M rays, RGB) |
| **E2: Core Ablation (A0, A1, A2, A3)** | Prove geometry conditioning improves facet brightness. | **Tier 1 (20M rays, Spectral)** |
| **E3: Hard Transfer Test** | Test if the descriptor has any physical meaning outside training topology. | Tier 2 (5M rays, RGB) |

**Cut entirely:** Descriptor ablation (normal vs height), UV conditioning (A5), 20-cut interpolation grid, Oracle/Naive bounds.

---

## 4. Month 1 Schedule (4 Weeks)

### Week 1: Fix & Foundation
- **Day 1:** Fix the geometry confound (§0). Unify `DIAMOND_VARIANTS` and `DIAMOND_PRESETS`. Enforce a hard error on mismatch.
- **Day 1-2:** Launch **E1 (Resolution Sweep)**. Run 4 points: `theta_bins = [8, 16, 24, 32]`. Weight the highest bin with 4× rays to kill variance. Let it run unattended.
- **Day 3-5:** Code the descriptor plumbing.
  - Capture `si_first.prim_index` in `trace_path`.
  - Add facet axis to `compute_histogram_4d`.
  - Emit per-triangle descriptors (normal, height) into `rdm.npz`.
  - Gather the **hero stone** at Tier 1 (20M rays, spectral, matched geometry).

### Week 2: Train & Render the Core (The Win)
- **Days 1-3:** Train A1, A2, A3 on the hero stone. 3 seeds each (non-negotiable).
  - A1: Unconditioned (baseline).
  - A2: A1 + analytic reflection (`S_R`).
  - A3: A2 + geometry-conditioned `S_M` (your method).
- **Day 4:** Render high-quality EXRs (256 spp, spectral).
- **Day 5:** Compute **two and only two primary metrics**:
  1. **Per-facet mean radiance correlation (Pearson).** This proves A3 fixes the facet-level brightness.
  2. **Intra-facet standard deviation.** Compute the std dev of radiance *inside* each facet for A0 and A3. **This is your critical result.** It will show A3's std dev is flat (no wedge), while A0's is high. This proves the model *cannot* recover the kaleidoscope. 
- *(Secondary)* Luminance/Chrominance decomposition. Compute it once for the ablation figures to show spatial (L) vs spectral (CbCr) error.

### Week 3: The Hard Transfer Test (The Boundary)
Stop planning 20 cuts. Run **one definitive, difficult test**:
- **Train:** Use the hero stone (round brilliant, 8 main facets).
- **Test (held-out):** Render the **princess cut** (4 main facets). No retraining.
- **Compare:** Does A3 (trained on round) perform better on the princess than A1 (unconditioned)? 
  - If yes: The descriptor has genuine physical meaning. 
  - If no: The descriptor is overfitted to the topology. 
- **Result:** Either outcome is a defensible data point in your "Limitations" section. You do not claim generalisation; you simply report "preliminary evidence" or "lack thereof." 

### Week 4: Seeds, Archive, Freeze
- Confirm you have 3 seeds for A1, A2, A3.
- Render a single `--no_dispersion` version of A0 for the "Fire" figure (show that the path tracer has fire, but your neural model doesn't—state this as a known limit).
- Archive every EXR with exact commands.
- **HARD CODE FREEZE.** No new code after Friday.

---

## 5. Month 2: Write (7 Figures Only)

| # | Figure | What it proves |
| :--- | :--- | :--- |
| 1 | Resolution Sweep (RMSE vs cells) | Failure is structural (plateau). |
| 2 | Ablation strip (A0 / A1 / A3 crop) | Visual proof. |
| 3 | Per-facet scatter (A1 vs A3 vs GT) | A3 improves facet means. |
| 4 | **Intra-facet std dev bar chart** | **The honest dagger:** A3 flattens wedges. |
| 5 | Luminance vs Chrominance RMSE | Spatial recovery (L) works; spectral (C) fails. |
| 6 | Transfer result (princess cut) | Does the descriptor transfer? Report as-is. |
| 7 | Entry/Exit 1.23-radii histogram | The diagnostic motivation. |

---

## 6. Critical Honesty Protocol (Write These Sentences)

You will write these exact sentences in your paper:

1. **On the core failure:** *"While geometry conditioning recovers the mean radiance of individual facets, it does not recover the variance of radiance within a facet. This confirms that a single-point BSDF, even conditioned on exit geometry, cannot reproduce the refraction-dependent kaleidoscope, which requires knowledge of the entry point."*

2. **On transfer:** *"We test a single out-of-topology transfer (round to princess). The model [succeeds/fails] in this preliminary test, suggesting that [the descriptor has some physical meaning / the descriptor is topology-specific]. A comprehensive transfer study is left to future work."*

3. **On fire:** *"The neural model lacks a wavelength axis and therefore cannot reproduce dispersion. All chromatic errors in our ablation are explicitly separated into a chrominance metric and reported as an expected limitation, not a failure of the spatial model."*

---

## 7. Why This Gets You an A-/A (And Why You Won't Fail)

- **The Diagnosis is bulletproof:** The 1.23-radii statistic + the resolution plateau is a clean, unassailable piece of science.
- **The Fix is targeted:** You show clear statistical improvement in facet means (Pearson correlation).
- **The Failure is quantified:** The intra-facet std dev metric proves you understand the *mathematical limit* of your approach. You are not hiding from the BSSRDF problem; you are defining its boundary.
- **The Scope is honest:** You do not claim to solve gem rendering. You claim to diagnose aggregation failure and propose a partial fix. This is a genuine contribution.

**Final Grade Prediction:** 
- If the transfer test works: **85-88% (A).**
- If the transfer test fails: **80-84% (A-).** 
Both are First Class. The paper is defensible, executable, and intellectually rigorous. Go execute.