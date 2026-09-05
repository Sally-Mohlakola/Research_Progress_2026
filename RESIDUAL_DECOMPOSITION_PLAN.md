# Bounce-Count Residual Decomposition for Neural Gemstone Appearance

**Target Grade:** First Class (70-85%)  
**Core Claim:** Pure RDM-based surface aggregation systematically under-represents non-local specular paths. We propose bounce-count residual decomposition that handles short deterministic paths analytically and learns only the high-bounce residual.

**Timeline:** 3-4 weeks to complete implementation + write-up

---

## The Thesis Statement

> We analyse the conditions under which neural appearance aggregation of internal gemstone transport succeeds or fails, showing that pure RDM-based surface aggregation systematically under-represents non-local specular paths (median 1.23 radii exit distance). This motivates a bounce-count residual decomposition that evaluates short deterministic paths analytically and learns only the residual high-bounce contribution. We validate the method across polished and frosted diamonds, showing that the decomposition recovers [X]% of missing specular structure while retaining [Y]% neural efficiency for diffuse-like transport.

**Contribution:** A principled decomposition with empirical validation of the boundary between deterministic (analytic) and stochastic (neural) transport regimes.

---

## Why This Gets First Class

1. **Novel method:** Bounce-count decomposition is not in prior work
2. **Systematic analysis:** You prove *why* aggregation fails (non-local paths)
3. **Principled solution:** Not a patch—a theoretically motivated split
4. **Empirical validation:** Works on the actual failure case (polished gems)
5. **Honest scope:** States what neural *should* do (high-bounce residual)
6. **Generalizable:** Applies beyond gems (any deterministic specular media)

---

## Three-Week Implementation Plan

### Week 1: Diagnostic Study (Foundation)

**Goal:** Prove that pure aggregation fails systematically for non-local paths

#### Day 1: Fix Geometry Confound
```bash
# Unify DIAMOND_PRESETS and DIAMOND_VARIANTS
# Add geometry assertion to eval.py
# One table, one source of truth
```

#### Days 2-3: Path Length Analysis
Modify `gather_rdm.py` to record:
```python
# For each gathered path:
- entry_point (x, y, z)
- exit_point (x, y, z)  
- internal_bounce_count (k)
- path_length
- entry_facet, exit_facet

# Compute and save:
entry_exit_distance = ||exit - entry|| / bounding_radius
```

**Gather one instrumented RDM:**
```bash
python gather_rdm.py --checkpoint_name diagnostic_run \
    --diamond_name round_diamond_gia \
    --batch_size 1000000 --num_batches 20 \
    --theta_bins 8 --phi_bins 16 --max_depth 64 \
    --save_path_stats  # NEW FLAG
```

**Generate diagnostic plots:**
1. Entry-exit distance histogram (the 1.23-radii result)
2. Distance vs bounce count scatter
3. Per-facet-pair distance distribution

**Deliverable:** Quantitative proof that paths are non-local, with clear plots

#### Days 4-5: Bounce-Count Stratification
Analyze the gathered RDM by bounce count:
```python
# Split rdm_m by bounce count:
rdm_m_k0 = paths with k=0 bounces (direct transmission)
rdm_m_k1 = paths with k=1 bounce
rdm_m_k2 = paths with k=2 bounces
rdm_m_k3plus = paths with k≥3 bounces

# Compute for each:
- Energy contribution (% of total)
- Spatial variance (std dev of exit positions)
- Directional variance (entropy of outgoing distribution)
```

**Key insight to measure:**
> "Low-bounce paths (k≤2) are deterministic (low entropy, concentrated energy). High-bounce paths (k≥3) are diffuse-like (high entropy, smooth distribution). The neural model should handle the latter, not the former."

**Deliverable:** A table showing:
| Bounce Count | Energy % | Exit Spread | Dir Entropy | Suitable for Neural? |
|--------------|----------|-------------|-------------|---------------------|
| k=0-1        | X%       | Y radii     | Z bits      | ❌ (deterministic)   |
| k=2          | X%       | Y radii     | Z bits      | ? (boundary)         |
| k≥3          | X%       | Y radii     | Z bits      | ✅ (stochastic)      |

#### Days 6-7: Choose Split Point k*
Based on the table, choose k* where:
- k ≤ k*: analytic rendering
- k > k*: neural aggregation

**Likely k* = 2 or 3** (short specular chains vs diffuse tail)

Write validation: ground truth render with `--max_depth k*+1` should show most visible structure

**Deliverable:** Justified choice of k* with supporting data

---

### Week 2: Residual Decomposition Implementation

**Goal:** Build the hybrid renderer

#### Days 8-9: Residual RDM Gather
```bash
# Gather conditioned on bounce count > k*
python gather_rdm.py --checkpoint_name residual_k3plus \
    --diamond_name round_diamond_gia \
    --batch_size 1000000 --num_batches 20 \
    --theta_bins 8 --phi_bins 16 --max_depth 64 \
    --min_internal_bounces 3  # NEW: Only learn k≥3 transport
```

**Critical:** This RDM must exclude k≤2 paths, otherwise double-counting

Train Model_M on residual only:
```bash
python train_models.py --checkpoint_name residual_k3plus \
    --epochs_m 5000 --lr_m 0.001 --batch_size 4096
```

#### Days 10-11: Hybrid BSDF Implementation

Create `bsdf/hybrid_residual.py`:
```python
class HybridResidualBSDF:
    """
    Decomposed appearance:
      L = L_analytic(k ≤ k*) + L_neural(k > k*)
    """
    
    def __init__(self, k_star=2):
        self.k_star = k_star
        self.analytic = DispersiveDielectric(...)
        self.neural_residual = NeuralDiamond(...)  # trained on k>k* only
        
    def eval(self, ctx, si, wo):
        depth = ctx.depth  # from BSDFContext or custom integrator
        
        if depth <= self.k_star:
            # Short paths: analytic specular
            return self.analytic.eval(ctx, si, wo)
        else:
            # Long paths: neural residual
            return self.neural_residual.eval(ctx, si, wo)
    
    def sample(self, ctx, si, sample1):
        depth = ctx.depth
        
        if depth <= self.k_star:
            # Sample analytic (Fresnel + Snell)
            return self.analytic.sample(ctx, si, sample1)
        else:
            # Sample from neural RDM
            return self.neural_residual.sample(ctx, si, sample1)
```

**Implementation note:** If `ctx.depth` not available, use two-pass rendering (fallback):
- Pass 1: `--max_depth 3` with analytic only
- Pass 2: neural residual with modified integrator
- Composite: `L = L_pass1 + L_pass2`

#### Days 12-13: Validation Renders

Render ablation ladder (all on matched geometry):
```bash
# A0: Ground truth (full path tracing)
python eval.py --checkpoint_name none \
    --diamond_name round_diamond_gia --no_neural \
    --spp 256 --output renders/A0_ground_truth

# A1: Pure neural (baseline that fails)
python eval.py --checkpoint_name diagnostic_run \
    --diamond_name round_diamond_gia \
    --spp 256 --output renders/A1_pure_neural

# A2: Pure analytic (per-facet Fresnel, no internal multi-scatter)
python eval.py --checkpoint_name none \
    --diamond_name round_diamond_gia --analytic_only --max_depth 3 \
    --spp 256 --output renders/A2_analytic_k3

# A3: Hybrid residual (your method)
python eval.py --checkpoint_name residual_k3plus \
    --diamond_name round_diamond_gia --hybrid_residual --k_star 2 \
    --spp 256 --output renders/A3_hybrid_residual
```

#### Day 14: Compute Metrics

For each render, compute:
```python
metrics = {
    # Energy distribution
    'stone_p99_over_median': ...,
    'top_1pct_energy_share': ...,
    'mean_over_median': ...,
    
    # Spatial structure
    'per_pixel_correlation_vs_GT': ...,
    'lowpass_8px_correlation': ...,
    'lowpass_16px_correlation': ...,
    
    # Per-facet statistics
    'per_facet_mean_correlation': ...,
    'intra_facet_variance': ...,
    
    # Black pixels
    'black_pixel_fraction': ...,
}
```

**Target results:**
- A1 (pure neural): r_pixel ≈ 0.01, p99/med ≈ 15, top-1% ≈ 9%
- A2 (analytic only): recovers specular but missing diffuse glow
- **A3 (hybrid)**: r_pixel > 0.4, p99/med > 25, top-1% > 20%

**Deliverable:** Metrics table comparing all ablations

---

### Week 3: Validation & Write-Up

#### Days 15-16: Frosted Diamond Extension

**Goal:** Show method adapts across determinism spectrum

Add roughness to geometry:
```python
# In brilliant_geometry.py or as post-process:
def add_surface_roughness(vertices, normals, roughness=0.1):
    """Perturb normals to simulate frosting"""
    noise = np.random.randn(*normals.shape) * roughness
    normals_rough = normals + noise
    normals_rough /= np.linalg.norm(normals_rough, axis=1, keepdims=True)
    return normals_rough
```

Gather + render for:
- Polished (roughness=0.0) - already done
- Lightly frosted (roughness=0.05)
- Heavily frosted (roughness=0.15)

**Hypothesis:** As roughness increases:
- Paths become more stochastic (higher entropy)
- Neural-only performance improves (aggregation more valid)
- Hybrid advantage decreases (less deterministic structure)

**Deliverable:** Table showing method performance vs roughness

#### Days 17-18: Generate Figures

**Figure 1: The Diagnostic** (2 panels)
- (a) Entry-exit distance histogram (1.23-radii median)
- (b) Distance vs bounce count scatter
- Caption: "Pure aggregation assumes local transport. Internal specular paths are non-local."

**Figure 2: Bounce-Count Stratification** (bar chart)
- X-axis: bounce count (0, 1, 2, 3, 4+)
- Y-axis: energy contribution (%)
- Color: directional entropy
- Caption: "Low-bounce paths dominate energy and are deterministic (low entropy)."

**Figure 3: Ablation Strip** (image grid, 4 columns)
- A0 (GT) | A1 (pure neural) | A2 (analytic only) | A3 (hybrid)
- Same crop, exposure
- Caption: "Hybrid decomposition recovers specular structure."

**Figure 4: Metrics Spider Plot** (radar chart)
- Axes: per-pixel correlation, p99/median, top-1% energy, mean/median, black px (inverted)
- Lines: A1, A2, A3, A0 (reference circle)
- Caption: "Hybrid achieves balanced performance across metrics."

**Figure 5: Per-Facet Analysis** (scatter + heatmap)
- (a) Scatter: GT facet mean vs predicted (A1, A3 overlaid)
- (b) Heatmap: residual by facet for A1 and A3
- Caption: "Decomposition reduces per-facet error."

**Figure 6: Roughness Study** (line plot)
- X-axis: surface roughness
- Y-axis: correlation / energy metrics
- Lines: pure neural, analytic, hybrid
- Caption: "Hybrid advantage decreases as transport becomes stochastic."

**Figure 7: Energy Distribution** (histogram overlay)
- Per-pixel radiance distributions for A0, A1, A3
- Log scale
- Caption: "Hybrid recovers high-radiance specular peaks."

#### Days 19-21: Write The Paper

**Structure:**

**1. Introduction (800 words)**
- Neural aggregation successful for stochastic materials (cloth)
- Gems have deterministic specular paths → aggregation should fail
- Contribution: diagnose failure, propose principled decomposition
- Preview results: hybrid recovers X% of structure

**2. Background (600 words)**
- Neural appearance aggregation (Soh & Montazeri 2024)
- Gemstone rendering (Guy & Soler 2004)
- When is aggregation valid? (Loubet & Neyret 2018)

**3. Analysis: Why Pure Aggregation Fails (1200 words)**
- 3.1 Path non-locality (1.23-radii measurement, Figure 1)
- 3.2 Bounce-count stratification (Table, Figure 2)
- 3.3 Deterministic vs stochastic regimes (entropy analysis)
- **Key result:** Low-bounce paths are deterministic, cannot be aggregated

**4. Method: Bounce-Count Residual Decomposition (1000 words)**
- 4.1 Choosing the split point k*
- 4.2 Residual RDM gathering (conditioned on k > k*)
- 4.3 Hybrid rendering (analytic + neural)
- 4.4 Implementation (two-pass or depth-aware BSDF)

**5. Validation (1500 words)**
- 5.1 Experimental setup (matched geometry, spp, etc)
- 5.2 Ablation study (A0-A3, Figure 3, metrics Table)
- 5.3 Per-facet analysis (Figure 5)
- 5.4 Roughness study (Figure 6)
- **Key result:** Hybrid achieves r>0.4 vs pure neural r≈0.01

**6. Discussion (600 words)**
- What the decomposition recovers (specular structure)
- What it doesn't (wavelength-dependent fire, full BSSRDF)
- Comparison to Guy & Soler (analytic precomputation)
- When to use neural vs analytic

**7. Limitations (400 words)**
- Requires choosing k* (though principled)
- Wavelength axis not included
- Entry-point independence (still single-point BSDF)
- Two-pass rendering has compositing artifacts

**8. Conclusion (300 words)**
- Established conditions for aggregation failure
- Proposed principled decomposition
- Future: entry-exit conditioning, wavelength axis

**Total: ~6400 words + figures**

---

## Fallback Options (If Time Runs Short)

### If Week 3 gets compressed:

**Cut:**
- Frosted diamond study (Figure 6)
- Spider plot (Figure 4)

**Keep:**
- Core ablation (Figures 1-3, 5)
- Diagnostic analysis (Section 3)
- Method & validation (Sections 4-5)

**Grade impact:** 75-80% (still solid First, less comprehensive)

### If Week 2 implementation blocked:

**Fallback to two-pass rendering:**
Instead of depth-aware BSDF, render:
- Pass 1: analytic with `max_depth=3`
- Pass 2: neural residual (gather conditioned on k≥3)
- Composite: `L_final = L_pass1 + L_pass2`

This is scientifically valid (proves the decomposition works) even if not a production renderer.

**Grade impact:** 72-78% (method proven, implementation simplified)

---

## Risk Register

| Risk | Likelihood | Mitigation | Impact if occurs |
|------|------------|------------|------------------|
| Cannot access ctx.depth | Medium | Two-pass rendering fallback | -3% (still works) |
| k=3 split doesn't work | Low | Try k=2 or k=4, report best | -2% (adaptive) |
| Residual RDM too sparse | Medium | Increase ray count for k≥3 | -0% (longer gather) |
| Frosted diamond expensive | High | Cut from plan, focus on core | -5% (less generalization) |
| Writing takes longer | High | Start early, cut figures | -0 to -10% (quality) |

**Most likely issue:** Two-pass compositing instead of depth-aware BSDF. **This is fine**—proves the concept.

---

## Expected Grade Breakdown

**If executed fully:**
- Technical implementation (30%): 26/30 (novel hybrid method)
- Experimental design (20%): 18/20 (systematic, principled)
- Analysis & results (25%): 22/25 (comprehensive validation)
- Understanding (15%): 14/15 (deep insight into failure mode)
- Presentation (10%): 8/10 (clear figures and writing)

**Total: 88/100 = 88% (strong First)**

**If compressed (2.5 weeks, cuts roughness study):**
**Total: 78-82% (solid First)**

**If fallback (two-pass only, no depth-aware BSDF):**
**Total: 75-78% (First, less polished)**

---

## Why This Specific Plan Gets First Class

1. **Clear contribution:** Not "fixing" aggregation, but decomposing it principled
2. **Systematic validation:** Ablation study with metrics, not just eyeballing
3. **Honest scope:** States what neural should/shouldn't do
4. **Generalizable insight:** Deterministic vs stochastic regime applies broadly
5. **Publication-quality:** Could submit to EGSR or CGF with polish

**The key:** You're not claiming neural rendering "works for gems." You're claiming **principled decomposition works, and here's the boundary between regimes.**

That's a defensible, novel contribution.

---

## Timeline Summary

| Week | Focus | Deliverables | Critical Path |
|------|-------|--------------|---------------|
| 1 | Diagnostic analysis | Path statistics, bounce stratification, k* choice | Instrumented gather |
| 2 | Implementation | Residual RDM, hybrid BSDF, ablation renders | Model training |
| 3 | Validation & writing | Figures, metrics, 6400-word paper | Writing quality |

**Code freeze:** End of Day 18 (leave 3 days pure writing)

**First complete draft:** Day 21

**Final submission:** After advisor feedback

---

## First Steps (Start Today)

1. **Read and commit to this plan**
2. **Fix geometry confound** (3 hours)
3. **Modify gather_rdm.py to save path statistics** (4 hours)
4. **Launch overnight gather with `--save_path_stats`** (ready tomorrow)
5. **Start writing Section 3.1** (why aggregation fails) while gather runs

You can execute Steps 2-5 **today** and be on track for 80%+ by week 3.

**Do you want me to help implement the path statistics gathering code?**
