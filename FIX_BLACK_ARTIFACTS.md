# Fixing Black Sparse Areas in Diamond Renders

## Problem Diagnosis

The rendered diamond frames show severe artifacts:
- **Large black areas**, especially at the top of the diamond
- **Sparse colored noise** (speckled regions)
- **Inconsistent shading** across facets

### Root Causes

1. **Extremely sparse training data**: 
   - RDM_M fill rate: 6.25% (only 1 in 16 direction pairs has data)
   - RDM_T fill rate: 0.21% (almost no transmission data!)
   - Grazing angles (θ_i > 70°): 0% fill rate

2. **Neural network extrapolation failure**:
   - For untrained direction pairs, the neural network outputs near-zero values
   - This creates the large black regions in renders
   - The sparse colored noise is where a few training samples exist

3. **No fallback mechanism**:
   - Original code had no fallback when neural predictions are invalid
   - Black regions occur where the network has no confidence

## Solutions Implemented

### 1. Immediate Fix: Physics-Based Fallback

**Modified**: `bsdf/neural_bsdf.py`

Added fallback to analytic dielectric BSDF when neural predictions are too low:

```python
# Check neural prediction confidence
neural_magnitude = norm(neural_value)
if neural_magnitude < confidence_threshold:
    # Blend with physics-based dielectric BSDF
    analytic_value = eval_analytic_dielectric(wi, wo)
    blend_factor = clamp(neural_magnitude / threshold, 0, 1)
    value = analytic_value * (1 - blend_factor) + neural_value * blend_factor
```

**Benefits**:
- Eliminates black regions immediately
- Provides physically plausible appearance for untrained directions
- Smooth blending between neural and analytic responses

**To enable**: Already enabled by default in `eval.py`:
```python
props['use_physics_fallback'] = True
props['confidence_threshold'] = 0.05
```

### 2. Long-Term Fix: Improved RDM Collection

**Modified**: `utils/rdm.py`, `gather_rdm.py`

Implemented stratified sampling that ensures all direction bins receive samples:

- **Old method**: Uniform sphere sampling → 0% fill for grazing angles
- **New method**: Stratified sampling → consistent coverage across all angles

**Expected improvements**:
- RDM_M fill rate: >20% (vs 6.25%)
- RDM_T fill rate: >15% (vs 0.21%)
- Grazing angles: >15% fill (vs 0%)

**Usage**:
```bash
python regenerate_rdm_stratified.py \
    --checkpoint_name my_diamond_stratified \
    --num_batches 2048 \
    --batch_size 8192
```

### 3. Retrain Neural Models

After regenerating RDM with stratified sampling:

```bash
python train_models.py --checkpoint_name my_diamond_stratified
```

The neural networks will now have:
- Much denser training data
- Better coverage of grazing angles
- More consistent predictions across all directions

## Action Plan

### Quick Fix (10 minutes)
The physics fallback is already implemented. Just re-render:

```bash
python eval.py \
    --checkpoint_dir checkpoints/my_diamond_run \
    --output_dir renders/diamond_fallback \
    --bsdf neural \
    --spp 256
```

The black areas should now show physically plausible (though not learned) diamond appearance.

### Complete Fix (several hours depending on GPU)

1. **Regenerate RDM with stratified sampling** (~1-2 hours):
   ```bash
   python regenerate_rdm_stratified.py \
       --checkpoint_name my_diamond_stratified \
       --num_batches 2048
   ```

2. **Retrain neural models** (~30-60 minutes):
   ```bash
   python train_models.py --checkpoint_name my_diamond_stratified
   ```

3. **Render with new models**:
   ```bash
   python eval.py \
       --checkpoint_dir checkpoints/my_diamond_stratified \
       --output_dir renders/diamond_stratified \
       --bsdf neural \
       --spp 256
   ```

## Expected Results

### With Physics Fallback Only
- ✓ No more black regions
- ✓ Physically plausible appearance everywhere
- ⚠ Some regions may look "generic" rather than learned diamond behavior
- ⚠ Grazing angles will show analytic dielectric, not complex diamond caustics

### With Stratified RDM + Retrained Models
- ✓ No black regions
- ✓ Learned diamond appearance across all angles
- ✓ Proper caustics and fire at grazing angles
- ✓ Consistent shading across entire diamond
- ✓ High-quality neural rendering matching ground truth

## Verification

After implementing fixes, check:

1. **Visual inspection**: No black areas in renders
2. **RDM fill rate**: `>20%` overall, `>15%` at grazing angles
3. **Neural prediction confidence**: Most directions should have neural predictions > threshold
4. **Rendering quality**: Compare with ground truth analytic renders

## Technical Details

### Why Stratified Sampling Works

Uniform sphere sampling gives probability `p(ω) ∝ sin(θ)`, so:
- Near-normal angles (θ≈0°): `sin(0°) ≈ 0` → very few samples
- Grazing angles (θ≈90°): `sin(90°) ≈ 1` → many samples

But after dividing by sample count in histogram, grazing angles are **undervalued**!

Stratified sampling divides angular space into bins and samples uniformly within each:
- All θ_i bins get equal number of samples
- Histogram division by count doesn't bias against any angle
- Result: uniform coverage across all directions

### Physics Fallback Details

The analytic dielectric fallback uses:
```python
# Fresnel reflection coefficient
F = fresnel(cos_theta_i, eta=n1/n2)

# Reflection or transmission based on direction
same_side = (cos_theta_i * cos_theta_o) > 0
value = F if same_side else (1 - F)

# Scale by cosine term
value *= abs(cos_theta_o)
```

This provides a reasonable approximation but lacks:
- Multiple internal reflections (diamond fire)
- Path-dependent caustics
- Dispersion effects

Neural models trained on complete RDM data capture these effects.

## Files Modified

1. `bsdf/neural_bsdf.py` - Added physics fallback and blending
2. `utils/rdm.py` - Added stratified sampling methods
3. `gather_rdm.py` - Added `--sampling_method` parameter
4. `eval.py` - Enabled fallback by default
5. `regenerate_rdm_stratified.py` - NEW: Script to regenerate RDM

## References

- Original RDM collection: `gather_rdm.py`
- Sampling analysis: `sampling_comparison.png`
- Improved sampling docs: `IMPROVED_RDM_SAMPLING.md`