import numpy as np
data = np.load('rdm.npz')
rdm_m = data['rdm_m']
print(f"Shape: {rdm_m.shape}")
print(f"Non-zero bins: {(rdm_m.sum(axis=-1) > 0).sum()}")
print(f"Max value: {rdm_m.max():.4f}")
print(f"Mean (non-zero): {rdm_m[rdm_m > 0].mean():.4f}")