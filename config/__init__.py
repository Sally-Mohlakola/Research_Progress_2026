MITSUBA_PATH = './dependencies/mitsuba3/build/python'
DRJIT_LIBLLVM_PATH = '/usr/lib/llvm-14/lib/libLLVM.so'
# Spectral, so that wavelength-dependent refraction (dispersion) can produce
# a diamond's fire. Note there is no `llvm_spectral`: the LLVM spectral
# variant Mitsuba ships is `llvm_ad_spectral`. Run
#   python -c "import mitsuba as mi; print(mi.variants())"
# to see what this build actually offers.
VARIANT = 'llvm_ad_spectral'
#VARIANT = 'llvm_ad_rgb'      # achromatic; no fire is possible in RGB
#VARIANT = 'cuda_ad_spectral'
DEVICE = 'cuda' if VARIANT.startswith('cuda') else 'cpu'

# Aliases
variant = VARIANT
device = DEVICE

import sys
sys.path.append(MITSUBA_PATH)

import os
os.environ["DRJIT_LIBLLVM_PATH"] = DRJIT_LIBLLVM_PATH