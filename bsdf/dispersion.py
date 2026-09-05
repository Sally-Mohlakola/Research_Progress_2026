"""
dispersion.py - wavelength-dependent IOR for diamond, plus the small set of
spectral helpers the rest of the codebase needs to work under both
`llvm_ad_rgb` and `llvm_ad_spectral`.

Why this exists
---------------
A diamond's "fire" -- the flashes of spectral colour -- is *dispersion*: the
refractive index varies with wavelength, so different wavelengths refract
through different angles and are spatially separated by the time they leave
the stone. An RGB renderer cannot produce it at all, because RGB has no
wavelength axis to disperse along; the three channels refract identically and
recombine to white. Fire therefore requires a spectral variant.

Mitsuba's stock `dielectric` BSDF takes a single scalar `int_ior`, so it is
achromatic no matter which variant it runs in. `dispersive_dielectric.py`
supplies the replacement; this module supplies the physics it needs.

Sellmeier coefficients
----------------------
Two-term Sellmeier fit for diamond (Peter 1923), with lambda in micrometres:

    n^2 - 1 = 0.3306 l^2 / (l^2 - 0.175^2) + 4.3356 l^2 / (l^2 - 0.106^2)

Verified against published refractive indices:

    430.8 nm  2.45172  (ref 2.4515)      589.3 nm  2.41726  (ref 2.4175)
    486.1 nm  2.43555  (ref 2.4354)      656.3 nm  2.40990  (ref 2.4099)
    686.7 nm  2.40728  (ref 2.4073)

giving n(430.8) - n(686.7) = 0.0444 against the gemmological dispersion
figure of 0.044 for diamond. That number is the whole reason diamond shows
more fire than almost any other colourless stone -- for comparison, quartz is
0.013 -- so it is worth keeping the fit rather than approximating it.
"""

import mitsuba as mi
import drjit as dr
from config import variant

# Respect a variant the caller has already chosen (ground_truth/diamond.py
# runs under scalar_spectral) instead of forcing the configured one.
if mi.variant() is None:
    mi.set_variant(variant)

# True when the active variant carries wavelengths (llvm_ad_spectral and
# friends). Everything dispersion-related is a no-op when this is False, so
# the same source keeps working under llvm_ad_rgb.
IS_SPECTRAL = 'spectral' in mi.variant()

# 4 in the spectral variants, 3 in the RGB ones.
N_SPECTRUM = dr.size_v(mi.UnpolarizedSpectrum)

_B1, _C1 = 0.3306, 0.175 ** 2
_B2, _C2 = 4.3356, 0.106 ** 2

# Sodium D line: the wavelength at which "the" refractive index of a material
# is conventionally quoted, and the value the achromatic pipeline uses.
REFERENCE_WAVELENGTH = 589.3


def diamond_ior(wavelength_nm):
    """
    Refractive index of diamond at the given wavelength(s) in nanometres.

    Accepts a Python float, a Dr.Jit Float, or an UnpolarizedSpectrum of
    wavelengths, and returns a matching type.

    The lower clamp keeps the expression away from the Sellmeier pole at
    175 nm. Mitsuba samples 360-830 nm so this never binds in practice; it
    exists so that a stray or uninitialised wavelength cannot produce a NaN
    that then propagates silently through a whole render.
    """
    lam = dr.maximum(wavelength_nm, 200.0) * 1e-3
    l2 = lam * lam
    n2 = 1.0 + _B1 * l2 / (l2 - _C1) + _B2 * l2 / (l2 - _C2)
    return dr.sqrt(dr.maximum(n2, 1.0))


# n at the sodium D line -- 2.41726, which is where the 2.419 used elsewhere
# in this project comes from.
REFERENCE_IOR = float(diamond_ior(REFERENCE_WAVELENGTH))


def hero_wavelength(si):
    """
    The wavelength a dispersive path follows.

    Mitsuba carries MI_WAVELENGTH_SAMPLES (4) wavelengths per ray. A
    dispersive refraction would send each of them in a different direction,
    but a ray has one direction, so exactly one wavelength can survive the
    event. Channel 0 is the "hero": Mitsuba's `sample_shifted` draws it
    uniformly over the full spectral range and derives the other three by
    rotation, so channel 0 on its own is an unbiased uniform sample of the
    spectrum -- which is what makes it valid to keep it and drop the rest.
    """
    return si.wavelengths[0]


def hero_eta(si, int_ior, ext_ior, dispersion=True):
    """
    Relative IOR to use for a refraction event, as a plain Float.

    Under dispersion this is n(hero wavelength) / ext_ior, so each ray bends
    by its own wavelength's amount and the spectrum fans out across the
    stone. Otherwise it is the fixed achromatic ratio.
    """
    if IS_SPECTRAL and dispersion:
        return diamond_ior(hero_wavelength(si)) / ext_ior
    return mi.Float(int_ior / ext_ior)


def hero_collapse_weight():
    """
    Spectral weight that keeps only the hero wavelength: [N, 0, 0, 0].

    Zeroing the other three channels is what actually separates the colours
    -- without it every wavelength would be forced along the hero's path and
    the render would come back white. The factor N compensates for the three
    dropped channels: Mitsuba's spectrum-to-XYZ conversion averages over the
    N channels (it carries a 1/N), so a single surviving channel would
    otherwise land at 1/N of the correct brightness.

    Apply this ONCE per path, on the refraction that enters the stone.
    Applying it again on the way out would multiply by N a second time.
    """
    w = mi.UnpolarizedSpectrum(0.0)
    w[0] = float(N_SPECTRUM)
    return w


def fresnel_spectral(cos_theta, si, int_ior, ext_ior, dispersion=True):
    """
    Fresnel reflectance evaluated per wavelength, as an UnpolarizedSpectrum.

    `mi.fresnel` only accepts a scalar eta, so the channels are filled in a
    Python loop -- N is 4, it unrolls at trace time, and the cost is
    negligible. Used for the *reflection* lobe, which needs no hero collapse
    because reflection direction is wavelength-independent; only the
    reflectance differs, and for diamond that difference is small (F varies
    about 1.7% across the visible at normal incidence). Keeping it exact
    anyway costs nothing and avoids a needless approximation.
    """
    if not (IS_SPECTRAL and dispersion):
        F, _, _, _ = mi.fresnel(cos_theta, mi.Float(int_ior / ext_ior))
        return mi.UnpolarizedSpectrum(F)

    out = mi.UnpolarizedSpectrum(0.0)
    for c in range(N_SPECTRUM):
        eta_c = diamond_ior(si.wavelengths[c]) / ext_ior
        F_c, _, _, _ = mi.fresnel(cos_theta, eta_c)
        out[c] = F_c
    return out


def rgb_to_spectrum(rgb, wavelengths):
    """
    Lift an RGB value to a spectrum via Mitsuba's sRGB upsampling.

    Needed because Model_M was trained to emit three numbers, and under a
    spectral variant a BSDF must return N wavelength samples. `srgb_model_*`
    expects a reflectance in [0, 1], so the value is split into a scalar
    magnitude and a unit-range chroma, only the chroma is upsampled, and the
    magnitude is multiplied back afterwards. Model_M's output is a radiance
    ratio that routinely exceeds 1, so skipping that split would silently
    clip every bright sample.

    Note this gives the neural term a *colour*, not fire: the RDM it was
    fitted to has no wavelength axis, so the same spectrum is returned
    whatever wavelength the ray carries. See FACET_CONDITIONED_BSDF.md.
    """
    if not IS_SPECTRAL:
        return rgb

    scale = dr.maximum(dr.maximum(rgb.x, rgb.y), rgb.z)
    unit = dr.clip(rgb / dr.maximum(scale, 1e-8), 0.0, 1.0)
    coeff = mi.srgb_model_fetch(unit)
    return mi.srgb_model_eval(coeff, wavelengths) * scale


def to_rgb3(value):
    """
    Reduce a throughput value to three channels for RDM storage.

    The RDM has no wavelength axis and the gather scene uses a plain
    achromatic `dielectric`, so every channel of the gathered throughput
    holds the same number and taking channel 0 is exact rather than an
    approximation. Doing it this way keeps the measured albedo at 1.0 under
    both variants.

    This is also the seam where a wavelength axis would be added: gathering
    the RDM per wavelength is what a spectrally-aware neural term would
    need.
    """
    if not IS_SPECTRAL:
        return mi.Color3f(value.x, value.y, value.z)
    v = value[0]
    return mi.Color3f(v, v, v)
