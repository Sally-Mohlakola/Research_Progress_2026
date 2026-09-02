"""
dispersive_dielectric.py - a smooth dielectric whose refractive index varies
with wavelength, so that a diamond rendered with it shows fire.

This is a drop-in replacement for Mitsuba's `dielectric`. That plugin takes
`int_ior` as a single scalar and is therefore achromatic in every variant,
including the spectral ones -- which is why switching to `llvm_ad_spectral`
on its own changes nothing visible. Dispersion has to come from a BSDF that
actually reads `si.wavelengths`.

Behaviour matches `dielectric` exactly (delta reflection + delta refraction,
Fresnel split, radiance-mode eta^2 scaling) apart from two things:

  1. eta is n(lambda) / ext_ior from the Sellmeier fit in dispersion.py,
     evaluated at the ray's hero wavelength, so blue bends more than red.
  2. The refraction that enters the stone collapses the spectrum to the hero
     wavelength (see dispersion.hero_collapse_weight). Reflection does not
     collapse, since its direction does not depend on wavelength.

Under an RGB variant, or with `dispersion` set to false, both of those
degrade to the achromatic behaviour and this BSDF is equivalent to
`dielectric` at the sodium D line.
"""

import mitsuba as mi
import drjit as dr
from config import variant

# Respect a variant the caller has already chosen (ground_truth/diamond.py
# runs under scalar_spectral) instead of forcing the configured one.
if mi.variant() is None:
    mi.set_variant(variant)

from bsdf.dispersion import (
    IS_SPECTRAL, REFERENCE_IOR, hero_eta, hero_collapse_weight,
)


class DispersiveDielectric(mi.BSDF):

    def __init__(self, props):
        mi.BSDF.__init__(self, props)

        self.ext_ior = props.get('ext_ior', 1.000277)
        self.int_ior = props.get('int_ior', REFERENCE_IOR)

        # Lets a render be run with dispersion off while everything else
        # stays identical -- the control condition for any "does the fire
        # come from dispersion?" comparison.
        self.dispersion = bool(props.get('dispersion', True))

        reflection = (mi.BSDFFlags.DeltaReflection |
                      mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide)
        transmission = (mi.BSDFFlags.DeltaTransmission |
                        mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide)
        self.m_components = [reflection, transmission]
        self.m_flags = reflection | transmission

    def sample(self, ctx, si, sample1, sample2, active):
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)

        eta = hero_eta(si, self.int_ior, self.ext_ior, self.dispersion)
        F, cos_theta_t, eta_it, eta_ti = mi.fresnel(cos_theta_i, eta)
        T = dr.maximum(1.0 - F, 0.0)

        # Written as two independent comparisons rather than `~selected_r`:
        # under the scalar variants a Dr.Jit mask is a plain Python bool, and
        # `~True` there is the integer -2, not False.
        selected_r = active & (sample1 <= F)
        selected_t = active & (sample1 > F)

        bs = mi.BSDFSample3f()
        bs.pdf = dr.select(selected_r, F, T)
        bs.sampled_component = dr.select(selected_r, mi.UInt32(0), mi.UInt32(1))
        bs.sampled_type = dr.select(
            selected_r,
            mi.UInt32(+mi.BSDFFlags.DeltaReflection),
            mi.UInt32(+mi.BSDFFlags.DeltaTransmission),
        )
        bs.wo = dr.select(selected_r,
                          mi.reflect(si.wi),
                          mi.refract(si.wi, cos_theta_t, eta_ti))
        bs.eta = dr.select(selected_r, mi.Float(1.0), eta_it)

        # Radiance compresses into the denser medium by eta^2; same factor
        # Mitsuba's own dielectric applies, and it must not be applied when
        # tracing importance rather than radiance. ctx.mode is known when the
        # kernel is traced, so this is an ordinary Python branch.
        if ctx.mode == mi.TransportMode.Radiance:
            factor = dr.select(selected_t, dr.sqr(eta_ti), mi.Float(1.0))
        else:
            factor = mi.Float(1.0)
        weight = mi.UnpolarizedSpectrum(factor)

        if IS_SPECTRAL and self.dispersion:
            # Collapse to the hero wavelength on the way *in* only. Once the
            # spectrum is single-channel the path is committed to that
            # wavelength, and every later refraction inside the stone already
            # bends by the same n(lambda) because si.wavelengths is unchanged
            # along the path -- so the exit refraction needs no second
            # collapse, and must not get a second factor of N.
            entering = cos_theta_i > 0.0
            weight = dr.select(selected_t & entering,
                               weight * hero_collapse_weight(),
                               weight)

        valid = active & (bs.pdf > 0.0)
        return bs, dr.select(valid, weight, mi.UnpolarizedSpectrum(0.0))

    # Both lobes are Dirac deltas, so an explicit direction query can never
    # land on them.
    def eval(self, ctx, si, wo, active):
        return mi.UnpolarizedSpectrum(0.0)

    def pdf(self, ctx, si, wo, active):
        return mi.Float(0.0)

    def eval_pdf(self, ctx, si, wo, active):
        return mi.UnpolarizedSpectrum(0.0), mi.Float(0.0)

    def component_count(self, active=True):
        return 2

    def flags(self, index=None, active=True):
        if index is None:
            return self.m_flags
        return self.m_components[index]

    def to_string(self):
        return (f"DispersiveDielectric[int_ior={self.int_ior}, "
                f"ext_ior={self.ext_ior}, dispersion={self.dispersion}]")


mi.register_bsdf("dispersive_dielectric", lambda props: DispersiveDielectric(props))
