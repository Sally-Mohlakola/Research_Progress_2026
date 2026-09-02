"""
neural_diamond_bsdf.py - Fully neural diamond BSDF.

This version uses neural models for everything:
- Model_T: Predicts transmittance fraction f(wi) -> [0,1]
- Model_M: Predicts multi-scatter color f(wi, wo) -> RGB
- RDM sampler: Samples outgoing directions from learned distribution

The neural models are loaded from checkpoint and used during rendering.
No PDF fitting needed - Model_T and Model_M are used directly.
"""

import os
import sys
import numpy as np
import mitsuba as mi
import drjit as dr
from config import device, variant

# Respect a variant the caller has already chosen (e.g. a scalar_spectral
# test harness) instead of forcing the configured one.
if mi.variant() is None:
    mi.set_variant(variant)

from bsdf.dispersion import (
    IS_SPECTRAL, REFERENCE_IOR, fresnel_spectral, hero_eta, rgb_to_spectrum,
)


class NeuralDiamond(mi.BSDF):
    """
    Fully neural diamond BSDF.
    
    - sample(): Uses Model_T for R/T ratio, RDM sampler for direction
    - eval(): Returns Model_M output for any (wi, wo)
    - pdf(): Returns RDM pdf if available, else uniform
    """
    
    def __init__(self, props):
        mi.BSDF.__init__(self, props)
        self.int_ior = props.get('int_ior', REFERENCE_IOR)
        self.ext_ior = props.get('ext_ior', 1.000277)
        self.eta = self.int_ior / self.ext_ior

        # Wavelength-dependent IOR for the analytic reflection lobe. Only S_R
        # can disperse: the neural S_M is fitted to an RDM with no wavelength
        # axis, so it returns the same spectrum whatever wavelength a ray
        # carries. Fire in the neural term needs that axis added to the
        # gather -- see FACET_CONDITIONED_BSDF.md.
        self.dispersion = bool(props.get('dispersion', True))

        # Roughness of the analytic direct-reflection lobe S_R. A polished
        # facet is very nearly a perfect mirror, but keeping a small non-zero
        # roughness leaves S_R an ordinary sampleable lobe instead of a Dirac
        # delta -- no delta bookkeeping in eval()/pdf(), at the cost of
        # slightly softened highlights. Lower it toward 0 for sharper ones.
        self.r_alpha = props.get('r_alpha', 0.05)

        # Neural models (set after construction)
        self.model_m = None  # Multi-scatter color: f(wi, wo) -> RGB
        self.model_t = None  # Transmittance fraction: f(wi) -> [0,1]
        self.rdm_sampler = None  # RDM direction sampler

        # Glossy flags - we're sampling from learned distribution
        # Not delta because directions come from RDM (continuous)
        self.m_flags = (mi.BSDFFlags.GlossyReflection | 
                        mi.BSDFFlags.GlossyTransmission |
                        mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide)
        self.m_components = [self.m_flags, self.m_flags]
    
    # ------------------------------------------------------------------
    # Neural model evaluation
    # ------------------------------------------------------------------
    
    def _to_tensor(self, wi, wo=None):
        """Convert directions to tensor format for neural models."""
        if wo is None:
            # Model_T: 3D input (wi)
            nn_in = dr.zeros(mi.TensorXf, shape=[3, dr.width(wi)])
            nn_in[0] = wi.x
            nn_in[1] = wi.y
            nn_in[2] = wi.z
        else:
            # Model_M: 6D input (wi, wo)
            nn_in = dr.zeros(mi.TensorXf, shape=[6, dr.width(wi)])
            nn_in[0] = wi.x
            nn_in[1] = wi.y
            nn_in[2] = wi.z
            nn_in[3] = wo.x
            nn_in[4] = wo.y
            nn_in[5] = wo.z
        return nn_in
    
    def eval_model_t(self, wi):
        """
        Evaluate transmittance model.
        Returns fraction in [0, 1] where:
        - 0 = total reflection
        - 1 = total transmission
        """
        if self.model_t is None:
            # Fallback: Fresnel
            cos_theta = mi.Frame3f.cos_theta(wi)
            eta = self.int_ior / self.ext_ior
            F,a,b,c = mi.fresnel(cos_theta, eta)
            return 1.0 - F  # Transmittance
        
        nn_in = self._to_tensor(wi)
        t_raw = mi.Float(self.model_t.forward(nn_in).array)
        # Clamp to valid range
        return dr.clamp(t_raw, 0.0, 1.0)
    
    def eval_model_m(self, si, wo):
        """
        Evaluate multi-scatter colour model.

        The network emits three numbers; under a spectral variant those are
        treated as an sRGB colour and upsampled to the ray's wavelengths.
        Returns an UnpolarizedSpectrum either way, which is what a BSDF has
        to hand back.
        """
        if self.model_m is None:
            return mi.UnpolarizedSpectrum(0.0)

        nn_in = self._to_tensor(si.wi, wo)
        nn_out = self.model_m.forward(nn_in)
        rgb = dr.unravel(mi.Color3f, dr.ravel(nn_out), order='C')
        return rgb_to_spectrum(rgb, si.wavelengths)
    
    # ------------------------------------------------------------------
    # Analytic direct-reflection component S_R
    # ------------------------------------------------------------------

    def _r_distr(self):
        return mi.MicrofacetDistribution(mi.MicrofacetType.GGX, self.r_alpha, True)

    def _eta(self, si):
        """Relative IOR at this ray's hero wavelength."""
        return hero_eta(si, self.int_ior, self.ext_ior, self.dispersion)

    def fresnel_i(self, si):
        """
        Fresnel reflectance at the first surface, as a scalar, for lobe
        weighting. The hero wavelength is enough here: this only picks which
        lobe to sample, and any reasonable probability leaves the one-sample
        mixture estimator unbiased.
        """
        F, _, _, _ = mi.fresnel(mi.Frame3f.cos_theta(si.wi), self._eta(si))
        return F

    def eval_r(self, si, wo, active):
        """
        Direct single-bounce reflection off the facet (paper Eq. 8, adapted).

        Verified against the gathered data: rdm_r's energy per incoming bin
        matches analytic Fresnel F(theta_i) to within 0.06% at near-normal
        incidence, so this component is exactly Fresnel and needs no fitting.
        That match also fixes the units -- integral over omega_o is F(theta_i),
        i.e. the returned value already carries the outgoing cosine, which is
        the same convention Model_M is trained in, so the two are summable.

        The paper's (1 - P(T|omega_i)) prefactor is deliberately NOT applied:
        their P(T) is "light misses every fiber", which has no analogue in a
        solid gem, and the measured energy is already exactly F with no
        further attenuation.
        """
        wi = si.wi
        cos_i = mi.Frame3f.cos_theta(wi)
        cos_o = mi.Frame3f.cos_theta(wo)
        valid = active & (cos_i > 0.0) & (cos_o > 0.0)

        m = dr.normalize(wi + wo)
        distr = self._r_distr()
        D = distr.eval(m)
        G = distr.G(wi, wo, m)

        # Per-wavelength Fresnel. Reflection needs no hero collapse -- its
        # direction is wavelength-independent -- so all channels stay live
        # and simply carry slightly different reflectances.
        F = fresnel_spectral(dr.dot(wi, m), si,
                             self.int_ior, self.ext_ior, self.dispersion)

        # f_r * cos_o  ==  F*D*G / (4 * cos_i)
        val = F * (D * G / dr.maximum(4.0 * cos_i, 1e-8))
        return dr.select(valid, val, mi.UnpolarizedSpectrum(0.0))

    def pdf_r(self, si, wo, active):
        wi = si.wi
        cos_i = mi.Frame3f.cos_theta(wi)
        cos_o = mi.Frame3f.cos_theta(wo)
        valid = active & (cos_i > 0.0) & (cos_o > 0.0)
        m = dr.normalize(wi + wo)
        distr = self._r_distr()
        pdf = distr.pdf(wi, m) / dr.maximum(4.0 * dr.abs(dr.dot(wo, m)), 1e-8)
        return dr.select(valid, pdf, 0.0)

    def sample_r(self, si, sample2):
        distr = self._r_distr()
        m, _ = distr.sample(si.wi, sample2)
        return mi.reflect(si.wi, m)

    # ------------------------------------------------------------------
    # RDM-based direction sampling
    # ------------------------------------------------------------------
    
    def _dir_to_spherical(self, v):
        """
        Convert a Dr.Jit Vector3f to (theta, phi) angle arrays,
        one value per ray -- fully vectorized, no dr.slice().
        Convention: theta from +Z (matching how the RDM was collected
        in gather_rdm.py: theta_i = dr.acos(omega_i.z)).
        """
        theta = dr.acos(dr.clip(v.z, -1.0, 1.0))
        phi   = dr.atan2(v.y, v.x)
        return theta, phi

    def sample_from_rdm(self, wi, sample1, sample2):
        """
        Per-ray vectorized direction sampling from the RDM alias table.

        Every ray in the batch independently looks up its own
        (theta_i, phi_i) bin and draws its own outgoing direction --
        no dr.slice(), no broadcast of a single shared direction.
        The alias sampler needs three independent variates per ray:
        sample2.x drives the alias lookup, and sample2.y / sample1 jitter
        theta_o and phi_o independently. sample1 is Mitsuba's spare
        component-selection variate, unused by this BSDF (there's a single
        neural lobe), so it's free to serve as the third dimension --
        without it the two jitters would have to share one variate, which
        collapses every sample onto its bin's diagonal.
        """
        if self.rdm_sampler is None:
            wo  = mi.warp.square_to_uniform_hemisphere(sample2)
            pdf = mi.warp.square_to_uniform_hemisphere_pdf(wo)
            return wo, pdf

        theta_i, phi_i = self._dir_to_spherical(wi)

        # sample_outgoing expects Dr.Jit Float arrays (one per ray).
        # Returns theta_o, phi_o, pdf -- all Dr.Jit Float, same width.
        theta_o, phi_o, pdf = self.rdm_sampler.sample_outgoing(
            theta_i, phi_i, sample2.x, sample2.y, mi.Float(sample1),
        )

        wo = mi.Vector3f(
            dr.sin(theta_o) * dr.cos(phi_o),
            dr.sin(theta_o) * dr.sin(phi_o),
            dr.cos(theta_o),
        )
        return wo, pdf
    
    # ------------------------------------------------------------------
    # BSDF interface - Fully neural
    # ------------------------------------------------------------------
    
    def _lobe_prob(self, si):
        """
        Probability of picking the analytic reflection lobe. Total albedo is
        ~1 and the directly-reflected share is exactly F(theta_i) (measured),
        so Fresnel is the physically-right split. Clamped so neither lobe can
        become unsamplable.
        """
        return dr.clip(self.fresnel_i(si), 0.05, 0.95)

    def sample(self, ctx, si, sample1, sample2, active):
        """
        Two-lobe sample: the analytic direct reflection S_R and the neural
        multi-scatter S_M, combined as S = S_R + S_M (paper Eq. 6).

        Note S_M is NOT scaled by Model_T. rdm_m already integrates to the
        measured multi-scatter albedo, so weighting it by a separately-learned
        R/T fraction applies the split a second time -- the paper's Eq. 6 is a
        plain sum. Model_T's job here is lobe bookkeeping, not attenuation.
        """
        bs = mi.BSDFSample3f()

        p_lobe = self._lobe_prob(si)
        pick_r = sample1 < p_lobe

        # Lobe choice consumes only the comparison, so rescale sample1 back to
        # a fresh uniform variate rather than burning a dimension.
        u_reuse = dr.select(pick_r,
                            sample1 / dr.maximum(p_lobe, 1e-6),
                            (sample1 - p_lobe) / dr.maximum(1.0 - p_lobe, 1e-6))
        u_reuse = dr.clip(u_reuse, 0.0, 1.0)

        wo_r = self.sample_r(si, sample2)
        wo_m, _ = self.sample_from_rdm(si.wi, u_reuse, sample2)
        wo = dr.select(pick_r, wo_r, wo_m)

        # One-sample mixture estimator: evaluate BOTH lobes and divide by the
        # combined pdf, which is correct whichever lobe was actually drawn.
        value = self.eval_r(si, wo, active) + self._eval_m_clamped(si, wo)
        pdf = (p_lobe * self.pdf_r(si, wo, active)
               + (1.0 - p_lobe) * self.pdf_m(si.wi, wo, active))

        bs.wo = wo
        bs.pdf = pdf
        bs.eta = mi.Float(1.0)
        bs.sampled_component = dr.select(pick_r, mi.UInt32(0), mi.UInt32(1))
        bs.sampled_type = mi.UInt32(+mi.BSDFFlags.GlossyReflection)

        safe_pdf = dr.maximum(pdf, 1e-6)
        weight = dr.select(active & (pdf > 0), value / safe_pdf,
                           mi.UnpolarizedSpectrum(0.0))

        return bs, weight

    def _eval_m_clamped(self, si, wo):
        return dr.clamp(self.eval_model_m(si, wo), 0.0, 100.0)

    def pdf_m(self, wi, wo, active):
        """PDF of the neural (RDM) lobe alone."""
        if self.rdm_sampler is None:
            return mi.warp.square_to_uniform_hemisphere_pdf(wo)
        theta_i, phi_i = self._dir_to_spherical(wi)
        theta_o, phi_o = self._dir_to_spherical(wo)
        return self.rdm_sampler.pdf_outgoing(theta_i, phi_i, theta_o, phi_o)
    
    def eval(self, ctx, si, wo, active):
        """
        S = S_R + S_M -- the analytic direct reflection plus the neural
        multi-scatter term, in matching units (both carry the outgoing
        cosine). No Model_T modulation: see sample().
        """
        value = self.eval_r(si, wo, active)
        if self.model_m is not None:
            value = value + self._eval_m_clamped(si, wo)

        return dr.select(active, dr.maximum(value, 0.0),
                         mi.UnpolarizedSpectrum(0.0))

    def pdf(self, ctx, si, wo, active):
        """
        Mixture pdf matching sample()'s lobe split, so the two stay consistent
        for MIS and for any integrator that queries pdf() directly.
        """
        p_lobe = self._lobe_prob(si)
        pdf = (p_lobe * self.pdf_r(si, wo, active)
               + (1.0 - p_lobe) * self.pdf_m(si.wi, wo, active))
        return dr.select(active, pdf, 0.0)
    
    def component_count(self, active=True):
        return 2
    
    def flags(self, index=None, active=True):
        if index is None:
            return self.m_flags
        return self.m_components[index]
    
    def eval_attribute(self, name, si, active):
        return mi.UnpolarizedSpectrum(0.0)
    
    def to_string(self):
        has_m = self.model_m is not None
        has_t = self.model_t is not None
        has_rdm = self.rdm_sampler is not None
        return (f"NeuralDiamond["
                f"Model_M={has_m}, Model_T={has_t}, RDM={has_rdm}]")


mi.register_bsdf("neural_diamond", lambda props: NeuralDiamond(props))