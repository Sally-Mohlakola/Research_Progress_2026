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
    IS_SPECTRAL, REFERENCE_IOR, fresnel_spectral, hero_eta,
    hero_collapse_weight, rgb_to_spectrum,
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

        # Route A (CAUSTIC_LAYER_PLAN.md section 5), side-gated at k = 1.
        #
        # With this on, a ray arriving from *outside* the stone is handled by
        # an explicit smooth dielectric -- Fresnel split, Snell refraction,
        # total internal reflection -- exactly as DispersiveDielectric does
        # it, so the ray genuinely enters the solid. A ray arriving from
        # *inside* is handed to the neural residual S_M.
        #
        # The previous behaviour set bs.eta = 1 on every sample and drew both
        # lobes from the upper hemisphere, so no camera ray ever crossed the
        # surface. There were no internal specular chains to be sharp or
        # blurry -- E S+ L transport did not exist in the render at all, which
        # is the mechanism behind the missing high-frequency structure.
        #
        # Set false to recover the old two-lobe S_R + S_M behaviour, which is
        # the ablation baseline for the write-up.
        self.explicit_entry = bool(props.get('explicit_entry', True))

        # Neural models (set after construction)
        self.model_m = None  # Multi-scatter color: f(wi, wo) -> RGB
        self.model_t = None  # Transmittance fraction: f(wi) -> [0,1]
        self.rdm_sampler = None  # RDM direction sampler

        # Component 0 is the explicit entry lobe, component 1 the neural
        # residual. Under Route A the entry lobe is a true Dirac dielectric,
        # so it must be declared Delta or the path integrator will try to
        # connect next-event estimation through it and MIS will double-count.
        # The residual stays Glossy: its directions come from the RDM alias
        # table and are continuous, so NEE through it is legitimate.
        glossy = (mi.BSDFFlags.GlossyReflection |
                  mi.BSDFFlags.GlossyTransmission |
                  mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide)
        if self.explicit_entry:
            entry = (mi.BSDFFlags.DeltaReflection |
                     mi.BSDFFlags.DeltaTransmission |
                     mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide)
        else:
            entry = glossy
        self.m_components = [entry, glossy]
        self.m_flags = entry | glossy
    
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
        return self.eval_model_m_wi(si.wi, wo, si.wavelengths)

    def eval_model_m_wi(self, wi, wo, wavelengths):
        """
        eval_model_m with the incoming direction supplied explicitly, so the
        residual can be queried on a folded frame (see _fold) rather than on
        whatever si.wi happens to hold.
        """
        if self.model_m is None:
            return mi.UnpolarizedSpectrum(0.0)

        nn_in = self._to_tensor(wi, wo)
        nn_out = self.model_m.forward(nn_in)
        rgb = dr.unravel(mi.Color3f, dr.ravel(nn_out), order='C')
        return rgb_to_spectrum(rgb, wavelengths)
    
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
    
    # ------------------------------------------------------------------
    # Route A: explicit entry transport + folded residual
    # ------------------------------------------------------------------

    @staticmethod
    def _fold(v, flip):
        """
        Mirror a direction through the tangent plane where `flip` is set.

        The RDM was gathered by firing rays at the stone from *outside*, so
        its incoming axis only spans theta_i in [0, pi/2]
        (compute_histogram_4d bins theta_i over that range, and
        RDMSampler._incoming_bin clips to it). Once Route A lets rays inside,
        the residual gets queried with wi.z < 0, which would silently clip
        every internal hit into the single grazing bin and push Model_M --
        trained only on wi.z >= 0 -- far outside its training domain.

        Folding wi keeps the query in range. `wo` is NOT folded: the RDM's
        outgoing axis is already expressed relative to the outward normal,
        and "outward" means the same thing on both sides of the surface.
        Measured on run_15, 70.5% of rdm_m's outgoing energy sits on the
        normal's side -- the escaping half -- so mirroring wo as well (as the
        first version of this did) aimed that 70.5% back into the solid. With
        a median residual weight of 0.42 per interaction and 3.4 interactions
        before escape, that delivered about 1/19 of the lobe's energy.

        THIS IS AN APPROXIMATION, and it is the one the k = 1 side gate
        forces: it reuses the aggregate measured for incidence angle theta
        from outside as the aggregate for incidence angle theta from inside.
        Those are not the same distribution -- internal incidence past the
        critical angle (24.4 deg for diamond) is totally reflected, which has
        no counterpart on the outside. Removing this approximation means
        either gathering a second RDM for internal incidence, or moving to
        the custom-integrator form of Route A where k is a free parameter.
        """
        return mi.Vector3f(v.x, v.y, dr.select(flip, -v.z, v.z))

    def sample_entry(self, ctx, si, sample1, active):
        """
        The explicit k < 1 term: a smooth dispersive dielectric, matching
        bsdf/dispersive_dielectric.py exactly. This is what puts real
        refraction, total internal reflection and E S+ L chains back into the
        render; the residual RDM never produced them because it only ever
        returned directions in the hemisphere the ray arrived from.
        """
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)

        eta = self._eta(si)
        F, cos_theta_t, eta_it, eta_ti = mi.fresnel(cos_theta_i, eta)
        T = dr.maximum(1.0 - F, 0.0)

        # Two independent comparisons rather than `~selected_r`: under the
        # scalar variants a Dr.Jit mask is a plain Python bool and `~True` is
        # the integer -2, not False. Same reasoning as DispersiveDielectric.
        selected_r = active & (sample1 <= F)
        selected_t = active & (sample1 > F)

        wo = dr.select(selected_r,
                       mi.reflect(si.wi),
                       mi.refract(si.wi, cos_theta_t, eta_ti))
        eta_out = dr.select(selected_r, mi.Float(1.0), eta_it)
        pdf = dr.select(selected_r, F, T)
        sampled_type = dr.select(
            selected_r,
            mi.UInt32(+mi.BSDFFlags.DeltaReflection),
            mi.UInt32(+mi.BSDFFlags.DeltaTransmission),
        )

        # Radiance compresses into the denser medium by eta^2, and must not
        # be applied when tracing importance rather than radiance.
        if ctx.mode == mi.TransportMode.Radiance:
            factor = dr.select(selected_t, dr.sqr(eta_ti), mi.Float(1.0))
        else:
            factor = mi.Float(1.0)
        weight = mi.UnpolarizedSpectrum(factor)

        if IS_SPECTRAL and self.dispersion:
            # Collapse to the hero wavelength on the way *in* only -- a second
            # collapse on the way out would apply the factor N twice. This is
            # where fire enters the neural render: without a real refraction
            # to attach it to, the spectrum had nothing to fan out along.
            entering = cos_theta_i > 0.0
            weight = dr.select(selected_t & entering,
                               weight * hero_collapse_weight(),
                               weight)

        return wo, eta_out, pdf, sampled_type, weight

    def sample_residual(self, si, sample1, sample2, active):
        """
        The learned term, sampled once: draw an outgoing direction from the
        RDM and return the one-sample estimator weight.

        Split out of _sample_route_a so an integrator can select it by path
        depth instead of by which side of the surface the ray arrived from.
        Depth is the parameter the decomposition in CAUSTIC_LAYER_PLAN.md
        section 4.2 is written in; the side gate was only ever a stand-in for
        it, because a BSDF cannot see the depth.

        Returns (wo, pdf, weight) with wo in the local shading frame.
        """
        flip = mi.Frame3f.cos_theta(si.wi) < 0.0
        wi_f = self._fold(si.wi, flip)
        wo, _ = self.sample_from_rdm(wi_f, sample1, sample2)

        value = dr.clamp(self.eval_model_m_wi(wi_f, wo, si.wavelengths), 0.0, 100.0)
        pdf = self.pdf_m(wi_f, wo, active)
        weight = dr.select(active & (pdf > 0.0),
                           value / dr.maximum(pdf, 1e-6),
                           mi.UnpolarizedSpectrum(0.0))
        return wo, pdf, weight

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
        if self.explicit_entry:
            return self._sample_route_a(ctx, si, sample1, sample2, active)

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

    def _sample_route_a(self, ctx, si, sample1, sample2, active):
        """
        Side-gated k = 1 decomposition (CAUSTIC_LAYER_PLAN.md section 4.2):

            L = L_specular(entry interaction, traced explicitly)
              + L_rdm     (everything after, learned)

        The gate is the side the ray arrives from. A hit from outside is the
        entry interaction and goes through the explicit dielectric; every
        later hit is from inside the solid and goes to the residual.

        The learned half already carries the matching conditioning with no
        re-gather: utils/rdm.py sets select_m = escaped & (depth >= 3), so
        rdm_m -- the only histogram Model_M is trained on -- already excludes
        the short paths this function now traces explicitly. Note the
        mismatch that leaves: the explicit term covers one interaction while
        the residual excludes two, so the direct enter-and-exit path
        (depth == 2, gathered into rdm_t) is represented by neither. That is
        the known cost of pinning k = 1 to the side gate, and it is the first
        thing to check if the render comes out dim.
        """
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        outside = cos_theta_i > 0.0

        wo_e, eta_e, pdf_e, type_e, weight_e = self.sample_entry(
            ctx, si, sample1, active)

        # Residual branch, queried on the folded frame.
        flip = ~outside
        wi_f = self._fold(si.wi, flip)
        wo_m, _ = self.sample_from_rdm(wi_f, sample1, sample2)
        wo_f = wo_m          # queried and scattered in the same direction

        value_m = dr.clamp(
            self.eval_model_m_wi(wi_f, wo_f, si.wavelengths), 0.0, 100.0)
        pdf_m = self.pdf_m(wi_f, wo_f, active)
        weight_m = dr.select(active & (pdf_m > 0.0),
                             value_m / dr.maximum(pdf_m, 1e-6),
                             mi.UnpolarizedSpectrum(0.0))

        bs = mi.BSDFSample3f()
        bs.wo = dr.select(outside, wo_e, wo_m)
        bs.pdf = dr.select(outside, pdf_e, pdf_m)
        bs.eta = dr.select(outside, eta_e, mi.Float(1.0))
        bs.sampled_component = dr.select(outside, mi.UInt32(0), mi.UInt32(1))
        bs.sampled_type = dr.select(
            outside, type_e, mi.UInt32(+mi.BSDFFlags.GlossyReflection))

        weight = dr.select(outside, weight_e, weight_m)
        valid = active & (bs.pdf > 0.0)
        return bs, dr.select(valid, weight, mi.UnpolarizedSpectrum(0.0))

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
        if self.explicit_entry:
            # Outside hits are a Dirac dielectric, so a direction query can
            # never land on them and must return exactly zero -- returning a
            # finite value there would let next-event estimation add energy
            # the delta lobe already accounts for. Inside hits evaluate the
            # residual on the folded frame, matching _sample_route_a.
            outside = mi.Frame3f.cos_theta(si.wi) > 0.0
            flip = ~outside
            value = dr.clamp(
                self.eval_model_m_wi(self._fold(si.wi, flip), wo,
                                     si.wavelengths), 0.0, 100.0)
            return dr.select(active & ~outside, dr.maximum(value, 0.0),
                             mi.UnpolarizedSpectrum(0.0))

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
        if self.explicit_entry:
            outside = mi.Frame3f.cos_theta(si.wi) > 0.0
            flip = ~outside
            pdf = self.pdf_m(self._fold(si.wi, flip), wo, active)
            return dr.select(active & ~outside, pdf, 0.0)

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