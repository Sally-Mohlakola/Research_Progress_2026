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

mi.set_variant(variant)


class NeuralDiamond(mi.BSDF):
    """
    Fully neural diamond BSDF.
    
    - sample(): Uses Model_T for R/T ratio, RDM sampler for direction
    - eval(): Returns Model_M output for any (wi, wo)
    - pdf(): Returns RDM pdf if available, else uniform
    """
    
    def __init__(self, props):
        mi.BSDF.__init__(self, props)
        self.int_ior = props.get('int_ior', 2.419)
        self.ext_ior = props.get('ext_ior', 1.000277)
        
        # Neural models (set after construction)
        self.model_m = None  # Multi-scatter color: f(wi, wo) -> RGB
        self.model_t = None  # Transmittance fraction: f(wi) -> [0,1]
        self.rdm_sampler = None  # RDM direction sampler
        
        # Fallback parameters
        self.use_physics_fallback = props.get('use_physics_fallback', True)
        self.confidence_threshold = props.get('confidence_threshold', 0.01)  # Min value to trust neural output
        
        # Glossy flags - we're sampling from learned distribution
        # Not delta because directions come from RDM (continuous)
        self.m_flags = (mi.BSDFFlags.GlossyReflection | 
                        mi.BSDFFlags.GlossyTransmission |
                        mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide)
        self.m_components = [self.m_flags, self.m_flags]
    
    # ------------------------------------------------------------------
    # Physics-based fallback for untrained directions
    # ------------------------------------------------------------------
    
    def eval_analytic_dielectric(self, wi, wo):
        """
        Evaluate analytic dielectric BSDF for fallback.
        Returns approximate color for reflection/transmission.
        """
        cos_theta_i = mi.Frame3f.cos_theta(wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        
        # Fresnel reflection coefficient
        eta = self.int_ior / self.ext_ior
        F, _, _, _ = mi.fresnel(dr.abs(cos_theta_i), eta)
        
        # Determine if this is reflection or transmission
        same_side = (cos_theta_i * cos_theta_o) > 0
        
        # Reflection: Fresnel * white (diamond is colorless)
        # Transmission: (1-Fresnel) * white
        value = dr.select(same_side, F, 1.0 - F)
        
        # Scale by geometric term (simplified)
        value = value * dr.abs(cos_theta_o)
        
        return mi.Color3f(value, value, value)
    
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
    
    def eval_model_m(self, wi, wo):
        """
        Evaluate multi-scatter color model.
        Returns RGB color for any (wi, wo).
        """
        if self.model_m is None:
            return mi.Color3f(0.0)
        
        nn_in = self._to_tensor(wi, wo)
        nn_out = self.model_m.forward(nn_in)
        return dr.unravel(mi.Color3f, dr.ravel(nn_out), order='C')
    
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
        sample2 supplies two independent uniform variates per ray
        (sample2.x for alias selection, sample2.y for alias accept/reject).
        """
        if self.rdm_sampler is None:
            wo  = mi.warp.square_to_uniform_hemisphere(sample2)
            pdf = mi.warp.square_to_uniform_hemisphere_pdf(wo)
            return wo, pdf

        theta_i, phi_i = self._dir_to_spherical(wi)

        # sample_outgoing expects Dr.Jit Float arrays (one per ray).
        # Returns theta_o, phi_o, pdf -- all Dr.Jit Float, same width.
        theta_o, phi_o, pdf = self.rdm_sampler.sample_outgoing(
            theta_i, phi_i, sample2.x, sample2.y,
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
    
    def sample(self, ctx, si, sample1, sample2, active):
        """
        Fully neural sampling with physics-based fallback.
        
        1. Model_T predicts transmittance (R/T ratio) or use Fresnel
        2. RDM sampler gives outgoing direction or use hemisphere
        3. Model_M predicts color for (wi, wo) or use analytic
        """
        bs = mi.BSDFSample3f()
        
        # 1. Get transmittance - use Fresnel as fallback
        if self.model_t is not None:
            p_t = self.eval_model_t(si.wi)
        else:
            cos_theta = mi.Frame3f.cos_theta(si.wi)
            eta = self.int_ior / self.ext_ior
            F, _, _, _ = mi.fresnel(cos_theta, eta)
            p_t = 1.0 - F  # Transmittance
        
        p_r = 1.0 - p_t  # Reflectance
        
        # 2. Sample direction from RDM or uniform hemisphere
        wo, pdf = self.sample_from_rdm(si.wi, sample1, sample2)
        bs.wo = wo
        bs.pdf = pdf
        bs.eta = mi.Float(1.0)
        
        # 3. Get color - try neural first, fallback to analytic if low confidence
        if self.model_m is not None:
            color = self.eval_model_m(si.wi, wo)
            color = dr.clamp(color, 0.0, 100.0)
            
            # Check confidence
            if self.use_physics_fallback:
                neural_magnitude = dr.norm(color)
                use_fallback = neural_magnitude < self.confidence_threshold
                
                if dr.any(use_fallback):
                    analytic_color = self.eval_analytic_dielectric(si.wi, wo)
                    blend_factor = dr.clamp(neural_magnitude / self.confidence_threshold, 0.0, 1.0)
                    color = analytic_color * (1.0 - blend_factor) + color * blend_factor
        else:
            # Pure analytic fallback
            color = self.eval_analytic_dielectric(si.wi, wo)
        
        # 4. Combine: color * (reflectance or transmittance)
        # Use Model_T to weight reflection vs transmission
        energy = dr.select(wo.z > 0, p_r, p_t)  # Upward = reflection, downward = transmission
        value = color * energy
        
        # Set BSDF flags based on direction
        bs.sampled_component = mi.UInt32(0)
        bs.sampled_type = mi.UInt32(+mi.BSDFFlags.GlossyReflection)
        
        # Weight = value / pdf (standard for Monte Carlo)
        safe_pdf = dr.maximum(pdf, 1e-6)
        weight = dr.select(active & (pdf > 0), value / safe_pdf, mi.Color3f(0.0))
        
        return bs, weight
    
    def eval(self, ctx, si, wo, active):
        """
        Evaluate BSDF for given directions.
        
        Returns Model_M output for any (wi, wo), with physics-based
        fallback for untrained directions (where neural output is near-zero).
        """
        if self.model_m is None:
            # No neural model, use pure analytic
            return self.eval_analytic_dielectric(si.wi, wo)
        
        # Get neural color
        neural_value = self.eval_model_m(si.wi, wo)
        neural_value = dr.clamp(neural_value, 0.0, 100.0)
        
        # Modulate by Model_T if available
        if self.model_t is not None:
            p_t = self.eval_model_t(si.wi)
            # If wo points upward, it's reflection; downward = transmission
            energy = dr.select(wo.z > 0, 1.0 - p_t, p_t)
            neural_value = neural_value * energy
        
        # Fallback to physics-based BSDF for low-confidence predictions
        if self.use_physics_fallback:
            neural_magnitude = dr.norm(neural_value)
            use_fallback = neural_magnitude < self.confidence_threshold
            
            if dr.any(use_fallback):
                analytic_value = self.eval_analytic_dielectric(si.wi, wo)
                # Blend: use analytic when neural is too low
                # For smooth transition, blend between analytic and neural
                blend_factor = dr.clamp(neural_magnitude / self.confidence_threshold, 0.0, 1.0)
                value = analytic_value * (1.0 - blend_factor) + neural_value * blend_factor
            else:
                value = neural_value
        else:
            value = neural_value
        
        return dr.select(active, dr.maximum(value, 0.0), mi.Color3f(0.0))
    
    def pdf(self, ctx, si, wo, active):
        """
        Per-ray vectorized PDF lookup from the RDM alias table.
        No dr.slice(), no broadcast -- every ray gets its own pdf value.
        """
        if self.rdm_sampler is not None:
            theta_i, phi_i = self._dir_to_spherical(si.wi)
            theta_o, phi_o = self._dir_to_spherical(wo)
            pdf = self.rdm_sampler.pdf_outgoing(theta_i, phi_i, theta_o, phi_o)
            return dr.select(active, pdf, 0.0)

        # Uniform fallback
        return mi.warp.square_to_uniform_hemisphere_pdf(wo)
    
    def component_count(self, active=True):
        return 2
    
    def flags(self, index=None, active=True):
        if index is None:
            return self.m_flags
        return self.m_components[index]
    
    def eval_attribute(self, name, si, active):
        return mi.Color3f(0.0)
    
    def to_string(self):
        has_m = self.model_m is not None
        has_t = self.model_t is not None
        has_rdm = self.rdm_sampler is not None
        return (f"NeuralDiamond["
                f"Model_M={has_m}, Model_T={has_t}, RDM={has_rdm}]")


mi.register_bsdf("neural_diamond", lambda props: NeuralDiamond(props))