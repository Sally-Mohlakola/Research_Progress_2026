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
    
    def _wi_to_numpy(self, wi):
        """Extract first active wi as numpy vector for RDM bin lookup."""
        wx = dr.slice(wi.x, 0)
        wy = dr.slice(wi.y, 0)
        wz = dr.slice(wi.z, 0)
        return np.array([float(wx), float(wy), float(wz)], dtype=np.float32)
    
    def sample_from_rdm(self, wi, sample1, sample2):
        """
        Sample outgoing direction from RDM histogram.
        
        Uses the RDM sampler to draw directions from the learned distribution.
        """
        if self.rdm_sampler is None:
            # Fallback: uniform hemisphere
            wo = mi.warp.square_to_uniform_hemisphere(sample2)
            pdf = mi.warp.square_to_uniform_hemisphere_pdf(wo)
            return wo, pdf
        
        # Get representative incoming direction
        wi_np = self._wi_to_numpy(wi)
        
        # Convert to spherical coordinates
        theta_i = float(dr.acos(wi_np[2]))
        phi_i = float(dr.atan2(wi_np[1], wi_np[0]))
        
        # Sample from RDM
        u1 = float(dr.slice(sample1, 0))
        u2 = float(dr.slice(sample2, 0))
        
        theta_o, phi_o, pdf_val = self.rdm_sampler.sample_outgoing(
            theta_i, phi_i, u1, u2
        )
        
        # Convert to vector
        wo = mi.Vector3f(
            dr.sin(theta_o) * dr.cos(phi_o),
            dr.sin(theta_o) * dr.sin(phi_o),
            dr.cos(theta_o)
        )
        pdf = mi.Float(pdf_val)
        
        return wo, pdf
    
    # ------------------------------------------------------------------
    # BSDF interface - Fully neural
    # ------------------------------------------------------------------
    
    def sample(self, ctx, si, sample1, sample2, active):
        """
        Fully neural sampling.
        
        1. Model_T predicts transmittance (R/T ratio)
        2. RDM sampler gives outgoing direction
        3. Model_M predicts color for (wi, wo)
        """
        bs = mi.BSDFSample3f()
        
        # 1. Get transmittance from Model_T (NEURAL)
        p_t = self.eval_model_t(si.wi)
        p_r = 1.0 - p_t  # Reflectance
        
        # 2. Sample direction from RDM (NEURAL)
        wo, pdf = self.sample_from_rdm(si.wi, sample1, sample2)
        bs.wo = wo
        bs.pdf = pdf
        bs.eta = mi.Float(1.0)
        
        # 3. Get color from Model_M (NEURAL)
        color = self.eval_model_m(si.wi, wo)
        color = dr.clamp(color, 0.0, 100.0)  # Allow bright diamond fire
        
        # 4. Combine: color * (reflectance or transmittance)
        # Use Model_T to weight reflection vs transmission
        # This ensures energy conservation
        energy = dr.select(wo.z > 0, p_r, p_t)  # Upward = reflection, downward = transmission
        value = color * energy
        
        # Set BSDF flags based on direction
        bs.sampled_component = mi.UInt32(0)
        bs.sampled_type = mi.UInt32(+mi.BSDFFlags.GlossyReflection)
        #if wo.z < 0:
            #bs.sampled_type = mi.UInt32(+mi.BSDFFlags.GlossyTransmission)
        
        # Weight = value / pdf (standard for Monte Carlo)
        safe_pdf = dr.maximum(pdf, 1e-6)
        weight = dr.select(active & (pdf > 0), value / safe_pdf, mi.Color3f(0.0))
        
        return bs, weight
    
    def eval(self, ctx, si, wo, active):
        """
        Evaluate BSDF for given directions.
        
        Returns Model_M output for any (wi, wo).
        """
        if self.model_m is None:
            return mi.Color3f(0.0)
        
        # Get neural color
        value = self.eval_model_m(si.wi, wo)
        value = dr.clamp(value, 0.0, 100.0)
        
        # Modulate by Model_T if available
        if self.model_t is not None:
            p_t = self.eval_model_t(si.wi)
            # If wo points upward, it's reflection; downward = transmission
            energy = dr.select(wo.z > 0, 1.0 - p_t, p_t)
            value = value * energy
        
        return dr.select(active, dr.maximum(value, 0.0), mi.Color3f(0.0))
    
    def pdf(self, ctx, si, wo, active):
        """
        PDF for neural BSDF.
        
        Uses RDM sampler if available, else uniform.
        """
        if self.rdm_sampler is not None:
            wi_np = self._wi_to_numpy(si.wi)
            wo_np = self._wi_to_numpy(wo)
            
            theta_i = float(dr.acos(wi_np[2]))
            phi_i = float(dr.atan2(wi_np[1], wi_np[0]))
            theta_o = float(dr.acos(wo_np[2]))
            phi_o = float(dr.atan2(wo_np[1], wo_np[0]))
            
            pdf_val = self.rdm_sampler.pdf_outgoing(theta_i, phi_i, theta_o, phi_o)
            return dr.full(mi.Float, pdf_val, dr.width(si.wi))
        
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