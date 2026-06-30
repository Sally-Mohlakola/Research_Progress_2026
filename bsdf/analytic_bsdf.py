from config import device, variant

import mitsuba as mi
import drjit as dr
mi.set_variant(variant)

class DiamondShading(mi.BSDF):
    def __init__(self, params):
        mi.BSDF.__init__(self,params)
        self.int_ior = params.get("int_ior",2.419)
        self.ext_ior = params.get("ext_ior", 1.000277)
        self.eta = self.int_ior/self.ext_ior

        reflection = mi.BSDFFlags.DeltaReflection | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        transmission = mi.BSDFFlags.DeltaTransmission | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide #refraction (Snell)
        self.m_components = [reflection, transmission]
        self.m_flags = reflection | transmission
    
    def sample(self, ctx, sc, sample1, sample2, active):
        cos_theta_i = mi.Frame3f.cos_theta(sc.wi)
        reflect_c, cos_theta_t, eta_it, eta_ti = mi.fresnel(cos_theta_i, mi.Float(self.eta))

        trans_c = dr.maximum(1.0 - reflect_c,0.0) # proportion of r and t

        selected = (sample1 <= reflect_c) & active

        bs = mi.BSDFSample3f()
        bs.pdf = dr.select(selected, reflect_c, trans_c)
        bs.sampled_component = dr.select(selected, mi.UInt32(0), mi.UInt32(1))
        bs.sampled_type = dr.select(selected, mi.UInt32(+mi.BSDFFlags.DeltaReflection), mi.UInt32(+mi.BSDFFlags.DeltaTransmission))

        wo_r = mi.reflect(sc.wi)
        wo_t = mi.refract(sc.wi, cos_theta_t, eta_ti)
        bs.wo = dr.select(selected, wo_r, wo_t)
        bs.eta = dr.select(selected, 1.0, eta_it)

        value = mi.Color3f(1.0) # always 1 for diamond dialectric

        valid = active & (bs.pdf >0.0)
        return bs, dr.select(valid, value, mi.Color3f(0.0))
    

    def eval(self, ctx, sc, wo, active):
        return mi.Color3f(0.0)

    def pdf (self, ctx, sc, wo, active):
        return mi.Float(0.0)
    
    def light_component_count(self, active=True):
        return 2
    
    def flags(self, index=None, active=True):
        if index is None:
            return self.m_flags
        return self.m_components[index]
    
    #missing 
    """
    def eval_attribute(self, name, si, active):
        return mi.Color3f(0.0)
    """
    def to_string (self):
        return f"DiamondFacet[int_ior={self.int_ior}, ext_ior={self.ext_ior}]"

mi.register_bsdf("diamond_shading", lambda props: DiamondShading(props))
