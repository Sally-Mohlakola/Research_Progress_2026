"""
depth_aware.py - a path integrator that switches from explicit transport to
the learned aggregate at a chosen internal-bounce depth.

This is Route A of CAUSTIC_LAYER_PLAN.md in its proper form. The decomposition
it implements is

    L  =  L_specular( < k diamond interactions, traced explicitly )
       +  L_rdm     ( the rest, evaluated once as an aggregate    )

A BSDF cannot express this. Mitsuba's `BSDFContext` carries no path depth
(checked against 3.9.1: it exposes only mode, type_mask, component, is_enabled
and reverse), and more fundamentally the RDM is not a material BSDF at all --
it is a boundary operator for the whole stone, measured by firing rays at it
from outside and recording where they escape. Applying it at every internal
interaction composes that aggregate with itself and queries it far outside the
domain it was measured on. It has to be applied once, and only something that
owns the path loop can do that.

So: below `k` diamond interactions the stone is a real dispersive dielectric
(NeuralDiamond.sample_entry, which reproduces analytic short-path transport to
within 0.02% on energy and peak). At the k-th, the RDM lobe is sampled once
and the path is marked as having consumed the aggregate; from then on the ray
passes straight through the stone without interacting, because the aggregate
has already accounted for the rest of the internal journey.

k is a free parameter here, which is the point -- the measured transition
between "polished glass" and "diamond" sits between depth 4 and 8, so k is the
dial that decides whether the render looks like a diamond, and sweeping it
gives the quality/cost curve the method's claim rests on.

Illumination is by BSDF sampling only, with no next-event estimation. That is
unbiased but noisier than Mitsuba's `path`, so it needs higher spp -- and it
means comparisons must render the analytic reference through *this* integrator
too, or the difference measured is the integrator, not the BSDF.
"""

import mitsuba as mi
import drjit as dr
from config import variant

if mi.variant() is None:
    mi.set_variant(variant)


class DepthAwarePath(mi.SamplingIntegrator):

    def __init__(self, props=None):
        mi.SamplingIntegrator.__init__(self, props or mi.Properties())
        self.max_depth = 16
        self.k = 2
        # Set these after the scene is built; see attach().
        self.dia_shape = None      # mi.ShapePtr for the diamond
        self.dia_bsdf = None       # the NeuralDiamond instance, or None
        self.use_residual = True

    def attach(self, shape, bsdf, k=2, max_depth=16, use_residual=True):
        """Point the integrator at the diamond and choose k."""
        self.dia_shape = mi.ShapePtr(shape)
        self.dia_bsdf = bsdf
        self.k = int(k)
        self.max_depth = int(max_depth)
        self.use_residual = bool(use_residual) and bsdf is not None
        return self

    def sample(self, scene, sampler, ray, medium=None, active=True, **kwargs):
        ray = mi.Ray3f(ray)
        L = mi.Spectrum(0.0)
        beta = mi.Spectrum(1.0)
        active = mi.Bool(active)
        done = mi.Bool(False)
        # Diamond interactions so far, and whether the aggregate was consumed.
        d_depth = mi.UInt32(0)
        spent = mi.Bool(False)

        ctx = mi.BSDFContext()

        for _ in range(self.max_depth):
            si = scene.ray_intersect(ray, active & ~done)

            # Rays that leave the scene pick up the constant ambient. Mitsuba
            # 3.9 has no Scene.eval_environment, so the environment emitter is
            # evaluated directly on a synthetic interaction carrying only the
            # direction the ray left along.
            escaped = active & ~done & ~si.is_valid()
            env = scene.environment()
            if env is not None:
                si_env = dr.zeros(mi.SurfaceInteraction3f)
                si_env.wi = -ray.d
                si_env.wavelengths = ray.wavelengths
                L += dr.select(escaped, beta * env.eval(si_env, escaped),
                               mi.Spectrum(0.0))
            done = done | escaped

            alive = active & ~done & si.is_valid()

            # Emitted radiance from an area panel we happened to hit.
            emitter = si.emitter(scene)
            has_em = alive & (emitter != None)  # noqa: E711 -- Dr.Jit ptr test
            L += dr.select(has_em, beta * emitter.eval(si, has_em),
                           mi.Spectrum(0.0))

            # With no diamond BSDF attached every shape goes through its own
            # BSDF, which makes this a plain BSDF-sampling path tracer -- the
            # configuration used to validate the loop against Mitsuba's `path`.
            if self.dia_bsdf is None:
                is_dia = mi.Bool(False)
            else:
                is_dia = alive & (si.shape == self.dia_shape)

            # A path that has already consumed the aggregate must not touch
            # the stone again -- the aggregate covers the remaining internal
            # journey. Pass straight through instead.
            through = is_dia & spent
            explicit = is_dia & ~spent & (d_depth < self.k)
            aggregate = is_dia & ~spent & (d_depth >= self.k)
            other = alive & ~is_dia

            s1 = sampler.next_1d(alive)
            s2 = sampler.next_2d(alive)

            # --- everything not special-cased: its own BSDF ------------
            bs_o, w_o = si.bsdf(ray).sample(ctx, si, s1, s2, other)
            wo_local = mi.Vector3f(bs_o.wo)
            weight = mi.Spectrum(w_o)
            pdf_any = mi.Float(bs_o.pdf)

            if self.dia_bsdf is not None:
                # --- the stone below k: a real dielectric --------------
                wo_e, _eta_e, pdf_e, _type_e, w_e = self.dia_bsdf.sample_entry(
                    ctx, si, s1, explicit)
                wo_local = dr.select(explicit, wo_e, wo_local)
                weight = dr.select(explicit, mi.Spectrum(w_e), weight)
                pdf_any = dr.select(explicit, pdf_e, pdf_any)

                if self.use_residual:
                    # --- the stone at k: the aggregate, once -----------
                    wo_a, pdf_a, w_a = self.dia_bsdf.sample_residual(
                        si, s1, s2, aggregate)
                    wo_local = dr.select(aggregate, wo_a, wo_local)
                    weight = dr.select(aggregate, mi.Spectrum(w_a), weight)
                    pdf_any = dr.select(aggregate, pdf_a, pdf_any)

            # A pass-through keeps the direction and loses no energy.
            scatter = alive & ~through
            beta = dr.select(scatter, beta * weight, beta)

            new_dir = dr.select(through, ray.d, si.to_world(wo_local))
            ray = si.spawn_ray(new_dir)

            d_depth = dr.select(is_dia & ~through, d_depth + 1, d_depth)
            spent = spent | aggregate

            # Kill dead paths: zero throughput, or a degenerate sample.
            dead = scatter & ((pdf_any <= 0.0) | (dr.max(beta) <= 0.0))
            done = done | dead

        return L, active, []

    def to_string(self):
        return f"DepthAwarePath[k={self.k}, max_depth={self.max_depth}]"
