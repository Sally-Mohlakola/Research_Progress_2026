"""
studio_env.py - the lighting rig used to render the stone.

Why this exists
---------------
The scenes here used to be lit by a bright `{'type': 'constant'}` environment
(radiance 0.55, 0.58, 0.65) plus four area lights. A constant environment is
the one environment in which a transparent object is guaranteed to look
featureless.

Radiance is preserved along a specular path: refraction into the stone
divides it by eta^2 and refraction back out multiplies it by eta^2, and total
internal reflection is lossless. So a camera ray that enters the crown,
bounces around the pavilion any number of times and finally escapes returns
*exactly* the environment radiance it would have returned had the stone not
been there, attenuated only by the Fresnel reflections it lost on the way.
Under a constant environment every escape direction carries the same value,
so all of those paths return the same number and the facets render as flat
unshaded grey. The kaleidoscope was being traced correctly the whole time;
there was simply nothing for it to be a kaleidoscope *of*.

A diamond only reads as a diamond in front of an environment with strong
angular structure, because each facet steers the eye toward a different part
of it. That is also what the stone's fire needs: dispersion spreads a path's
wavelengths across a small angular fan, so the colours only separate visibly
when that fan straddles an edge between a bright source and a dark surround.
Against a uniform white sky every wavelength picks up the same radiance and
the spectrum recombines to grey.

The rig
-------
The structure therefore comes from a set of area lights of deliberately
different angular sizes, over an ambient term left just bright enough to keep
the stone from being rendered in a void:

  - two broad, dim panels that carry the overall brightness and the soft
    gradients across the crown;
  - three mid-size panels of different colour temperature, which give the
    facets something to differ about, plus two more that close the azimuthal
    gaps between them (see the comment on `_PANELS`);
  - two small, very intense sparks. These are what produce both the sparkle
    and the fire -- a small source against a dark surround is the sharp
    bright/dark angular edge that a dispersed fan has to straddle before its
    colours become visible.

Each panel then carries a texture rather than a single radiance, because a
panel of constant radiance is its own way of flattening a facet: a facet is a
mirror, it reflects only a small part of one panel, and if that part is
uniform the whole facet returns one number. See "Panel textures" below.

Why area lights rather than an environment map
----------------------------------------------
An HDR `envmap` is the more natural way to express this and gives smoother
gradients. It is not used here because it adds a second emitter type to
Mitsuba's megakernel -- envmap sampling brings a 2D hierarchical distribution
and a texture lookup that the area-light branch does not need -- and under
`llvm_ad_spectral` with LLVM 18 that reliably fails to JIT at practical film
resolutions ("Failed to materialize symbols", then a segfault) while the same
scene renders fine with area lights only. Every emitter below is the type
already compiled into the kernel, so the rig costs essentially nothing extra.
`studio_envmap_dict()` is kept at the bottom for anyone who wants to try the
envmap route on a different backend.

Defining the rig here rather than inline in each scene keeps `eval.py` and
`ground_truth/diamond.py` lit identically, which matters because the neural
and analytic renders are compared to each other.
"""

import numpy as np
import mitsuba as mi


# Ambient fill -- the floor under everything, and the single knob that decides
# how much of the stone renders as black.
#
# A facet goes black when the paths leaving it escape into a direction where
# there is nothing: the panels cover only a small part of the sphere, so most
# escape directions see only this term. At the previous value of
# (0.045, 0.050, 0.062) that meant 6e-5 of the rig's peak, or 0.003 after
# display exposure -- below the black point, so those facets read as unshaded
# black rather than dark. Measured on round_diamond_gia at 192px/16spp,
# max_depth 64, as (share of the stone below 0.02, weighted within-facet
# gradient sigma/mu):
#
#     x1  (0.045)   30.7% black   grad 1.167     <- previous value
#     x4  (0.18 )   24.5% black   grad 1.118
#     x8  (0.36 )    4.4% black   grad 1.061     <- here
#     x16 (0.72 )    0.5% black   grad 0.967
#
# x8 is the knee: it removes 26 points of black for 9% of the facet gradient,
# where x16 spends another 9% to remove the last 4 points. Adding a ring of
# eight dim panels *below* the girdle instead was measured at 24.9% black and
# grad 1.114 -- indistinguishable from simply going to x4, for eight more
# emitters, so it is not worth doing.
#
# This is roughly two thirds of the (0.55, 0.58, 0.65) constant that produced
# the original flat-grey stone, which sounds alarming and is not: back then
# the constant was effectively the only source. The panels now reach 751 in
# these units, so this term sits about 2000x below the peak and acts as a
# floor rather than a wash -- which is what the measured gradient above shows.
AMBIENT_RADIANCE = (0.36, 0.40, 0.50)

# (name, origin, up, size, radiance). Every panel is aimed at the origin,
# which is where the stone sits.
#
# Azimuths (atan2(y, x), degrees) are chosen to keep no gap between
# neighbouring sources much above 45 deg. That number isn't arbitrary: a
# round brilliant has 8 main facet directions spaced every 45 deg, so any
# azimuthal gap wider than that is large enough for a whole facet's
# reflection/refraction cone to fall entirely between sources and see only
# the flat ambient term -- which is exactly what happened to the bezel
# facets before `rim_light_2` and `side_light_2` were added below (the old
# rig had a ~90 deg gap between spark_warm/54deg and rim_light/144deg, and a
# ~75 deg gap between fill_light/-124deg and key_light/-49deg).
_PANELS = [
    # --- broad, dim: overall brightness and soft gradients ----------------
    ('top_light',   (  0.0,  0.0,  6.0), (0.0, 1.0, 0.0), 2.0, ( 12.0,  12.0,  13.0)),
    ('side_light',  (  4.2,  1.0,  2.0), (0.0, 0.0, 1.0), 2.2, (  7.0,   7.0,   8.0)),  # az  13 deg

    # --- mid-size, varied colour temperature ------------------------------
    ('key_light',   (  3.0, -3.5,  5.0), (0.0, 0.0, 1.0), 1.2, ( 40.0,  38.0,  35.0)),  # az -49 deg
    ('rim_light',   ( -3.5,  2.5,  1.5), (0.0, 0.0, 1.0), 1.0, ( 10.0,  14.0,  22.0)),  # az 144 deg
    ('fill_light',  ( -2.0, -3.0,  2.0), (0.0, 0.0, 1.0), 1.0, (  8.0,  10.0,  12.0)),  # az -124 deg

    # closes the fill_light -> key_light gap (-124 to -49, 75 deg)
    ('side_light_2', (  0.23, -4.31,  2.0), (0.0, 0.0, 1.0), 1.0, (  9.0,   9.0,  10.0)),  # az -87 deg

    # closes the spark_warm -> rim_light gap (54 to 144, 90 deg)
    ('rim_light_2', ( -0.67,  4.25,  1.5), (0.0, 0.0, 1.0), 1.0, ( 12.0,  16.0,  20.0)),  # az  99 deg

    # --- small and intense: sparkle and fire ------------------------------
    ('spark_warm',  (  2.0,  2.8,  3.2), (0.0, 0.0, 1.0), 0.32, (240.0, 224.0, 196.0)),  # az  54 deg
    ('spark_cool',  ( -2.6, -1.6,  3.6), (0.0, 0.0, 1.0), 0.28, (130.0, 152.0, 230.0)),  # az -148 deg
]


# ---------------------------------------------------------------------------
# Panel textures
# ---------------------------------------------------------------------------
# A panel of *constant* radiance is a second way to lose the facets, and it
# survived the fix above. A polished facet is a mirror: the region of a panel
# that one facet reflects toward the camera is small, so if the panel is
# uniform, every pixel of that facet returns the same radiance and the facet
# renders as one flat polygon -- a white one this time rather than a grey one.
# The facet outlines were visible but nothing varied *inside* them.
#
# The cure is to give each panel angular structure of its own, which is what
# an environment map would have supplied. `area` emitters accept a texture for
# their radiance, and a `bitmap` texture keeps the same emitter type that is
# already compiled into the kernel -- so this buys the structure without
# reintroducing the `envmap` plugin that fails to JIT under `llvm_ad_spectral`
# (see the module docstring).
#
# Each texture carries three things:
#   * a broad linear ramp, which is what puts a smooth gradient across a facet
#     that mirrors a large part of one panel;
#   * a few soft Gaussian blobs at the scale a facet actually resolves, so
#     neighbouring facets looking at nearby patches disagree;
#   * a soft edge falloff, so a facet whose reflection straddles the rim of a
#     panel gets a gradient instead of a hard-edged white polygon.
#
# Every texture is normalised to unit mean *after* the falloff is applied, so
# each panel radiates exactly the power it did when it was uniform. Only the
# distribution changes, which keeps overall exposure comparable with renders
# made before this change.

_PANEL_TEXTURE_RES = 96

# name -> (seed, n_blobs, blob_sigma, contrast, edge_width, hue_spread)
#   contrast   : amplitude of the variation about the panel mean; 1.0 puts the
#                texture roughly in [0.6, 1.4].
#   edge_width : width of the soft border as a fraction of the panel, so the
#                sparks (0.45) are effectively soft discs while the broad
#                panels (~0.22) keep most of their area at full strength.
#   hue_spread : how far the colour temperature drifts across the panel.
_PANEL_TEXTURE = {
    'top_light':    (11, 5, 0.30, 1.15, 0.15, 0.10),
    'side_light':   (12, 4, 0.35, 1.15, 0.16, 0.12),
    'key_light':    (13, 3, 0.22, 1.05, 0.14, 0.08),
    'rim_light':    (14, 3, 0.25, 1.05, 0.15, 0.15),
    'fill_light':   (15, 4, 0.28, 1.05, 0.16, 0.12),
    'side_light_2': (16, 3, 0.28, 1.05, 0.16, 0.10),
    'rim_light_2':  (17, 3, 0.26, 1.05, 0.15, 0.14),
    'spark_warm':   (18, 1, 0.40, 0.50, 0.40, 0.05),
    'spark_cool':   (19, 1, 0.40, 0.50, 0.40, 0.05),
}

_DEFAULT_TEXTURE = (0, 3, 0.28, 1.00, 0.24, 0.10)

_TEXTURE_CACHE = {}


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _panel_texture(name, res=None):
    """Unit-mean RGB radiance pattern for one panel, as an (res, res, 3) array."""
    res = _PANEL_TEXTURE_RES if res is None else res
    key = (name, res)
    if key in _TEXTURE_CACHE:
        return _TEXTURE_CACHE[key]

    seed, n_blobs, sigma, contrast, edge, hue = _PANEL_TEXTURE.get(
        name, _DEFAULT_TEXTURE)
    rng = np.random.default_rng(seed)

    y, x = np.mgrid[0:res, 0:res] / (res - 1.0)

    # Broad ramp in a random direction: the large-scale gradient picked up by
    # a facet that mirrors most of the panel at once.
    ang = rng.uniform(0.0, 2.0 * np.pi)
    field = 0.8 * (np.cos(ang) * (x - 0.5) + np.sin(ang) * (y - 0.5) + 0.5)

    # Blobs, deliberately allowed to sit outside the panel so that some of
    # them enter only as a partial gradient rather than a complete disc.
    for _ in range(n_blobs):
        cx, cy = rng.uniform(-0.2, 1.2, 2)
        s = sigma * rng.uniform(0.6, 1.6)
        field += rng.uniform(0.4, 1.0) * np.exp(
            -((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * s * s))

    field -= field.min()
    field /= max(field.max(), 1e-9)
    lum = 1.0 + contrast * (field - field.mean())

    # Colour temperature drifts along a second, independent direction, so the
    # warm and cool halves of a panel are not aligned with its bright half.
    ang2 = rng.uniform(0.0, 2.0 * np.pi)
    tint = np.cos(ang2) * (x - 0.5) + np.sin(ang2) * (y - 0.5)
    tint = tint / max(np.abs(tint).max(), 1e-9)
    rgb = lum[..., None] * (
        1.0 + hue * tint[..., None] * np.array([1.0, 0.0, -1.0]))

    # Soft border, separable so the panel stays rectangular.
    falloff = (_smoothstep(np.minimum(x, 1.0 - x) / edge)
               * _smoothstep(np.minimum(y, 1.0 - y) / edge))
    rgb *= falloff[..., None]

    rgb = np.maximum(rgb, 0.0)
    rgb /= max(rgb.mean(), 1e-9)          # unit mean -> unchanged panel power

    rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    _TEXTURE_CACHE[key] = rgb
    return rgb


# ---------------------------------------------------------------------------
# Radiance units
# ---------------------------------------------------------------------------
# Under a spectral variant, a `bitmap` texture is fitted to the sRGB
# spectral-upsampling model, and that model is defined on [0, 1]: every texel
# above 1.0 comes back as exactly 1.0, silently. (An `rgb` texture inside an
# emitter does not have this problem -- emitters get the unbounded variant --
# which is why the flat panels below can carry a radiance of 240 and the
# textured ones cannot.) `bitmap` has no `scale` property and `area` has no
# `scale` property either, so there is nowhere to put the magnitude back.
#
# The whole rig is therefore emitted in units where 1.0 is the brightest
# radiance anywhere in it. That is a pure global scale on every light -- the
# rendered image is identical up to one constant factor -- and the divisor is
# *computed* from the panels rather than hardcoded, so raising a panel's
# radiance in `_PANELS` rescales the rig instead of quietly clipping it.
#
# `_PANELS` and `AMBIENT_RADIANCE` above stay in their own readable units, and
# `display_exposure()` converts the exposure that suits them into the exposure
# that suits the normalised output.

# Exposure that puts the stone in midtone, in the authored units of `_PANELS`.
# Measured on the analytic round brilliant: it leaves ~2% of the stone clipped
# and ~2/3 of it in midtone.
NOMINAL_EXPOSURE = 0.06

_RIG_PEAK = None


def rig_peak():
    """Brightest radiance in the rig, in the authored units of `_PANELS`."""
    global _RIG_PEAK
    if _RIG_PEAK is None:
        peak = max(AMBIENT_RADIANCE)
        for name, _origin, _up, _size, radiance in _PANELS:
            # Always measured on the textured panels, so that `textured=True`
            # and `textured=False` are scaled identically and stay directly
            # comparable as a before/after pair.
            peak = max(peak, float((_panel_texture(name)
                                    * np.asarray(radiance)).max()))
        _RIG_PEAK = max(peak, 1e-9)
    return _RIG_PEAK


def display_exposure(nominal=None):
    """The `--exposure` value matching `nominal` in the normalised units."""
    return (NOMINAL_EXPOSURE if nominal is None else nominal) * rig_peak()


def _panel_radiance(name, radiance, unit, textured):
    """The `radiance` entry for one panel's area emitter."""
    if not textured:
        return {'type': 'rgb', 'value': [c * unit for c in radiance]}

    tex = _panel_texture(name) * np.asarray(radiance, dtype=np.float32) * unit
    return {
        'type': 'bitmap',
        'bitmap': mi.Bitmap(np.ascontiguousarray(tex, dtype=np.float32)),
        'filter_type': 'bilinear',
        'wrap_mode': 'clamp',
    }


def studio_lighting(scale=1.0, ambient=None, textured=True):
    """
    Scene-dictionary entries for the whole rig: the ambient term and every
    panel. Merge into a scene dict with `**studio_lighting()`.

    Radiance is emitted in the normalised units described above, so the frames
    this rig produces want an exposure of `display_exposure()` rather than 1.

    Parameters
    ----------
    scale : multiplies every emitter, ambient included, so overall exposure
        can be trimmed without changing the character of the lighting.
    ambient : override the ambient radiance triple. Raising it back toward
        (0.55, 0.58, 0.65) reproduces the old flat-grey look, which makes it
        a convenient control for a figure demonstrating the effect.
    textured : give every panel the structured radiance described above.
        `textured=False` restores flat-radiance panels of identical power,
        which is the matching control for a figure about facet gradients --
        the two rigs then differ only in how each panel distributes what it
        emits.
    """
    unit = scale / rig_peak()
    amb = np.array(AMBIENT_RADIANCE if ambient is None else ambient) * unit

    entries = {
        'ambient': {
            'type': 'constant',
            'radiance': {'type': 'rgb', 'value': amb.tolist()},
        },
    }

    for name, origin, up, size, radiance in _PANELS:
        entries[name] = {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.look_at(
                origin=list(origin), target=[0.0, 0.0, 0.0], up=list(up),
            ).scale([size, size, 1.0]),
            'emitter': {
                'type': 'area',
                'radiance': _panel_radiance(name, radiance, unit, textured),
            },
        }

    return entries


# ---------------------------------------------------------------------------
# Optional environment-map route. Not used by default -- see the module
# docstring on why it does not survive the LLVM JIT here -- but kept because
# it is the cleaner formulation and works under the scalar and CUDA variants.
# ---------------------------------------------------------------------------

# (polar angle from up, azimuth, angular radius, peak radiance), degrees.
_SOURCES = [
    (22.0,   50.0, 26.0, ( 18.0,  18.0,  19.0)),
    (48.0,  205.0,  7.0, ( 90.0,  86.0,  78.0)),
    (63.0,  315.0,  5.0, ( 70.0,  74.0,  90.0)),
    (35.0,  140.0,  9.0, ( 46.0,  48.0,  52.0)),
    (72.0,   95.0,  4.0, (120.0, 112.0,  96.0)),
    (80.0,  260.0, 34.0, (  3.0,   3.4,   4.2)),
]

_ZENITH_COLOUR = (0.26, 0.33, 0.46)
_HORIZON_COLOUR = (0.42, 0.44, 0.48)
_FLOOR_COLOUR = (0.05, 0.05, 0.06)


def studio_environment(width=1024, height=512, mean_radiance=0.60):
    """Equirectangular HDR environment as an `mi.Bitmap`, same rig in spirit."""
    theta = np.pi * (np.arange(height, dtype=np.float64) + 0.5) / height
    phi = 2.0 * np.pi * (np.arange(width, dtype=np.float64) + 0.5) / width
    th, ph = np.meshgrid(theta, phi, indexing='ij')

    up = np.cos(th)
    sin_t = np.sin(th)
    dirs = np.stack([sin_t * np.cos(ph), up, sin_t * np.sin(ph)], axis=-1)

    t = np.clip(up, 0.0, 1.0)[..., None]
    sky = np.array(_HORIZON_COLOUR) * (1.0 - t) + np.array(_ZENITH_COLOUR) * t
    floor = np.broadcast_to(np.array(_FLOOR_COLOUR), sky.shape)
    # Smooth the horizon over a few degrees so it does not read as a hard ring
    # reflected in every pavilion facet.
    blend = np.clip((up + 0.06) / 0.12, 0.0, 1.0)[..., None]
    blend = blend * blend * (3.0 - 2.0 * blend)
    img = floor * (1.0 - blend) + sky * blend

    for src_theta, src_phi, radius, colour in _SOURCES:
        st, sp = np.radians(src_theta), np.radians(src_phi)
        d = np.array([np.sin(st) * np.cos(sp), np.cos(st), np.sin(st) * np.sin(sp)])
        ang = np.arccos(np.clip(dirs @ d, -1.0, 1.0))
        r = np.radians(radius)
        w = np.clip((r - ang) / (0.35 * r), 0.0, 1.0)
        w = w * w * (3.0 - 2.0 * w)
        img += w[..., None] * np.array(colour)

    # Solid angle per texel goes as sin(theta), so the mean must be weighted.
    weight = np.sin(th)[..., None]
    current = float((img * weight).sum() / (weight.sum() * 3.0))
    img *= mean_radiance / max(current, 1e-9)

    return mi.Bitmap(np.ascontiguousarray(img, dtype=np.float32))


def env_to_world():
    """Rotation placing the environment's up axis on world +Z.

    Mitsuba's `envmap` puts the top row of the image along its local +Y, and
    every scene in this project is Z-up (the table faces +Z).
    """
    return mi.ScalarTransform4f.rotate([1, 0, 0], 90)


def studio_envmap_dict(scale=1.0, **kwargs):
    """The `envmap` scene entry, for backends that can compile it."""
    return {
        'type': 'envmap',
        'bitmap': studio_environment(**kwargs),
        'to_world': env_to_world(),
        'scale': scale,
    }