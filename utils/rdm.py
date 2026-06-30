
from config import device, variant

import mitsuba as mi
import drjit as dr
mi.set_variant(variant)

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
import numpy as np
import os

from ground_truth.brilliant_geometry import make_round_brilliant, make_flat_shaded


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def ravel_index(indices, shape):
    """
    Ravel a list of integer index arrays (Dr.Jit UInt32, one per axis) into
    a single flat index, using row-major (C) order -- i.e. the same
    convention numpy.ravel_multi_index defaults to, and the convention
    `dr.zeros(mi.Float, dr.prod(shape))` implies for a flat buffer meant to
    be reshaped into `shape`.
    """
    flat = mi.UInt32(0)
    stride = 1
    for idx, dim in zip(reversed(indices), reversed(shape)):
        flat += mi.UInt32(idx) * mi.UInt32(stride)
        stride *= dim
    return flat


def solid_angle_grid(polar, azimuth):
    """
    Per-bin solid angle dOmega = sin(theta) dtheta dphi, for a grid with
    `polar` bins over the polar/colatitude angle theta in [0, pi] and
    `azimuth` bins over the azimuthal angle phi in [0, 2*pi].

    BUG FIX: this previously had the two ranges swapped (polar/theta
    spanning [0, 2*pi] and azimuth/phi spanning [0, pi]). Once a theta
    bin edge went past pi, cos(theta) started increasing again (cosine
    is only monotonically decreasing on [0, pi]), which made
    `cos(edges[i]) - cos(edges[i+1])` flip sign for the second half of
    the grid -- producing a solid-angle grid with negative entries in
    its bottom half. Any RDM histogram bin divided by one of those
    negative entries (`histogram / solid_angles`) silently flipped sign,
    which is exactly how a strictly non-negative quantity (radiometric
    energy) ended up negative in rdm_m for theta_o > 90 degrees.
    Verified: the corrected grid integrates to exactly 4*pi (the full
    sphere's solid angle); the swapped version did not.
    """
    polar_edges = np.linspace(0, np.pi, polar + 1)
    azimuth_edges = np.linspace(0, 2 * np.pi, azimuth + 1)

    angle_grid = np.zeros((polar, azimuth))
    for i in range(polar):
        delta_cos_polar_angle = np.cos(polar_edges[i]) - np.cos(polar_edges[i + 1])
        for j in range(azimuth):
            delta_azimuth_angle = azimuth_edges[j + 1] - azimuth_edges[j]
            angle_grid[i, j] = delta_cos_polar_angle * delta_azimuth_angle

    polar_bins = 0.5 * (polar_edges[:-1] + polar_edges[1:])
    azimuth_bins = 0.5 * (azimuth_edges[:-1] + azimuth_edges[1:])
    return angle_grid, polar_bins, azimuth_bins


# ─────────────────────────────────────────────
# Scene construction: the diamond as a real Mitsuba mesh
# ─────────────────────────────────────────────

def build_diamond_scene(
    girdle_radius=1.0,
    crown_angle_deg=34.5,
    pavilion_angle_deg=40.75,
    table_frac=0.56,
    num_main_facets=8,
    culet_radius=0.02,
    int_ior=2.419,
    ext_ior=1.000277,
):
    """
    Build a minimal Mitsuba scene containing only the round-brilliant
    diamond mesh, with the physically correct dielectric BSDF attached.
    Returns the scene and the stone's bounding radius (useful for placing
    ray origins safely outside it).
    """
    raw_verts, raw_faces = make_round_brilliant(
        girdle_radius=girdle_radius,
        crown_angle_deg=crown_angle_deg,
        pavilion_angle_deg=pavilion_angle_deg,
        table_frac=table_frac,
        num_main_facets=num_main_facets,
        culet_radius=culet_radius,
    )
    verts, normals, uvs, faces = make_flat_shaded(raw_verts, raw_faces)

    mesh = mi.Mesh(
        "diamond_mesh",
        vertex_count=len(verts),
        face_count=len(faces),
        has_vertex_normals=True,
        has_vertex_texcoords=True,
    )
    mesh_params = mi.traverse(mesh)
    mesh_params["vertex_positions"] = mi.Float(verts.flatten())
    mesh_params["vertex_normals"]   = mi.Float(normals.flatten())
    mesh_params["vertex_texcoords"] = mi.Float(uvs.flatten())
    mesh_params["faces"]            = mi.UInt32(faces.flatten())
    mesh_params.update()

    mesh.set_bsdf(mi.load_dict({
        "type": "dielectric",
        "int_ior": int_ior,
        "ext_ior": ext_ior,
    }))

    scene = mi.load_dict({
        "type": "scene",
        "diamond": mesh,
    })

    bounding_radius = float(np.linalg.norm(raw_verts, axis=1).max())
    return scene, bounding_radius


# ─────────────────────────────────────────────
# Path tracing: bounce rays through the diamond's BSDF
# ─────────────────────────────────────────────

def trace_path(scene, rays, max_depth=10000):
    """
    Trace `rays` through `scene`, bouncing according to the hit BSDF at
    each step, for up to `max_depth` bounces or until a ray exits the
    scene (no further intersection).

    Returns
    -------
    wi          : incoming direction (world space, pointing back toward
                  the ray origin) at the FIRST hit -- this is the true
                  incident illumination direction for the histogram.
    wo          : outgoing direction (world space) of the ray once it
                  escapes the scene (or once max_depth is reached).
    throughput  : accumulated BSDF throughput (mi.Spectrum) along the path.
    depth       : number of bounces taken before escaping / stopping.
    frame       : shading frame at the FIRST hit (used to convert wi/wo to
                  the first hit's local frame for binning).
    escaped     : boolean mask, True for rays that exited the scene
                  (no intersection) before hitting max_depth.
    """
    n_samples = dr.shape(rays.o)[1]
    sampler = mi.load_dict({"type": "independent"})
    sampler.seed(np.random.randint(np.iinfo(np.int32).max), wavefront_size=n_samples)

    si_first = scene.ray_intersect(rays)
    valid_first = si_first.is_valid()

    # True incoming direction at the first hit -- world space, pointing
    # back toward where the ray came from. This is what actually arrived
    # at the surface; it must NOT be replaced by an unrelated sampled
    # direction (that was the bug in the previous version).
    wi_world = -rays.d
    frame_first = mi.Frame3f(si_first.sh_frame)

    throughput = mi.Spectrum(1.0)
    active = mi.Bool(valid_first)
    depth = mi.UInt32(0)
    si = si_first

    state = (rays, si, throughput, active, depth)

    def loop_condition(rays, si, throughput, active, depth):
        return active & (depth < max_depth)

    def loop_body(rays, si, throughput, active, depth):
        ctx = mi.BSDFContext()
        # si.wi is populated automatically by ray_intersect (local shading
        # frame); BSDF sampling reads it directly -- nothing to override.
        bs, bsdf_val = si.bsdf().sample(ctx, si, sampler.next_1d(), sampler.next_2d(), active)

        throughput_new = dr.select(active, throughput * bsdf_val, throughput)
        depth_new = dr.select(active, depth + mi.UInt32(1), depth)

        next_rays = si.spawn_ray(si.to_world(bs.wo))
        next_rays = dr.select(active, next_rays, rays)

        si_new = scene.ray_intersect(next_rays, active)
        active_new = active & si_new.is_valid()

        return next_rays, si_new, throughput_new, active_new, depth_new

    rays_final, si_final, throughput, active_remaining, depth = dr.while_loop(
        state, loop_condition, loop_body,
        labels=["rays", "si", "throughput", "active", "depth"],
    )

    # A ray has "escaped" if the loop stopped because it left the scene
    # (no further intersection) rather than because it hit max_depth while
    # still on a valid surface.
    escaped = valid_first & ~si_final.is_valid()

    wo_world = rays_final.d
    wi_local = frame_first.to_local(wi_world)
    wo_local = frame_first.to_local(wo_world)

    print(f"Rays that hit the diamond: {dr.count(valid_first)}")

    return wi_local, wo_local, throughput, depth, frame_first, escaped


# ─────────────────────────────────────────────
# 4D histogram accumulation
# ─────────────────────────────────────────────

def compute_histogram_4d(omega_i, omega_o, outputs, theta_bins=180, phi_bins=360):
    theta_i = dr.acos(dr.clip(omega_i.z, -1.0, 1.0))
    phi_i = dr.atan2(omega_i.y, omega_i.x)
    theta_o = dr.acos(dr.clip(omega_o.z, -1.0, 1.0))
    phi_o = dr.atan2(omega_o.y, omega_o.x)

    theta_i_idx = mi.UInt32(theta_i / (dr.pi / 2) * (theta_bins // 2))
    phi_i_idx = mi.UInt32((phi_i + dr.pi) / dr.two_pi * phi_bins)
    theta_o_idx = mi.UInt32(theta_o / dr.pi * theta_bins)
    phi_o_idx = mi.UInt32((phi_o + dr.pi) / dr.two_pi * phi_bins)

    valid_mask = (
        (theta_i_idx >= 0) & (theta_i_idx < theta_bins // 2)
        & (phi_i_idx >= 0) & (phi_i_idx < phi_bins)
        & (theta_o_idx >= 0) & (theta_o_idx < theta_bins)
        & (phi_o_idx >= 0) & (phi_o_idx < phi_bins)
    )

    # NOTE: `arr[mask]` in Dr.Jit is NOT a NumPy-style boolean-mask
    # compaction -- it returns the array unchanged (verified directly:
    # mi.UInt32([1,2,3,4,5])[mi.Bool([T,F,T,F,T])] has length 5, not 3).
    # The previous version of this function relied on that pattern to
    # "filter out" invalid bin indices before scattering, which silently
    # did nothing -- out-of-range indices were then scattered directly
    # into the histogram buffer, corrupting memory. The correct fix is to
    # pass `active=valid_mask` straight to `scatter_reduce`, which
    # Dr.Jit supports natively and is confirmed safe even when the
    # (unused, inactive-lane) index value itself is out of range.

    shape = (theta_bins // 2, phi_bins, theta_bins, phi_bins, 3)
    s = dr.zeros(mi.Float, dr.prod(shape))

    idx_x = ravel_index([theta_i_idx, phi_i_idx, theta_o_idx, phi_o_idx, mi.UInt32(0)], shape)
    idx_y = ravel_index([theta_i_idx, phi_i_idx, theta_o_idx, phi_o_idx, mi.UInt32(1)], shape)
    idx_z = ravel_index([theta_i_idx, phi_i_idx, theta_o_idx, phi_o_idx, mi.UInt32(2)], shape)

    dr.scatter_reduce(dr.ReduceOp.Add, s, outputs.x, idx_x, active=valid_mask)
    dr.scatter_reduce(dr.ReduceOp.Add, s, outputs.y, idx_y, active=valid_mask)
    dr.scatter_reduce(dr.ReduceOp.Add, s, outputs.z, idx_z, active=valid_mask)

    count_i_shape = (theta_bins // 2, phi_bins)
    count_i = dr.zeros(mi.Float, dr.prod(count_i_shape))
    count_idx = ravel_index([theta_i_idx, phi_i_idx], count_i_shape)
    dr.scatter_reduce(dr.ReduceOp.Add, count_i, mi.Float(1), count_idx, active=valid_mask)

    solid_angles, _, _ = solid_angle_grid(theta_bins, phi_bins)

    histogram = mi.TensorXf(s, shape)
    histogram_count = mi.TensorXf(count_i, count_i_shape)
    histogram = histogram / histogram_count[:, :, None, None, None]
    histogram = histogram / mi.TensorXf(solid_angles)[None, None, :, :, None]
    histogram = dr.select(dr.isnan(histogram) | dr.isinf(histogram), 0.0, histogram)

    return histogram, histogram_count


# ─────────────────────────────────────────────
# Top-level RDM collection
# ─────────────────────────────────────────────

def collect_rdm(scene, bounding_radius, num_samples=1024 * 1024 * 16,
                 theta_bins=180, phi_bins=360, max_depth=4096):
    """
    Fire `num_samples` rays inward from random directions on a sphere that
    safely encloses the diamond, trace them through the dielectric, and
    bin the result into transmittance / reflectance / multiple-scatter
    histograms.
    """
    # Origins on a sphere of radius >> bounding_radius, aimed at the
    # origin -- comfortably outside the stone so rays start in free space,
    # not touching the girdle.
    origin_radius = 3.0 * bounding_radius

    if _HAS_TORCH:
        dirs = torch.randn(num_samples, 3, device=device)
        dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        dirs_np = dirs.cpu().numpy()
    else:
        dirs_np = np.random.randn(num_samples, 3)
        dirs_np /= np.linalg.norm(dirs_np, axis=-1, keepdims=True)

    o = mi.Point3f(
        mi.Float(dirs_np[:, 0]) * origin_radius,
        mi.Float(dirs_np[:, 1]) * origin_radius,
        mi.Float(dirs_np[:, 2]) * origin_radius,
    )
    d = mi.Vector3f(-mi.Float(dirs_np[:, 0]), -mi.Float(dirs_np[:, 1]), -mi.Float(dirs_np[:, 2]))

    ray = mi.Ray3f(o, d)

    wi, wo, throughput, depth, frame, escaped = trace_path(scene, ray, max_depth=max_depth)

    # Rays that never hit the diamond at all (depth==0, e.g. missed
    # entirely from sampling direction noise) carry no information.
    throughput = dr.select(depth < 1, 0.0, throughput)

    not_escaped = int(dr.count(~escaped & (depth >= 1))[0])
    if not_escaped > 0:
        print(f"  [warn] {not_escaped}/{num_samples} rays did not escape "
              f"within max_depth={max_depth} (trapped by total internal reflection)")

    # depth==1: a single bounce off the first surface, exiting back out
    # the same side it entered -> direct reflectance.
    # depth>=2 and escaped: light that passed all the way through and
    # out the scene (any exit direction) -> lumped as transmittance vs.
    # multi-scatter below based on which side of the first surface it
    # exits from, matching the original transmittance/reflectance/
    # multi-scatter split.
    #
    # IMPORTANT: rays that hit max_depth while still trapped inside the
    # stone (escaped == False) never produced a genuine exit direction --
    # their `wo` is just whatever direction the loop happened to be
    # pointing when it gave up, not a real measurement of where light
    # left the diamond. These must be excluded from every selection
    # below, not just tallied in the warning above.
    select_r = escaped & (depth == 1) & (dr.dot(wo, frame.n) > 0.0)
    select_t = escaped & (depth == 1) & ~select_r
    select_m = escaped & (depth >= 2)

    throughput_t = dr.select(select_t, throughput, 0.0)
    throughput_r = dr.select(select_r, throughput, 0.0)
    throughput_m = dr.select(select_m, throughput, 0.0)

    rdm_t, count_t = compute_histogram_4d(wi, wo, throughput_t, theta_bins, phi_bins)
    rdm_r, count_r = compute_histogram_4d(wi, wo, throughput_r, theta_bins, phi_bins)
    rdm_m, count_m = compute_histogram_4d(wi, wo, throughput_m, theta_bins, phi_bins)

    print(f"Total rays traced: {dr.width(wi)}")
   
    print(f"Rays that escaped: {dr.count(escaped)}")
    print(f"Rays classified as T: {dr.count(select_t)}")
    print(f"Rays classified as R: {dr.count(select_r)}")
    print(f"Rays classified as M: {dr.count(select_m)}")

    return rdm_t, rdm_r, rdm_m, count_t, count_r, count_m


def compute_rdm(
    theta_bins=180 // 4,
    phi_bins=360 // 4,
    max_depth=4096,
    num_batches=1024,
    batch_size=1024 * 8,
    diamond_kwargs=None,
):
    """
    Accumulate the RDM over many batches of randomly-directed rays fired
    at the diamond, averaging transmittance/reflectance/multi-scatter
    histograms across batches.
    """
    diamond_kwargs = diamond_kwargs or {}
    scene, bounding_radius = build_diamond_scene(**diamond_kwargs)

    rdm_t = rdm_r = rdm_m = None
    count_t = count_r = count_m = None

    for i in range(num_batches):
        rdm_t_, rdm_r_, rdm_m_, count_t_, count_r_, count_m_ = collect_rdm(
            scene, bounding_radius, batch_size, theta_bins, phi_bins, max_depth,
        )

        dr.eval(rdm_t_, rdm_r_, rdm_m_, count_t_, count_r_, count_m_)

        if i == 0:
            print("Collecting RDM")
            rdm_t, rdm_r, rdm_m = rdm_t_, rdm_r_, rdm_m_
            count_t, count_r, count_m = count_t_, count_r_, count_m_
        else:
            rdm_t += rdm_t_
            rdm_r += rdm_r_
            rdm_m += rdm_m_
            count_t += count_t_
            count_r += count_r_
            count_m += count_m_

        dr.eval(rdm_t, rdm_r, rdm_m, count_t, count_r, count_m)
        dr.flush_malloc_cache()

        print(f"Batch {i + 1}/{num_batches}")

    rdm_t /= num_batches
    rdm_r /= num_batches
    rdm_m /= num_batches

    theta_i_centers = np.linspace(0, np.pi / 2, theta_bins // 2, endpoint=False) + (np.pi / 2 / (theta_bins // 2)) / 2
    phi_i_centers = np.linspace(-np.pi, np.pi, phi_bins, endpoint=False) + (2 * np.pi / phi_bins) / 2
    theta_o_centers = np.linspace(0, np.pi, theta_bins, endpoint=False) + (np.pi / theta_bins) / 2
    phi_o_centers = np.linspace(-np.pi, np.pi, phi_bins, endpoint=False) + (2 * np.pi / phi_bins) / 2

    theta_i, phi_i, theta_o, phi_o = np.meshgrid(
        theta_i_centers, phi_i_centers, theta_o_centers, phi_o_centers, indexing="ij",
    )
    x = np.stack([theta_i, phi_i, theta_o, phi_o], axis=-1)
    x = mi.TensorXf(x)

    solid_angles = mi.TensorXf(solid_angle_grid(theta_bins, phi_bins)[0])

    # After trace_paths()


    return rdm_t, rdm_r, rdm_m, count_t, count_r, count_m, x, solid_angles