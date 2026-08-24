
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

def compute_histogram_4d(omega_i, omega_o, outputs, theta_bins=180, phi_bins=360, sampling_method='uniform'):
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
    
    # Get solid angles for incoming directions (θ_i only goes to 90°)
    solid_angles_incoming = solid_angles[:theta_bins//2, :]

    histogram = mi.TensorXf(s, shape)
    histogram_count = mi.TensorXf(count_i, count_i_shape)
    
    # Divide by count per incoming direction (average throughput per sample)
    histogram = histogram / histogram_count[:, :, None, None, None]
    
    # Apply correction based on sampling method
    if sampling_method == 'uniform':
        # Uniform sphere sampling: p(ω_i) = solid_angle_incoming / (4π)
        # Need to divide by p(ω_i) = multiply by 4π / solid_angle_incoming
        histogram = histogram * (4.0 * np.pi) / mi.TensorXf(solid_angles_incoming)[:, :, None, None, None]
    elif sampling_method == 'cos_theta':
        # Uniform in cos(θ) sampling: p(ω_i) = 1/(2π) (independent of θ)
        # PDF = 1/(2π) for all ω_i
        histogram = histogram * (2.0 * np.pi)
    elif sampling_method == 'stratified':
        # Stratified sampling: we sample uniformly within each bin
        # For bin (θ_bin, φ_bin): p(ω_i) = 1/(ΔΩ_bin) for ω_i in that bin
        # where ΔΩ_bin = solid angle of the bin
        # So we need to multiply by ΔΩ_bin
        # We'll approximate by using the average solid angle of the bin
        bin_solid_angles = np.zeros((theta_bins//2, phi_bins))
        theta_edges = np.linspace(0, np.pi/2, theta_bins//2 + 1)
        phi_edges = np.linspace(0, 2*np.pi, phi_bins + 1)
        for i in range(theta_bins//2):
            for j in range(phi_bins):
                delta_cos = np.cos(theta_edges[i]) - np.cos(theta_edges[i+1])
                delta_phi = phi_edges[j+1] - phi_edges[j]
                bin_solid_angles[i, j] = delta_cos * delta_phi
        histogram = histogram * mi.TensorXf(bin_solid_angles)[:, :, None, None, None]
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")
    
    # Divide by outgoing solid angle to get radiance per unit solid angle
    histogram = histogram / mi.TensorXf(solid_angles)[None, None, :, :, None]
    
    histogram = dr.select(dr.isnan(histogram) | dr.isinf(histogram), 0.0, histogram)

    return histogram, histogram_count


# ─────────────────────────────────────────────
# Top-level RDM collection
# ─────────────────────────────────────────────

def sample_directions_uniform_on_sphere(num_samples):
    """Sample directions uniformly on the sphere (current method)."""
    if _HAS_TORCH:
        dirs = torch.randn(num_samples, 3, device=device)
        dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        dirs_np = dirs.cpu().numpy()
    else:
        dirs_np = np.random.randn(num_samples, 3)
        dirs_np /= np.linalg.norm(dirs_np, axis=-1, keepdims=True)
    return dirs_np

def sample_directions_uniform_in_cos_theta(num_samples):
    """
    Sample directions with uniform distribution in cos(θ).
    This gives more samples near grazing angles to compensate for smaller solid angles.
    
    For uniform sampling in cos(θ):
    - Sample u1, u2 uniformly in [0, 1]
    - θ = acos(1 - 2*u1)  [gives uniform cos(θ) in [-1, 1]]
    - φ = 2π * u2
    - x = sin(θ)cos(φ), y = sin(θ)sin(φ), z = cos(θ)
    """
    if _HAS_TORCH:
        u1 = torch.rand(num_samples, device=device)
        u2 = torch.rand(num_samples, device=device)
        theta = torch.acos(1 - 2 * u1)
        phi = 2 * torch.pi * u2
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)
        dirs = torch.stack([x, y, z], dim=-1)
        dirs_np = dirs.cpu().numpy()
    else:
        u1 = np.random.rand(num_samples)
        u2 = np.random.rand(num_samples)
        theta = np.arccos(1 - 2 * u1)
        phi = 2 * np.pi * u2
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        dirs_np = np.stack([x, y, z], axis=-1)
    return dirs_np

def sample_directions_stratified(num_samples, theta_bins=8, phi_bins=16):
    """
    Stratified sampling to ensure all direction bins get some samples.
    Samples are distributed evenly across θ_i bins (0-90°) and φ bins.
    """
    samples_per_bin = max(1, num_samples // (theta_bins * phi_bins))
    total_samples = samples_per_bin * theta_bins * phi_bins
    
    dirs_list = []
    for theta_idx in range(theta_bins):
        for phi_idx in range(phi_bins):
            # Sample within this bin
            theta_start = (theta_idx / theta_bins) * (np.pi / 2)
            theta_end = ((theta_idx + 1) / theta_bins) * (np.pi / 2)
            phi_start = (phi_idx / phi_bins) * 2 * np.pi
            phi_end = ((phi_idx + 1) / phi_bins) * 2 * np.pi
            
            # Use uniform in cos(θ) within the bin for better grazing angle coverage
            u1 = np.random.rand(samples_per_bin)
            u2 = np.random.rand(samples_per_bin)
            
            # Transform to uniform in cos(θ) within bin limits
            cos_theta_start = np.cos(theta_end)  # Note: cos is decreasing function
            cos_theta_end = np.cos(theta_start)
            cos_theta = cos_theta_start + (cos_theta_end - cos_theta_start) * u1
            theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            phi = phi_start + (phi_end - phi_start) * u2
            
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            
            dirs_list.append(np.stack([x, y, z], axis=-1))
    
    dirs_np = np.concatenate(dirs_list, axis=0)
    # If we have extra capacity, add some uniform samples
    if total_samples < num_samples:
        extra = sample_directions_uniform_in_cos_theta(num_samples - total_samples)
        dirs_np = np.concatenate([dirs_np, extra], axis=0)
    
    return dirs_np

def collect_rdm(scene, bounding_radius, num_samples=1024 * 1024 * 16,
                 theta_bins=180, phi_bins=360, max_depth=4096,
                 sampling_method='stratified'):
    """
    Fire `num_samples` rays inward from random directions on a sphere that
    safely encloses the diamond, trace them through the dielectric, and
    bin the result into transmittance / reflectance / multiple-scatter
    histograms.
    
    sampling_method: 'uniform' (original), 'cos_theta', or 'stratified'
    """
    # Origins on a sphere of radius >> bounding_radius, aimed at the
    # origin -- comfortably outside the stone so rays start in free space,
    # not touching the girdle.
    origin_radius = 3.0 * bounding_radius

    # Choose sampling method
    if sampling_method == 'uniform':
        dirs_np = sample_directions_uniform_on_sphere(num_samples)
    elif sampling_method == 'cos_theta':
        dirs_np = sample_directions_uniform_in_cos_theta(num_samples)
    elif sampling_method == 'stratified':
        # For stratified sampling, use incoming direction bins (θ_i only goes to 90°)
        dirs_np = sample_directions_stratified(num_samples, theta_bins=theta_bins//2, phi_bins=phi_bins)
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")

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

    select_r = escaped & (depth == 1) & (dr.dot(wo, frame.n) > 0.0)
    select_t = escaped & (depth == 1) & ~select_r
    select_m = escaped & (depth >= 2)

    throughput_t = dr.select(select_t, throughput, 0.0)
    throughput_r = dr.select(select_r, throughput, 0.0)
    throughput_m = dr.select(select_m, throughput, 0.0)

    rdm_t, count_t = compute_histogram_4d(wi, wo, throughput_t, theta_bins, phi_bins, sampling_method)
    rdm_r, count_r = compute_histogram_4d(wi, wo, throughput_r, theta_bins, phi_bins, sampling_method)
    rdm_m, count_m = compute_histogram_4d(wi, wo, throughput_m, theta_bins, phi_bins, sampling_method)

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
    sampling_method='stratified',
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
            scene, bounding_radius, batch_size, theta_bins, phi_bins, max_depth, sampling_method,
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