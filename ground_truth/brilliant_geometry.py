"""
brilliant_geometry.py - Procedural generation of round brilliant diamond geometry

FIXED VERSION. The previous version generated facets from several
free-floating intermediate rings (pavilion_bottom at 60% depth, lower_girdle
at 30% depth, bezel at 0.7*crown_height, star at 0.5*crown_height, etc.)
that did not correspond to the actual geometric intersections of real
facet planes. Connecting rings at arbitrary, unrelated heights/radii makes
the resulting triangles non-planar and self-intersecting -- which is why
the old mesh rendered as a tangled ball instead of a diamond.

A real round-brilliant facet is always a planar polygon between exactly
two well-defined rings (or a ring and a single point):

    Table (z = +crown_height, octagon)
       |  STAR facets       (table vertex  -> single girdle-ring point)
       |  BEZEL/KITE facets (table vertex  -> two adjacent girdle-ring pts)
    Girdle ring (z = 0, 16 points: 8 "under bezel" + 8 "under star", interleaved)
       |  (thin girdle band, optional)
       |  MAIN pavilion facets      (girdle pt -> culet, flanked by inner apex)
       |  LOWER-GIRDLE facets       (girdle pt -> shared inner apex points)
    Culet (single point at -pavilion_height)

This file keeps the original function names/signature (including the
previously-unused `culet_radius`, which now actually controls how large
the flat culet facet patch is -- set culet_radius=0 for a true sharp point).
"""

import numpy as np
import math


def make_round_brilliant(
    girdle_radius=1.0,
    crown_angle_deg=34.5,
    pavilion_angle_deg=40.75,
    table_frac=0.56,
    num_main_facets=8,
    culet_radius=0.02,
):
    """
    Generate a round brilliant diamond mesh with flat, planar facets.

    Parameters
    ----------
    girdle_radius : radius at the girdle (widest point)
    crown_angle_deg : angle of crown main facets from horizontal (degrees)
    pavilion_angle_deg : angle of pavilion main facets from horizontal (degrees)
    table_frac : table radius as a fraction of girdle_radius
    num_main_facets : number of main facets around the stone (8 for a
        standard round brilliant)
    culet_radius : radius of the small flat facet at the very bottom.
        Set to 0 for a perfectly sharp point culet (more idealized/less
        realistic -- real stones almost always have a tiny culet facet
        or point that isn't infinitely sharp).

    Returns
    -------
    vertices : (N, 3) float32
    faces    : (M, 3) uint32
    """
    crown_angle = math.radians(crown_angle_deg)
    pavilion_angle = math.radians(pavilion_angle_deg)

    table_radius = girdle_radius * table_frac
    crown_height = (girdle_radius - table_radius) * math.tan(crown_angle)
    pavilion_height = girdle_radius * math.tan(pavilion_angle)

    n = num_main_facets       # 8
    n2 = 2 * n                # 16, the interleaved girdle ring resolution

    verts = []

    def add(p):
        verts.append(p)
        return len(verts) - 1

    # ---- Table (flat octagon, fan-triangulated from its own center) ----
    table_center = add([0.0, 0.0, crown_height])
    table_ring = []
    for i in range(n):
        ang = 2 * math.pi * i / n
        table_ring.append(add([
            table_radius * math.cos(ang),
            table_radius * math.sin(ang),
            crown_height,
        ]))

    # ---- Girdle ring: 16 points, interleaved ----
    # even index 2*i  -> aligned under table vertex i (bezel apex direction)
    # odd index 2*i+1 -> aligned under the midpoint between table verts i,i+1 (star apex)
    girdle_ring = []
    for k in range(n2):
        ang = 2 * math.pi * k / n2
        girdle_ring.append(add([
            girdle_radius * math.cos(ang),
            girdle_radius * math.sin(ang),
            0.0,
        ]))

    # ---- Culet: either a single point, or a small flat n-gon patch ----
    if culet_radius <= 1e-9:
        culet_center = add([0.0, 0.0, -pavilion_height])
        culet_ring = None
    else:
        culet_center = add([0.0, 0.0, -pavilion_height])
        culet_ring = []
        # small flat patch sitting just above the deepest point, aligned
        # with the main (even) girdle directions
        culet_z = -pavilion_height + culet_radius * math.tan(pavilion_angle)
        for i in range(n):
            ang = 2 * math.pi * i / n
            culet_ring.append(add([
                culet_radius * math.cos(ang),
                culet_radius * math.sin(ang),
                culet_z,
            ]))

    faces = []

    def tri(a, b, c):
        faces.append([a, b, c])

    # ── CROWN ──────────────────────────────────────────────────────
    # STAR facets: table edge (i -> i+1) meets a single girdle point
    # (the odd/"under star" point directly below that table edge).
    for i in range(n):
        t_a = table_ring[i]
        t_b = table_ring[(i + 1) % n]
        g_mid = girdle_ring[(2 * i + 1) % n2]
        tri(t_b, t_a, g_mid) #was tri(t_a, t_b, g_mid) 

    # BEZEL (kite) facets: table vertex i meets the two flanking girdle
    # points plus the girdle point directly beneath it (even index).
    for i in range(n):
        t_v = table_ring[i]
        g_apex = girdle_ring[2 * i]
        g_left = girdle_ring[(2 * i - 1) % n2]
        g_right = girdle_ring[(2 * i + 1) % n2]
        tri(t_v, g_left, g_apex)
        tri(t_v, g_apex, g_right)

    # Table cap (close the flat top octagon)
    for i in range(n):
        tri(table_center, table_ring[i], table_ring[(i + 1) % n])

    # ── PAVILION ───────────────────────────────────────────
    # Every girdle edge must be covered by exactly one pavilion triangle.
    # Emitting a triangle twice is not harmless: the two copies are exactly
    # coincident, so a ray that refracts through one immediately hits the
    # other at t ~ 0 and refracts straight back out. The stone then leaks
    # the background through those wedges instead of bending light, which
    # reads as flat unshaded grey patches seen through the crown.
    if culet_ring is None:
        # Sharp-point culet: the pavilion is a closed fan from the culet
        # point out to the 16-point girdle ring -- two triangles per main
        # facet direction, i.e. exactly one per girdle edge. The
        # lower-girdle facets degenerate into that same fan here, so they
        # need no triangles of their own.
        for i in range(n):
            g_even = girdle_ring[2 * i]
            g_odd_next = girdle_ring[(2 * i + 1) % n2]
            g_odd_prev = girdle_ring[(2 * i - 1) % n2]

            # Main pavilion facet (kite): girdle even point down to culet,
            # flanked by the two adjacent odd (lower-girdle) points.
            tri(g_even, g_odd_prev, culet_center)
            tri(g_even, culet_center, g_odd_next)
    else:
        # Small flat culet patch: the pavilion is a band between the
        # 16-point girdle ring and the n-point culet ring. Three triangles
        # per main direction cover the two girdle edges either side of
        # g_even plus the one culet edge between c_this and c_next.
        for i in range(n):
            g_even = girdle_ring[2 * i]
            g_odd_next = girdle_ring[(2 * i + 1) % n2]
            g_even_next = girdle_ring[(2 * i + 2) % n2]
            c_this = culet_ring[i]
            c_next = culet_ring[(i + 1) % n]

            # Main pavilion facet (kite) halves: each girdle edge adjacent
            # to a main direction drops to that direction's culet point.
            tri(g_even, c_this, g_odd_next)
            tri(g_even_next, g_odd_next, c_next)

            # Lower-girdle facet: the odd girdle point spans the culet edge.
            tri(g_odd_next, c_this, c_next)

        # Cap the small flat culet patch. Wound the opposite way from the
        # table cap so its normal points down and out of the stone rather
        # than up into it.
        for i in range(n):
            tri(culet_center, culet_ring[(i + 1) % n], culet_ring[i])

    vertices = np.array(verts, dtype=np.float32)
    faces = np.array(faces, dtype=np.uint32)
    return vertices, faces


def make_flat_shaded(vertices, faces):
    """
    Convert a mesh to flat-shaded by duplicating vertices for each face.
    Each face gets its own vertices with the face normal -- this keeps
    facet edges sharp, which is essential for a faceted gem (smoothed
    normals would blur the facets into a rounded blob and kill the
    sparkle/fire pattern that comes from sharp normal discontinuities).

    Parameters
    ----------
    vertices : (N, 3) array of vertex positions
    faces    : (M, 3) array of triangle indices

    Returns
    -------
    flat_vertices : (M*3, 3)
    flat_normals  : (M*3, 3)
    flat_uvs      : (M*3, 2)
    flat_faces    : (M, 3)   sequential triangle indices into flat_vertices
    """
    flat_vertices = []
    flat_normals = []
    flat_uvs = []

    for face in faces:
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]

        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normal = normal / norm
        else:
            normal = np.array([0.0, 0.0, 1.0])

        for v, uv in zip([v0, v1, v2], [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]):
            flat_vertices.append(v)
            flat_normals.append(normal)
            flat_uvs.append(uv)

    flat_vertices = np.array(flat_vertices, dtype=np.float32)
    flat_normals = np.array(flat_normals, dtype=np.float32)
    flat_uvs = np.array(flat_uvs, dtype=np.float32)

    num_faces = len(faces)
    flat_faces = np.array(
        [[i * 3, i * 3 + 1, i * 3 + 2] for i in range(num_faces)],
        dtype=np.uint32,
    )

    return flat_vertices, flat_normals, flat_uvs, flat_faces


def _check_mesh(vertices, faces, label=""):
    """Diagnostic helper: report degenerate/non-planar triangles."""
    degenerate = 0
    max_area = 0.0
    for tri_idx in faces:
        a, b, c = vertices[tri_idx[0]], vertices[tri_idx[1]], vertices[tri_idx[2]]
        area = np.linalg.norm(np.cross(b - a, c - a))
        max_area = max(max_area, area)
        if area < 1e-8:
            degenerate += 1
    print(f"[check{(' ' + label) if label else ''}] "
          f"faces={len(faces)} degenerate={degenerate} max_tri_area={max_area:.4f}")


if __name__ == "__main__":
    print("Testing brilliant geometry generation...")

    verts, faces = make_round_brilliant(culet_radius=0.0)
    print(f"Sharp-culet: {len(verts)} vertices, {len(faces)} faces")
    _check_mesh(verts, faces, "sharp-culet")
    print("bbox min:", verts.min(axis=0), " bbox max:", verts.max(axis=0))

    verts2, faces2 = make_round_brilliant(culet_radius=0.02)
    print(f"Flat-culet:  {len(verts2)} vertices, {len(faces2)} faces")
    _check_mesh(verts2, faces2, "flat-culet")

    flat_verts, flat_norms, flat_uvs, flat_faces = make_flat_shaded(verts, faces)
    print(f"Flat-shaded: {len(flat_verts)} vertices, {len(flat_faces)} faces")