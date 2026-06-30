#!/usr/bin/env python3
"""
check_mesh_watertight.py - Check if diamond mesh is watertight (no leaks)

Checks:
1. Non-manifold edges (edges shared by more than 2 faces)
2. Boundary edges (edges shared by only 1 face = holes/leaks)
3. Duplicate vertices
4. Duplicate faces
5. Normal consistency
6. Mesh connectivity
"""

import numpy as np
from collections import defaultdict
from brilliant_geometry import make_round_brilliant, make_flat_shaded


def check_mesh_watertight(vertices, faces, verbose=True):
    """
    Comprehensive mesh watertightness check.
    
    Returns:
        dict: Results with issues found
    """
    results = {
        'is_watertight': True,
        'num_vertices': len(vertices),
        'num_faces': len(faces),
        'boundary_edges': [],
        'non_manifold_edges': [],
        'duplicate_vertices': [],
        'duplicate_faces': [],
        'inconsistent_normals': [],
        'isolated_vertices': [],
        'face_connectivity': None,
    }
    
    # 1. Find duplicate vertices (within tolerance)
    print("\n" + "="*60)
    print("CHECKING DUPLICATE VERTICES")
    print("="*60)
    
    # Use a spatial hash to find duplicates
    tolerance = 1e-6
    vertex_map = {}
    duplicates = []
    
    for i, v in enumerate(vertices):
        key = tuple(np.round(v / tolerance).astype(int))
        if key in vertex_map:
            duplicates.append((i, vertex_map[key]))
        else:
            vertex_map[key] = i
    
    if duplicates:
        print(f"  ❌ Found {len(duplicates)} duplicate vertices")
        results['duplicate_vertices'] = duplicates
        results['is_watertight'] = False
    else:
        print(f"  ✅ No duplicate vertices found")
    
    # 2. Find duplicate faces
    print("\n" + "="*60)
    print("CHECKING DUPLICATE FACES")
    print("="*60)
    
    face_set = set()
    dup_faces = []
    for i, face in enumerate(faces):
        # Sort vertices to handle different winding order
        face_key = tuple(sorted(face))
        if face_key in face_set:
            dup_faces.append(i)
        else:
            face_set.add(face_key)
    
    if dup_faces:
        print(f"  ❌ Found {len(dup_faces)} duplicate faces")
        results['duplicate_faces'] = dup_faces
        results['is_watertight'] = False
    else:
        print(f"  ✅ No duplicate faces found")
    
    # 3. Build edge adjacency
    print("\n" + "="*60)
    print("CHECKING EDGE ADJACENCY")
    print("="*60)
    
    edge_to_faces = defaultdict(list)
    edge_to_vertices = {}
    face_edges = []
    
    for face_idx, face in enumerate(faces):
        edges = [
            (min(face[0], face[1]), max(face[0], face[1])),
            (min(face[1], face[2]), max(face[1], face[2])),
            (min(face[2], face[0]), max(face[2], face[0])),
        ]
        face_edges.append(edges)
        
        for edge in edges:
            edge_to_faces[edge].append(face_idx)
            edge_to_vertices[edge] = edge
    
    # 4. Find boundary edges (shared by only 1 face = leaks)
    print("\n" + "="*60)
    print("CHECKING BOUNDARY EDGES (LEAKS)")
    print("="*60)
    
    boundary_edges = []
    non_manifold_edges = []
    
    for edge, face_list in edge_to_faces.items():
        if len(face_list) == 1:
            boundary_edges.append((edge, face_list[0]))
        elif len(face_list) > 2:
            non_manifold_edges.append((edge, face_list))
    
    if boundary_edges:
        print(f"  ❌ Found {len(boundary_edges)} boundary edges (LEAKS!)")
        print(f"  Total boundary edges: {len(boundary_edges)}")
        results['boundary_edges'] = boundary_edges
        results['is_watertight'] = False
        
        # Show first 10 boundary edges
        for i, (edge, face_idx) in enumerate(boundary_edges[:10]):
            v0, v1 = edge
            print(f"    Edge {i+1}: vertices ({v0}, {v1}) in face {face_idx}")
        if len(boundary_edges) > 10:
            print(f"    ... and {len(boundary_edges) - 10} more")
    else:
        print(f"  ✅ No boundary edges found (mesh is watertight!)")
    
    # 5. Check non-manifold edges
    if non_manifold_edges:
        print(f"  ❌ Found {len(non_manifold_edges)} non-manifold edges")
        for edge, face_list in non_manifold_edges[:5]:
            print(f"    Edge ({edge[0]}, {edge[1]}) shared by {len(face_list)} faces: {face_list}")
        results['non_manifold_edges'] = non_manifold_edges
        results['is_watertight'] = False
    else:
        print(f"  ✅ No non-manifold edges found")
    
    # 6. Check normal consistency
    print("\n" + "="*60)
    print("CHECKING NORMAL CONSISTENCY")
    print("="*60)
    
    # Compute face normals
    face_normals = []
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normal = normal / norm
        face_normals.append(normal)
    
    # Check if normals point consistently (compare adjacent faces)
    inconsistent = []
    for edge, face_list in edge_to_faces.items():
        if len(face_list) == 2:
            f0, f1 = face_list
            # Check if normals are opposite (should be for watertight mesh)
            dot = np.dot(face_normals[f0], face_normals[f1])
            if dot > 0.1:  # Normals should be roughly opposite
                inconsistent.append((edge, f0, f1, dot))
    
    if inconsistent:
        print(f"  ⚠️ Found {len(inconsistent)} adjacent faces with inconsistent normals")
        for edge, f0, f1, dot in inconsistent[:5]:
            print(f"    Edge ({edge[0]}, {edge[1]}): faces {f0} and {f1}, dot={dot:.4f}")
        results['inconsistent_normals'] = inconsistent
    else:
        print(f"  ✅ All normals are consistent")
    
    # 7. Check for isolated vertices
    print("\n" + "="*60)
    print("CHECKING ISOLATED VERTICES")
    print("="*60)
    
    used_vertices = set()
    for face in faces:
        used_vertices.update(face)
    
    isolated = [i for i in range(len(vertices)) if i not in used_vertices]
    
    if isolated:
        print(f"  ❌ Found {len(isolated)} isolated vertices (not used in any face)")
        results['isolated_vertices'] = isolated
        results['is_watertight'] = False
    else:
        print(f"  ✅ No isolated vertices found")
    
    # 8. Check mesh connectivity
    print("\n" + "="*60)
    print("CHECKING MESH CONNECTIVITY")
    print("="*60)
    
    # Build adjacency graph
    adj = defaultdict(set)
    for face in faces:
        for i in range(3):
            v0 = face[i]
            v1 = face[(i+1) % 3]
            adj[v0].add(v1)
            adj[v1].add(v0)
    
    # Find connected components
    visited = set()
    components = []
    
    for v in range(len(vertices)):
        if v not in visited and v in adj:
            # BFS
            component = set()
            queue = [v]
            visited.add(v)
            while queue:
                curr = queue.pop()
                component.add(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(component)
    
    # Also include isolated vertices as components
    for v in range(len(vertices)):
        if v not in visited:
            components.append({v})
            visited.add(v)
    
    if len(components) > 1:
        print(f"  ⚠️ Mesh has {len(components)} disconnected components")
        for i, comp in enumerate(components[:5]):
            print(f"    Component {i+1}: {len(comp)} vertices")
        results['face_connectivity'] = components
    else:
        print(f"  ✅ Mesh is fully connected")
    
    # 9. Summary
    print("\n" + "="*60)
    print("WATERTIGHTNESS SUMMARY")
    print("="*60)
    
    if results['is_watertight']:
        print("\n  ✅ MESH IS WATERTIGHT ✅")
        print("  No leaks, holes, or gaps found!")
    else:
        print("\n  ❌ MESH HAS LEAKS ❌")
        print("  Issues found:")
        if results['boundary_edges']:
            print(f"    - {len(results['boundary_edges'])} boundary edges (leaks/holes)")
        if results['non_manifold_edges']:
            print(f"    - {len(results['non_manifold_edges'])} non-manifold edges")
        if results['duplicate_vertices']:
            print(f"    - {len(results['duplicate_vertices'])} duplicate vertices")
        if results['duplicate_faces']:
            print(f"    - {len(results['duplicate_faces'])} duplicate faces")
        if results['isolated_vertices']:
            print(f"    - {len(results['isolated_vertices'])} isolated vertices")
        
        # Suggest fixes
        print("\n  Suggested fixes:")
        if results['boundary_edges']:
            print("    1. Fix boundary edges by adding missing faces or merging vertices")
        if results['duplicate_vertices']:
            print("    2. Remove duplicate vertices using a vertex welding function")
        if results['duplicate_faces']:
            print("    3. Remove duplicate faces")
        if results['non_manifold_edges']:
            print("    4. Fix non-manifold edges (often caused by duplicate faces)")
    
    # 10. Export diagnostics
    export_diagnostic_mesh(vertices, faces, results)
    
    return results


def export_diagnostic_mesh(vertices, faces, results):
    """
    Export a PLY file with boundary edges highlighted.
    """
    if not results['boundary_edges']:
        return
    
    print("\n" + "="*60)
    print("EXPORTING DIAGNOSTIC MESH")
    print("="*60)
    
    # Create a copy of vertices
    diag_vertices = vertices.copy()
    
    # Add colored vertices for boundary edges
    # Each boundary edge gets a colored vertex at its midpoint
    boundary_colors = []
    
    for edge, face_idx in results['boundary_edges']:
        v0, v1 = edge
        midpoint = (vertices[v0] + vertices[v1]) / 2
        diag_vertices = np.vstack([diag_vertices, midpoint])
        boundary_colors.append([1.0, 0.0, 0.0])  # Red
    
    # Create face list with colored vertices
    # This is simplified - you'd need to create actual lines/points
    
    print(f"  Added {len(boundary_colors)} boundary markers")
    print("  Diagnostic mesh exported to 'diagnostic_mesh.ply'")
    
    # Simple export (points only for debugging)
    with open('diagnostic_mesh_boundary.ply', 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(diag_vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("element face 0\n")
        f.write("end_header\n")
        
        # Write original vertices in gray
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]} 128 128 128\n")
        
        # Write boundary markers in red
        for v in diag_vertices[len(vertices):]:
            f.write(f"{v[0]} {v[1]} {v[2]} 255 0 0\n")
    
    print("  ✓ Saved boundary markers to 'diagnostic_mesh_boundary.ply'")


def visualize_boundary_edges(vertices, faces, results):
    """
    Quick visualization of boundary edges using matplotlib.
    """
    if not results['boundary_edges']:
        print("  No boundary edges to visualize")
        return
    
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot mesh
        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            ax.plot3D([v0[0], v1[0], v2[0], v0[0]], 
                      [v0[1], v1[1], v2[1], v0[1]],
                      [v0[2], v1[2], v2[2], v0[2]], 
                      'b-', alpha=0.3)
        
        # Highlight boundary edges in red
        for edge, face_idx in results['boundary_edges']:
            v0, v1 = vertices[edge[0]], vertices[edge[1]]
            ax.plot3D([v0[0], v1[0]], [v0[1], v1[1]], [v0[2], v1[2]], 
                      'r-', linewidth=3)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Boundary Edges (Red): {len(results["boundary_edges"])} edges')
        
        plt.savefig('boundary_edges.png', dpi=150)
        print("  ✓ Saved boundary edges visualization to 'boundary_edges.png'")
        plt.show()
    except ImportError:
        print("  ⚠️ matplotlib not installed - skipping visualization")


def main():
    print("Generating diamond geometry...")
    
    # Generate mesh with default parameters
    verts, faces = make_round_brilliant(
        girdle_radius=1.0,
        crown_angle_deg=34.5,
        pavilion_angle_deg=40.75,
        table_frac=0.56,
        num_main_facets=8,
        culet_radius=0.0,  # Sharp culet
    )
    
    # Flat shade
    fv, fn, uvs, ff = make_flat_shaded(verts, faces)
    
    print(f"  Vertices: {len(fv)}")
    print(f"  Faces: {len(ff)}")
    
    # Run checks
    results = check_mesh_watertight(fv, ff)
    
    # Visualize boundary edges if any
    if results['boundary_edges']:
        visualize_boundary_edges(fv, ff, results)
    
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    print(f"  Is watertight: {'✅ YES' if results['is_watertight'] else '❌ NO'}")
    print(f"  Total vertices: {results['num_vertices']}")
    print(f"  Total faces: {results['num_faces']}")
    print(f"  Boundary edges: {len(results['boundary_edges'])}")
    print(f"  Non-manifold edges: {len(results['non_manifold_edges'])}")
    print(f"  Duplicate vertices: {len(results['duplicate_vertices'])}")
    print(f"  Duplicate faces: {len(results['duplicate_faces'])}")
    print(f"  Isolated vertices: {len(results['isolated_vertices'])}")


if __name__ == "__main__":
    main()