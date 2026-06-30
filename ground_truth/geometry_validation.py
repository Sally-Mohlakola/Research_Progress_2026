#!/usr/bin/env python3
"""
visualize_mesh.py - Interactive 3D visualization of diamond mesh
with facet outlines only.

Usage:
    python visualize_mesh.py
    python visualize_mesh.py --culet 0.02
    python visualize_mesh.py --num_facets 12
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse

# Import your geometry generator
from brilliant_geometry import make_round_brilliant, make_flat_shaded


def create_mesh_with_outlines(vertices, faces, title="Diamond Mesh", colorscale='Viridis'):
    """
    Create interactive 3D mesh visualization with facet outlines.
    """
    # Calculate vertex heights for coloring
    heights = vertices[:, 2]
    
    # Create mesh3d trace (same as before)
    mesh = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        intensity=heights,
        colorscale=colorscale,
        opacity=0.85,
        showscale=True,
        colorbar=dict(title="Height (z)"),
        lighting=dict(
            ambient=0.6,
            diffuse=0.8,
            roughness=0.3,
            specular=0.5,
            fresnel=0.2
        ),
        lightposition=dict(
            x=100000, y=100000, z=100000
        ),
        flatshading=False,
    )
    
    # Add wireframe edges for ALL faces (not sampled)
    edge_traces = []
    
    # Use a set to avoid duplicate edges
    edges = set()
    
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        
        # Get edges for this triangle
        tri_edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]
        
        for edge in tri_edges:
            # Sort vertices to avoid duplicates
            edge_key = tuple(sorted(edge))
            if edge_key not in edges:
                edges.add(edge_key)
                
                # Get vertex positions
                v_start = vertices[edge_key[0]]
                v_end = vertices[edge_key[1]]
                
                # Draw edge
                edge_trace = go.Scatter3d(
                    x=[v_start[0], v_end[0]],
                    y=[v_start[1], v_end[1]],
                    z=[v_start[2], v_end[2]],
                    mode='lines',
                    line=dict(color='black', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                )
                edge_traces.append(edge_trace)
    
    print(f"  Edges drawn: {len(edges)}")
    
    # Create layout
    layout = go.Layout(
        title=dict(
            text=title,
            font=dict(size=20)
        ),
        scene=dict(
            xaxis=dict(title='X', gridcolor='lightgray'),
            yaxis=dict(title='Y', gridcolor='lightgray'),
            zaxis=dict(title='Z', gridcolor='lightgray'),
            aspectmode='data',
            camera=dict(
                eye=dict(x=2.5, y=1.5, z=1.2),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            )
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
        hovermode='closest',
    )
    
    # Combine all traces
    fig = go.Figure(data=[mesh] + edge_traces, layout=layout)
    
    return fig


def visualize_facet_types_with_outlines(vertices, faces):
    """
    Visualize diamond with facets colored by type AND black outlines.
    """
    # Compute face centers and normals
    face_centers = []
    face_normals = []
    facet_types = []
    
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        center = (v0 + v1 + v2) / 3
        face_centers.append(center)
        
        # Compute face normal
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normal = normal / norm
        face_normals.append(normal)
        
        # Classify facet type based on normal direction
        if normal[2] > 0.3:  # Upward facing (crown)
            facet_types.append('crown')
        elif normal[2] < -0.3:  # Downward facing (pavilion)
            facet_types.append('pavilion')
        else:  # Side facing (girdle)
            facet_types.append('girdle')
    
    # Color map
    color_map = {
        'crown': '#FF6B35',      # Orange
        'pavilion': '#4A6FA5',   # Blue
        'girdle': '#2ECC71'      # Green
    }
    
    # Create separate mesh for each facet type
    traces = []
    for ftype, color in color_map.items():
        ftype_faces = [f for f, t in zip(faces, facet_types) if t == ftype]
        if not ftype_faces:
            continue
            
        # Create mesh for this facet type
        mesh = go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=np.array(ftype_faces)[:, 0],
            j=np.array(ftype_faces)[:, 1],
            k=np.array(ftype_faces)[:, 2],
            color=color,
            opacity=0.9,
            flatshading=True,
            name=ftype.capitalize(),
            showscale=False,
        )
        traces.append(mesh)
    
    # Add ALL facet outlines in black
    edges = set()
    for face in faces:
        tri_edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]
        for edge in tri_edges:
            edge_key = tuple(sorted(edge))
            if edge_key not in edges:
                edges.add(edge_key)
                
                v_start = vertices[edge_key[0]]
                v_end = vertices[edge_key[1]]
                
                edge_trace = go.Scatter3d(
                    x=[v_start[0], v_end[0]],
                    y=[v_start[1], v_end[1]],
                    z=[v_start[2], v_end[2]],
                    mode='lines',
                    line=dict(color='black', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                )
                traces.append(edge_trace)
    
    # Add legend
    for ftype, color in color_map.items():
        traces.append(
            go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                name=ftype.capitalize(),
                showlegend=True
            )
        )
    
    # Create layout
    layout = go.Layout(
        title=dict(
            text="Diamond Facet Types with Outlines",
            font=dict(size=18)
        ),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            camera=dict(eye=dict(x=2.5, y=1.5, z=1.2))
        ),
        width=1000,
        height=800,
        legend=dict(
            x=1.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )
    
    fig = go.Figure(data=traces, layout=layout)
    return fig


def visualize_with_normals_and_outlines(vertices, normals, faces, title="Diamond with Normals"):
    """
    Visualize mesh with normal vectors AND black outlines.
    """
    # Create mesh
    mesh = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=0.4,
        color='lightblue',
        showscale=False,
        flatshading=True,
    )
    
    traces = [mesh]
    
    # Add ALL facet outlines in black
    edges = set()
    for face in faces:
        tri_edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]
        for edge in tri_edges:
            edge_key = tuple(sorted(edge))
            if edge_key not in edges:
                edges.add(edge_key)
                
                v_start = vertices[edge_key[0]]
                v_end = vertices[edge_key[1]]
                
                edge_trace = go.Scatter3d(
                    x=[v_start[0], v_end[0]],
                    y=[v_start[1], v_end[1]],
                    z=[v_start[2], v_end[2]],
                    mode='lines',
                    line=dict(color='black', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                )
                traces.append(edge_trace)
    
    # Sample normals (every 50th vertex for clarity)
    step = max(1, len(vertices) // 50)
    normals_sampled = normals[::step]
    verts_sampled = vertices[::step]
    
    # Create normal arrows as cones
    arrow_length = 0.15
    for i, (v, n) in enumerate(zip(verts_sampled, normals_sampled)):
        if np.linalg.norm(n) < 1e-6:
            continue
            
        cone = go.Cone(
            x=[v[0]],
            y=[v[1]],
            z=[v[2]],
            u=[n[0]],
            v=[n[1]],
            w=[n[2]],
            sizemode='absolute',
            sizeref=arrow_length,
            showscale=False,
            colorscale=[[0, 'red'], [1, 'red']],
            opacity=0.8,
            hoverinfo='skip'
        )
        traces.append(cone)
    
    # Create layout
    layout = go.Layout(
        title=dict(
            text=title,
            font=dict(size=20)
        ),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            camera=dict(eye=dict(x=2.5, y=1.5, z=1.2))
        ),
        width=1000,
        height=800
    )
    
    fig = go.Figure(data=traces, layout=layout)
    return fig


def create_mesh_plotly(vertices, faces, title="Diamond Mesh", colorscale='Viridis'):
    """
    Original function - creates mesh without outlines.
    """
    heights = vertices[:, 2]
    
    mesh = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        intensity=heights,
        colorscale=colorscale,
        opacity=0.85,
        showscale=True,
        colorbar=dict(title="Height (z)"),
        lighting=dict(
            ambient=0.6,
            diffuse=0.8,
            roughness=0.3,
            specular=0.5,
            fresnel=0.2
        ),
        lightposition=dict(
            x=100000, y=100000, z=100000
        ),
        flatshading=False,
    )
    
    layout = go.Layout(
        title=dict(text=title, font=dict(size=20)),
        scene=dict(
            xaxis=dict(title='X', gridcolor='lightgray'),
            yaxis=dict(title='Y', gridcolor='lightgray'),
            zaxis=dict(title='Z', gridcolor='lightgray'),
            aspectmode='data',
            camera=dict(
                eye=dict(x=2.5, y=1.5, z=1.2),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            )
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
        hovermode='closest',
    )
    
    fig = go.Figure(data=[mesh], layout=layout)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize diamond mesh")
    parser.add_argument("--culet", type=float, default=0.0,
                       help="Culet radius (0 for sharp culet)")
    parser.add_argument("--num_facets", type=int, default=8,
                       help="Number of main facets")
    parser.add_argument("--no_normals", action='store_true',
                       help="Don't show normals")
    parser.add_argument("--facet_types", action='store_true',
                       help="Color facets by type (crown/pavilion/girdle)")
    parser.add_argument("--slice", type=str, default=None,
                       help="Show a slice (z, y, or x)")
    parser.add_argument("--slice_value", type=float, default=0.0,
                       help="Slice position")
    parser.add_argument("--title", type=str, default="Diamond Mesh",
                       help="Plot title")
    parser.add_argument("--outlines", action='store_true',
                       help="Add black outlines to all facets")
    parser.add_argument("--no_outlines", action='store_true',
                       help="Remove outlines (default for standard views)")
    args = parser.parse_args()
    
    print("Generating diamond geometry...")
    print(f"  Culet radius: {args.culet}")
    print(f"  Main facets: {args.num_facets}")
    
    # Generate mesh
    verts, faces = make_round_brilliant(
        girdle_radius=1.0,
        crown_angle_deg=34.5,
        pavilion_angle_deg=40.75,
        table_frac=0.56,
        num_main_facets=args.num_facets,
        culet_radius=args.culet,
    )
    
    # Flat shade
    fv, fn, uvs, ff = make_flat_shaded(verts, faces)
    
    print(f"  Vertices: {len(fv)}")
    print(f"  Faces: {len(ff)}")
    print(f"  Normals: {len(fn)}")
    
    # Create visualization
    if args.facet_types:
        if args.outlines:
            fig = visualize_facet_types_with_outlines(fv, ff)
        else:
            from visualize_mesh_original import visualize_facet_types
            fig = visualize_facet_types(fv, ff)
    elif args.slice:
        fig = visualize_slice(fv, ff, args.slice, args.slice_value)
    elif not args.no_normals:
        if args.outlines:
            fig = visualize_with_normals_and_outlines(fv, fn, ff, args.title)
        else:
            fig = visualize_with_normals(fv, fn, ff, args.title)
    else:
        if args.outlines:
            fig = create_mesh_with_outlines(fv, ff, args.title)
        else:
            fig = create_mesh_plotly(fv, ff, args.title)
    
    fig.show()
    print("\n✓ Visualization complete! Rotate and zoom with mouse.")
    
    # Print mesh statistics
    print("\nMesh Statistics:")
    print(f"  Vertices: {len(fv)}")
    print(f"  Faces: {len(ff)}")
    print(f"  Normals: {len(fn)}")
    print(f"  Bounding box:")
    print(f"    x: [{fv[:,0].min():.3f}, {fv[:,0].max():.3f}]")
    print(f"    y: [{fv[:,1].min():.3f}, {fv[:,1].max():.3f}]")
    print(f"    z: [{fv[:,2].min():.3f}, {fv[:,2].max():.3f}]")


if __name__ == "__main__":
    main()