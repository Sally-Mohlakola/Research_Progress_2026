"""
Render a rotating round-brilliant-cut diamond in Mitsuba 3.

Physical model:
  - Geometry: procedural round brilliant (table, crown bezel+star facets,
    girdle, pavilion main+lower-girdle facets, culet). Flat-shaded so each
    facet keeps a sharp, distinct normal -- essential for realistic facet
    reflections/refractions ("fire" and "scintillation").
  - Material: `dielectric` BSDF with int_ior = 2.419 (diamond at ~589nm),
    ext_ior = 1.000277 (air). This is the physically correct way to render
    diamond -- NOT the d-C/d-C_palik conductor presets, which model opaque
    crystalline-carbon surfaces (absorbing, metal-like), not a clear
    gemstone. A real diamond's brilliance comes entirely from its very
    high IOR driving total internal reflection inside a dielectric, not
    from any absorption.
  - Environment: a constant/gradient environment emitter plus a couple of
    area lights and a ground plane, so the dielectric has something
    well-structured to refract/reflect -- a diamond rendered against pure
    black with no scene around it just looks like a dark blob, because
    there's nothing for the facets to catch.
"""

import mitsuba as mi
import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from brilliant_geometry import make_round_brilliant, make_flat_shaded

# Spectral rather than RGB: dispersion is what produces a diamond's fire,
# and an RGB renderer has no wavelength axis to disperse along.
mi.set_variant("scalar_spectral")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bsdf.dispersive_dielectric import DispersiveDielectric
from utils.studio_env import studio_lighting, display_exposure

# ─────────────────────────────────────────────
# 1. Geometry
# ─────────────────────────────────────────────

raw_verts, raw_faces = make_round_brilliant(
    girdle_radius=1.0,
    crown_angle_deg=34.5,
    pavilion_angle_deg=40.75,
    table_frac=0.56,
)
verts, normals, uvs, faces = make_flat_shaded(raw_verts, raw_faces)

print(f"[mesh] flat-shaded vertices : {len(verts)}")
print(f"[mesh] faces                : {len(faces)}")

mesh = mi.Mesh(
    "diamond_mesh",
    vertex_count=len(verts),
    face_count=len(faces),
    has_vertex_normals=True,
    has_vertex_texcoords=True,
)
mesh_params = mi.traverse(mesh)
mesh_params["vertex_positions"] = mi.ArrayXf(verts.flatten())
mesh_params["vertex_normals"]   = mi.ArrayXf(normals.flatten())
mesh_params["vertex_texcoords"] = mi.ArrayXf(uvs.flatten())
mesh_params["faces"]            = mi.ArrayXu(faces.flatten())
mesh_params.update()

# ─────────────────────────────────────────────
# 2. Material: physically correct diamond dielectric
# ─────────────────────────────────────────────

diamond_bsdf = {
    # Dispersive: n varies with wavelength per the Sellmeier fit in
    # bsdf/dispersion.py, so blue refracts more strongly than red and the
    # stone throws coloured flashes. Mitsuba's stock `dielectric` takes a
    # single scalar IOR and stays achromatic even under a spectral variant,
    # so it cannot show fire; set "type" back to "dielectric" to see the
    # difference.
    "type": "dispersive_dielectric",
    "ext_ior": 1.000277,  # air
    "dispersion": True,
}
mesh.set_bsdf(mi.load_dict(diamond_bsdf))

# ─────────────────────────────────────────────
# 3. Scene: ground plane + environment + lights
# ─────────────────────────────────────────────

ground_bsdf = {
    "type": "diffuse",
    "reflectance": {"type": "rgb", "value": [0.12, 0.12, 0.14]},
}

# Base scene dictionary (without mesh transformation)
base_scene_dict = {
    "type": "scene",

    "integrator": {
        "type": "path",
        # See the note in eval.py: 24 truncates paths that a diamond
        # genuinely needs, and those show up as black facets.
        "max_depth": 64,
    },

    "sensor": {
        "type": "perspective",
        "fov": 32,
        "to_world": mi.ScalarTransform4f.look_at(
            origin=[0.0, -4.6, 2.4],
            target=[0.0,  0.0, -0.05],
            up    =[0.0,  0.0, 1.0],
        ),
        "film": {
            "type": "hdrfilm", # vary the lighting conditions
            "width":  768,
            "height": 768,
            "rfilter": {"type": "gaussian"},
            "pixel_format": "rgb",
        },
        "sampler": {
            "type": "independent",
            "sample_count": 256,  # Reduced for faster rendering
        },
    },

    # Lighting comes from utils/studio_env.py, shared with eval.py so the
    # analytic and neural renders are lit identically. It must not be a
    # single constant environment: every path escaping the stone would then
    # return the same radiance whatever route it took, and the facets would
    # render as flat unshaded grey.
    **studio_lighting(),

    "ground": {
        "type": "rectangle",
        "to_world": mi.ScalarTransform4f.translate([0, 0, -0.86]).scale([6, 6, 1]),
        "bsdf": ground_bsdf,
    },

    # Mesh will be added with rotation
}

# ─────────────────────────────────────────────
# 4. Render Animation
# ─────────────────────────────────────────────

def render_frame(frame_num, total_frames, rotation_axis='y'):
    """
    Render a single frame with diamond rotated around specified axis.
    
    Parameters:
    - frame_num: current frame number
    - total_frames: total number of frames in animation
    - rotation_axis: 'x', 'y', or 'z' for rotation axis
    """
    
    # Calculate rotation angle (full 360° rotation)
    angle = (frame_num / total_frames) * 2 * np.pi
    
    # Create rotation transform based on axis
    if rotation_axis.lower() == 'x':
        rotation = mi.ScalarTransform4f.rotate([1, 0, 0], np.degrees(angle))
    elif rotation_axis.lower() == 'y':
        rotation = mi.ScalarTransform4f.rotate([0, 1, 0], np.degrees(angle))
    elif rotation_axis.lower() == 'z':
        rotation = mi.ScalarTransform4f.rotate([0, 0, 1], np.degrees(angle))
    else:
        rotation = mi.ScalarTransform4f.rotate([0, 1, 0], np.degrees(angle))
    
    # Apply rotation to mesh
    rotated_mesh = mi.Mesh(
        "diamond_mesh",
        vertex_count=len(verts),
        face_count=len(faces),
        has_vertex_normals=True,
        has_vertex_texcoords=True,
    )
    
    # Rotate vertices manually
    rotated_verts = []
    for v in verts:
        x, y, z = v
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        if rotation_axis.lower() == 'y':
            rx = x * cos_a + z * sin_a
            ry = y
            rz = -x * sin_a + z * cos_a
        elif rotation_axis.lower() == 'x':
            rx = x
            ry = y * cos_a - z * sin_a
            rz = y * sin_a + z * cos_a
        elif rotation_axis.lower() == 'z':
            rx = x * cos_a - y * sin_a
            ry = x * sin_a + y * cos_a
            rz = z
        else:
            rx, ry, rz = x, y, z
        rotated_verts.append([rx, ry, rz])
    
    rotated_verts = np.array(rotated_verts, dtype=np.float32)
    
    # Rotate normals (same rotation without translation)
    rotated_normals = []
    for n in normals:
        nx, ny, nz = n
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        if rotation_axis.lower() == 'y':
            rnx = nx * cos_a + nz * sin_a
            rny = ny
            rnz = -nx * sin_a + nz * cos_a
        elif rotation_axis.lower() == 'x':
            rnx = nx
            rny = ny * cos_a - nz * sin_a
            rnz = ny * sin_a + nz * cos_a
        elif rotation_axis.lower() == 'z':
            rnx = nx * cos_a - ny * sin_a
            rny = nx * sin_a + ny * cos_a
            rnz = nz
        else:
            rnx, rny, rnz = nx, ny, nz
        rotated_normals.append([rnx, rny, rnz])
    
    rotated_normals = np.array(rotated_normals, dtype=np.float32)
    
    # Populate rotated mesh
    mesh_params_rot = mi.traverse(rotated_mesh)
    mesh_params_rot["vertex_positions"] = mi.ArrayXf(rotated_verts.flatten())
    mesh_params_rot["vertex_normals"]   = mi.ArrayXf(rotated_normals.flatten())
    mesh_params_rot["vertex_texcoords"] = mi.ArrayXf(uvs.flatten())
    mesh_params_rot["faces"]            = mi.ArrayXu(faces.flatten())
    mesh_params_rot.update()
    
    # Set BSDF
    rotated_mesh.set_bsdf(mi.load_dict(diamond_bsdf))
    
    # Build scene for this frame
    scene_dict = dict(base_scene_dict)
    scene_dict["mesh"] = rotated_mesh
    
    scene = mi.load_dict(scene_dict)
    
    # Render
    image = mi.render(scene, spp=256)
    
    return image

# ─────────────────────────────────────────────
# 5. Generate Animation
# ─────────────────────────────────────────────

def create_rotation_animation(
    total_frames=36, 
    rotation_axis='y',
    output_dir="diamond_animation",
    make_video=True
):
    """
    Create a rotating diamond animation.
    
    Parameters:
    - total_frames: number of frames (36 = 10° per frame for smooth rotation)
    - rotation_axis: 'x', 'y', or 'z'
    - output_dir: directory to save frames
    - make_video: whether to create a video from frames
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[animation] Generating {total_frames} frames...")
    print(f"[animation] Rotation axis: {rotation_axis}")
    print(f"[animation] Output directory: {output_dir}")
    
    frames = []
    start_time = time.time()
    
    for i in range(total_frames):
        frame_start = time.time()
        print(f"  Rendering frame {i+1}/{total_frames}...", end="", flush=True)
        
        # Render frame
        image = render_frame(i, total_frames, rotation_axis)
        
        # Save frame as PNG. The studio rig emits normalised radiance (1.0 is
        # its brightest source), so the frame has to be scaled by
        # `display_exposure()` before the sRGB curve or it comes out near black.
        # The EXR below stays raw.
        frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        bmp = mi.Bitmap(np.array(image) * display_exposure())
        bmp = bmp.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, srgb_gamma=True)
        bmp.write(frame_path)
        frames.append(frame_path)
        
        # Optionally save EXR for HDR
        if i % 10 == 0:  # Save every 10th frame as EXR
            exr_path = os.path.join(output_dir, f"frame_{i:04d}.exr")
            mi.util.write_bitmap(exr_path, image)
        
        frame_time = time.time() - frame_start
        print(f" done ({frame_time:.1f}s)")
    
    total_time = time.time() - start_time
    print(f"\n[animation] Saved {len(frames)} frames to {output_dir}")
    print(f"[animation] Total time: {total_time:.1f}s ({total_time/len(frames):.1f}s per frame)")
    
    # ─────────────────────────────────────────────
    # 6. Create video from frames
    # ─────────────────────────────────────────────
    
    if make_video:
        try:
            import cv2
            import imageio
            
            print("[video] Creating video from frames...")
            
            # Read first frame to get dimensions
            first_frame = cv2.imread(frames[0])
            height, width = first_frame.shape[:2]
            
            # Create video writer
            video_path = os.path.join(output_dir, "diamond_rotation.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
            
            for frame_path in frames:
                frame = cv2.imread(frame_path)
                video_writer.write(frame)
            
            video_writer.release()
            print(f"[video] Video saved to {video_path}")
            
            # Also create a GIF (for easy sharing)
            print("[video] Creating GIF...")
            gif_path = os.path.join(output_dir, "diamond_rotation.gif")
            
            # Read all frames for GIF (downsample for size)
            gif_frames = []
            step = max(1, len(frames) // 36)  # ~36 frames for GIF
            for i in range(0, len(frames), step):
                img = cv2.imread(frames[i])
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                gif_frames.append(img_rgb)
            
            imageio.mimsave(gif_path, gif_frames, duration=0.05, loop=0)
            print(f"[video] GIF saved to {gif_path}")
            
        except ImportError as e:
            print(f"\n[video] Required library not installed: {e}")
            print("[video] To create video, install: pip install imageio opencv-python")
        except Exception as e:
            print(f"\n[video] Error creating video: {e}")

# ─────────────────────────────────────────────
# 7. Run Animation
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Create rotating animation
    create_rotation_animation(
        total_frames=36,  # 36 frames = 10° per step
        rotation_axis='y',  # Rotate around Y axis (spinning like a top)
        output_dir="diamond_animation",
        make_video=True
    )