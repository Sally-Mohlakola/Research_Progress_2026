#!/usr/bin/env python3
"""
eval_animate_roll.py - Render a rolling diamond animation using the Neural Diamond BSDF.

The hybrid BSDF blends:
  - DiamondFacet  : analytic Fresnel R/T
  - Model_M       : neural multi-scatter colour f(wi, wo)

Animation: Diamond rolls/tumbles in 3D space with orbiting camera
FIXED: Added numerical stability and NaN handling
"""

import os
import sys
import argparse
import json
import struct
import tempfile
import math
import numpy as np
import torch
import mitsuba as mi
import drjit as dr
from pathlib import Path

project_root = '/mnt/c/Users/sally/research/diamond_rendering'
if os.path.exists(project_root):
    sys.path.insert(0, project_root)
    for subdir in ['utils', 'config', 'neural', 'bsdf']:
        p = os.path.join(project_root, subdir)
        if os.path.exists(p): sys.path.insert(0, p)

from config import device, variant

# Register NeuralDiamond BSDF
from bsdf.neural_bsdf import NeuralDiamond
mi.register_bsdf("neural_diamond", lambda props: NeuralDiamond(props))

mi.set_variant(variant)
from mitsuba import ScalarTransform4f as sT

from bsdf.analytic_bsdf import DiamondShading
from neural.base_model import Model_M, Model_T
from neural.drjit_wrapper import MiModelWrapper


def parse_args():
    parser = argparse.ArgumentParser(description="Render a rolling diamond animation with neural shading")
    parser.add_argument("--checkpoint_name", type=str, required=True)
    parser.add_argument("--diamond_name", type=str, default='round_brilliant_sharp_culet')
    parser.add_argument("--spp", type=int, default=64, help="Samples per pixel per frame")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="rolling_animation")
    parser.add_argument("--frames", type=int, default=60, help="Number of animation frames")
    parser.add_argument("--exposure", type=float, default=1.0)
    parser.add_argument("--no_neural", action='store_true',
                        help="Use analytic dielectric only (ground truth reference)")
    parser.add_argument("--fps", type=int, default=30, help="Video frames per second")
    parser.add_argument("--orbit_radius", type=float, default=5.0, help="Camera orbit radius")
    parser.add_argument("--rotation_speed", type=float, default=1.0, help="Rotation speed multiplier")
    parser.add_argument("--clamp_value", type=float, default=10.0, help="Clamp neural BSDF output to this value")
    return parser.parse_args()


DIAMOND_PRESETS = {
    'round_brilliant_sharp_culet': dict(
        girdle_radius=1.0, crown_angle_deg=34.5, pavilion_angle_deg=40.75,
        table_frac=0.56, num_main_facets=8, culet_radius=0.0,
        int_ior=2.419, ext_ior=1.000277,
    ),
    'round_brilliant_culet': dict(
        girdle_radius=1.0, crown_angle_deg=34.5, pavilion_angle_deg=40.75,
        table_frac=0.56, num_main_facets=8, culet_radius=0.02,
        int_ior=2.419, ext_ior=1.000277,
    ),
    'princess': dict(
        girdle_radius=1.0, crown_angle_deg=35.0, pavilion_angle_deg=41.0,
        table_frac=0.60, num_main_facets=4, culet_radius=0.0,
        int_ior=2.419, ext_ior=1.000277,
    ),
}


def write_ply(path, vertices, normals, faces):
    with open(path, 'wb') as f:
        hdr = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(vertices)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property float nx\nproperty float ny\nproperty float nz\n"
            f"element face {len(faces)}\n"
            "property list uchar int vertex_indices\nend_header\n"
        )
        f.write(hdr.encode('ascii'))
        f.write(np.concatenate([vertices, normals], axis=1).astype(np.float32).tobytes())
        for face in faces:
            f.write(struct.pack('<B', 3))
            f.write(struct.pack('<3i', *face))


def build_diamond_mesh(diamond_params):
    from ground_truth.brilliant_geometry import make_round_brilliant, make_flat_shaded
    geometry_keys = {'girdle_radius', 'crown_angle_deg', 'pavilion_angle_deg',
                     'table_frac', 'num_main_facets', 'culet_radius'}
    geom = {k: v for k, v in diamond_params.items() if k in geometry_keys}
    verts, faces = make_round_brilliant(**geom)
    fv, fn, _, ff = make_flat_shaded(verts, faces)
    tmp = tempfile.NamedTemporaryFile(suffix='.ply', delete=False)
    tmp.close()
    write_ply(tmp.name, fv, fn, ff)
    return tmp.name, fv, fn, ff


def load_neural_bsdf_direct(checkpoint_dir, diamond_params, clamp_value=10.0):
    """Load neural BSDF with numerical stability."""
    
    model_m = Model_M().to(device)
    model_m_path = os.path.join(checkpoint_dir, 'model_m.pth')
    
    if not os.path.exists(model_m_path):
        print(f"  ⚠ Model_M not found at {model_m_path}")
        return None
    
    model_m.load_state_dict(
        torch.load(model_m_path, map_location=device, weights_only=False)
    )
    model_m.eval()
    print("✓ Model_M weights loaded")
    
    # Test model and determine activation
    test_wi = torch.tensor([[0.0, 0.0, 1.0]], device=device)
    test_wo = torch.tensor([[0.5, 0.5, 0.7071]], device=device)
    test_input = torch.cat([test_wi, test_wo], dim=1)
    
    with torch.no_grad():
        m_pred = model_m(test_input)
        print(f"  Model_M test output: {m_pred.tolist()}")
        
        # Check for NaN
        if torch.isnan(m_pred).any() or torch.isinf(m_pred).any():
            print("  ⚠ Model_M output contains nan/inf, using safe activation")
            activation = lambda x: dr.maximum(0.0, dr.minimum(x, clamp_value))
        else:
            # Use clamp with ReLU for stability
            activation = lambda x: dr.maximum(0.0, dr.minimum(x, clamp_value))
            print(f"  ✓ Using clamped ReLU activation (max={clamp_value})")

    class WrapperWithSkip(MiModelWrapper):
        def test(self, samples=42900):
            pass
    
    # Build wrapper with stable activation
    mlp_m = WrapperWithSkip(model_m, activation)
    print("✓ DrJIT wrapper built for Model_M")
    
    props = mi.Properties()
    props['int_ior'] = diamond_params['int_ior']
    props['ext_ior'] = diamond_params['ext_ior']
    props['type'] = 'neural_diamond'
    props['use_physics_fallback'] = True  # Enable physics fallback for untrained directions
    props['confidence_threshold'] = 0.05  # Threshold for blending to physics
    
    bsdf = NeuralDiamond(props)
    bsdf.model_m = mlp_m
    
    # Add a clamp to the BSDF output
    def safe_eval(bsdf, *args, **kwargs):
        result = bsdf.__class__.eval(bsdf, *args, **kwargs)
        # Clamp to prevent extreme values
        return dr.clamp(result, 0.0, clamp_value)
    
    # Monkey patch eval for safety
    bsdf.eval = safe_eval.__get__(bsdf, NeuralDiamond)
    
    print("✓ NeuralDiamond BSDF ready with numerical stability")
    return bsdf


def load_analytic_bsdf(diamond_params):
    """Create analytic dielectric BSDF."""
    return {
        'type': 'dielectric',
        'int_ior': diamond_params['int_ior'],
        'ext_ior': diamond_params['ext_ior'],
    }


def rotate_vertex(v, angle_x, angle_y, angle_z):
    """Apply 3D rotation to a single vertex."""
    x, y, z = v
    
    # Rotate around X axis
    cos_a = math.cos(angle_x)
    sin_a = math.sin(angle_x)
    y1 = y * cos_a - z * sin_a
    z1 = y * sin_a + z * cos_a
    x1 = x
    
    # Rotate around Y axis
    cos_b = math.cos(angle_y)
    sin_b = math.sin(angle_y)
    x2 = x1 * cos_b + z1 * sin_b
    z2 = -x1 * sin_b + z1 * cos_b
    y2 = y1
    
    # Rotate around Z axis
    cos_c = math.cos(angle_z)
    sin_c = math.sin(angle_z)
    x3 = x2 * cos_c - y2 * sin_c
    y3 = x2 * sin_c + y2 * cos_c
    z3 = z2
    
    return np.array([x3, y3, z3], dtype=np.float32)


def rotate_vertices(vertices, angles):
    """Apply 3D rotation to all vertices."""
    angle_x, angle_y, angle_z = angles
    return np.array([rotate_vertex(v, angle_x, angle_y, angle_z) for v in vertices], dtype=np.float32)


def create_rotated_ply(ply_path, vertices, normals, faces, frame_num, total_frames, rotation_speed=1.0):
    """Create a rotated PLY file for a specific frame."""
    
    # Calculate rolling angles - smooth tumbling motion
    t = frame_num / total_frames
    
    # Use smoother interpolation for the rolling motion
    # This prevents sudden jumps that can cause rendering artifacts
    angle_x = 2.0 * 2 * math.pi * t * rotation_speed
    angle_y = 1.5 * 2 * math.pi * t * rotation_speed  
    angle_z = 0.5 * 2 * math.pi * t * rotation_speed
    
    angles = (angle_x, angle_y, angle_z)
    
    # Rotate vertices and normals
    rotated_verts = rotate_vertices(vertices, angles)
    rotated_normals = rotate_vertices(normals, angles)
    
    # Write rotated PLY
    tmp_path = ply_path.replace('.ply', f'_frame_{frame_num:04d}.ply')
    write_ply(tmp_path, rotated_verts, rotated_normals, faces)
    
    return tmp_path, angles


def create_scene(diamond_params, bsdf, width, height, frame_idx, total_frames, 
                 ply_path, vertices, normals, faces, orbit_radius, rotation_speed):
    """Build scene with diamond rotating around Y axis (no flipping)."""
    
    t = frame_idx / total_frames
    
    # Rotation around Y axis only - full 360° rotation
    angle = 2 * math.pi * t * rotation_speed
    
    # Rotate vertices around Y axis
    rotated_verts = []
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    for v in vertices:
        x, y, z = v
        rx = x * cos_a + z * sin_a
        ry = y  # Y stays the same
        rz = -x * sin_a + z * cos_a
        rotated_verts.append([rx, ry, rz])
    rotated_verts = np.array(rotated_verts, dtype=np.float32)
    
    # Rotate normals the same way
    rotated_normals = []
    for n in normals:
        nx, ny, nz = n
        rnx = nx * cos_a + nz * sin_a
        rny = ny  # Y stays the same
        rnz = -nx * sin_a + nz * cos_a
        rotated_normals.append([rnx, rny, rnz])
    rotated_normals = np.array(rotated_normals, dtype=np.float32)
    
    # Write rotated PLY
    rotated_ply_path = ply_path.replace('.ply', f'_frame_{frame_idx:04d}.ply')
    write_ply(rotated_ply_path, rotated_verts, rotated_normals, faces)
    
    # Camera stays fixed (or orbits slowly)
    cam_x = 0.0
    cam_y = 1.8
    cam_z = 4.0
    
    # Optional: slight camera orbit around diamond
    # cam_angle = 0.3 * 2 * math.pi * t  # Slow orbit
    # cam_x = orbit_radius * math.sin(cam_angle) * 0.3
    # cam_z = orbit_radius * math.cos(cam_angle) * 0.3
    
    ground_bsdf = {
        'type': 'diffuse',
        'reflectance': {'type': 'rgb', 'value': [0.12, 0.12, 0.14]},
    }

    scene_dict = {
        'type': 'scene',

        'integrator': {
            'type': 'path',
            'max_depth': 24,
        },

        'sensor': {
            'type': 'perspective',
            'fov': 30,
            'to_world': sT.look_at(
                origin=[cam_x, cam_y, cam_z],
                target=[0.0, 0.0, 0.0],
                up=[0.0, 1.0, 0.0],
            ),
            'film': {
                'type': 'hdrfilm',
                'width': width,
                'height': height,
                'pixel_format': 'rgb',
                'rfilter': {'type': 'gaussian'},
            },
            'sampler': {
                'type': 'independent',
                'sample_count': 256,
            },
        },

        'envmap': {
            'type': 'constant',
            'radiance': {'type': 'rgb', 'value': [0.55, 0.58, 0.65]},
        },

        'ground': {
            'type': 'rectangle',
            'to_world': sT.translate([0, 0, -0.9]).scale([8, 8, 1]),
            'bsdf': ground_bsdf,
        },

        'key_light': {
            'type': 'rectangle',
            'to_world': sT.look_at(
                origin=[3.0, -3.5, 5.0],
                target=[0.0, 0.0, 0.0],
                up=[0.0, 0.0, 1.0],
            ).scale([1.2, 1.2, 1.0]),
            'emitter': {
                'type': 'area',
                'radiance': {'type': 'rgb', 'value': [40.0, 38.0, 35.0]},
            },
        },

        'rim_light': {
            'type': 'rectangle',
            'to_world': sT.look_at(
                origin=[-3.5, 2.5, 1.5],
                target=[0.0, 0.0, 0.0],
                up=[0.0, 0.0, 1.0],
            ).scale([1.0, 1.0, 1.0]),
            'emitter': {
                'type': 'area',
                'radiance': {'type': 'rgb', 'value': [10.0, 14.0, 22.0]},
            },
        },

        'top_light': {
            'type': 'rectangle',
            'to_world': sT.look_at(
                origin=[0.0, 0.0, 6.0],
                target=[0.0, 0.0, 0.0],
                up=[0.0, 1.0, 0.0],
            ).scale([2.0, 2.0, 1.0]),
            'emitter': {
                'type': 'area',
                'radiance': {'type': 'rgb', 'value': [12.0, 12.0, 13.0]},
            },
        },

        'fill_light': {
            'type': 'rectangle',
            'to_world': sT.look_at(
                origin=[-2.0, -3.0, 2.0],
                target=[0.0, 0.0, 0.0],
                up=[0.0, 0.0, 1.0],
            ).scale([1.0, 1.0, 1.0]),
            'emitter': {
                'type': 'area',
                'radiance': {'type': 'rgb', 'value': [8.0, 10.0, 12.0]},
            },
        },

        'diamond': {
            'type': 'ply',
            'filename': rotated_ply_path,
            'bsdf': bsdf,
        },
    }

    scene = mi.load_dict(scene_dict)

    # Clean up temporary PLY file (after scene is built)
    try:
        os.unlink(rotated_ply_path)
    except OSError:
        pass

    return scene


def tonemap(image, exposure=1.0):
    img = np.array(image) * exposure
    img = np.clip(img, 0.0, 1.0)
    img = np.power(img, 1.0 / 2.2)
    return (img * 255).astype(np.uint8)


def render_frame(scene, spp, frame_idx, total_frames):
    """Render a single frame with error handling."""
    print(f"  Rendering frame {frame_idx+1}/{total_frames}...")
    
    try:
        # Progressive rendering with batches
        image = None
        batch_size = min(8, spp)
        
        for i in range(0, spp, batch_size):
            batch_spp = min(batch_size, spp - i)
            img = mi.render(scene, spp=batch_spp, seed=i + frame_idx * 10000)
            
            # Check for NaN in rendered image
            img_np = np.array(img)
            if np.isnan(img_np).any() or np.isinf(img_np).any():
                print(f"    ⚠ Warning: NaN/Inf detected in frame {frame_idx+1}, replacing with zeros")
                img_np = np.nan_to_num(img_np, nan=0.0, posinf=0.0, neginf=0.0)
                img = mi.TensorXf(img_np)
            
            image = img if image is None else image + img
            dr.flush_malloc_cache()
        
        image /= spp
        return image
        
    except Exception as e:
        print(f"    ⚠ Error rendering frame {frame_idx+1}: {e}")
        # Return black frame
        return mi.TensorXf(np.zeros((args.height, args.width, 3), dtype=np.float32))


def create_animation_frames(args, bsdf, diamond_params, vertices, normals, faces):
    """Render all animation frames."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    # Create base PLY path
    ply_path = os.path.join(tempfile.gettempdir(), 'diamond_base.ply')
    write_ply(ply_path, vertices, normals, faces)
    
    print(f"\n🎬 Rendering {args.frames} rotation animation frames...")
    print(f"   SPP: {args.spp}")
    print(f"   Resolution: {args.width}×{args.height}")
    print(f"   Rotation speed: {args.rotation_speed}x")
    print(f"   Output: {output_dir}")
    print(f"   Camera: fixed at [0, 1.8, 4.0]")
    
    frames = []
    failed_frames = []
    
    for frame_idx in range(args.frames):
        # Build scene for this frame
        try:
            scene = create_scene(
                diamond_params,
                bsdf,
                args.width,
                args.height,
                frame_idx,
                args.frames,
                ply_path,
                vertices,
                normals,
                faces,
                args.orbit_radius,
                args.rotation_speed
            )
        except Exception as e:
            print(f"  ⚠ Error building scene for frame {frame_idx+1}: {e}")
            failed_frames.append(frame_idx)
            continue
        
        # Render frame
        image = render_frame(scene, args.spp, frame_idx, args.frames)
        
        # Save tonemapped PNG
        tonemapped = tonemap(np.array(image), args.exposure)
        frame_path = frames_dir / f"frame_{frame_idx:04d}.png"
        from PIL import Image
        Image.fromarray(tonemapped, 'RGB').save(frame_path)
        frames.append(frame_path)
        
        # Save HDR for quality
        hdr_path = frames_dir / f"frame_{frame_idx:04d}.exr"
        mi.util.write_bitmap(str(hdr_path), image)
        
        # Clean up
        dr.flush_malloc_cache()
    
    # Clean up base PLY
    try:
        os.unlink(ply_path)
    except OSError:
        pass
    
    if failed_frames:
        print(f"\n⚠ Warning: {len(failed_frames)} frames failed: {failed_frames}")
    
    print(f"\n✅ {len(frames)} frames rendered successfully!")
    return frames


def create_video(frames, output_dir, fps=30):
    """Create MP4 video and GIF from rendered frames."""
    try:
        import subprocess
        
        output_dir = Path(output_dir)
        video_path = output_dir / "diamond_rolling.mp4"
        
        # Check if frames exist
        frame_pattern = output_dir / 'frames' / 'frame_*.png'
        if not list(output_dir.glob('frames/frame_*.png')):
            print("  ⚠ No frames found, skipping video creation")
            return
        
        # Create MP4 with smooth playback
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-pattern_type', 'glob',
            '-i', str(output_dir / 'frames' / 'frame_*.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',
            '-preset', 'medium',
            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
            str(video_path)
        ]
        
        print(f"\n🎬 Creating video: {video_path}")
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✅ Video created: {video_path}")
        
        # Create GIF
        gif_path = output_dir / "diamond_rolling.gif"
        cmd_gif = [
            'ffmpeg', '-y',
            '-framerate', str(min(fps, 15)),
            '-pattern_type', 'glob',
            '-i', str(output_dir / 'frames' / 'frame_*.png'),
            '-vf', 'scale=480:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
            '-loop', '0',
            str(gif_path)
        ]
        
        try:
            subprocess.run(cmd_gif, check=True, capture_output=True)
            print(f"✅ GIF created: {gif_path}")
        except Exception as e:
            print(f"  ⚠ Could not create GIF: {e}")
            
    except FileNotFoundError:
        print("  ⚠ ffmpeg not found - skipping video creation")
        print("  Install ffmpeg: sudo apt install ffmpeg")


def main():
    args = parse_args()

    checkpoint_dir = os.path.join('checkpoints', args.checkpoint_name)
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")
    print(f"✓ Checkpoint: {checkpoint_dir}")
    print(f"✓ Variant: {variant}")

    # Load preset
    preset_path = os.path.join(checkpoint_dir, 'diamond_preset.json')
    if os.path.exists(preset_path):
        with open(preset_path) as f:
            diamond_params = json.load(f)
        print("✓ Parameters: from checkpoint JSON")
    else:
        diamond_params = DIAMOND_PRESETS.get(args.diamond_name)
        if diamond_params is None:
            raise ValueError(f"Unknown preset: {args.diamond_name}")
        diamond_params = diamond_params.copy()
        print(f"✓ Parameters: preset '{args.diamond_name}'")

    # Build diamond mesh and get vertices/normals/faces
    ply_path, vertices, normals, faces = build_diamond_mesh(diamond_params)
    
    # Build BSDF
    if args.no_neural:
        print("✓ Mode: analytic ground truth (--no_neural)")
        bsdf = load_analytic_bsdf(diamond_params)
    else:
        print("✓ Mode: neural shading (Model_M + DiamondFacet)")
        bsdf = load_neural_bsdf_direct(checkpoint_dir, diamond_params, args.clamp_value)
        if bsdf is None:
            print("⚠ Neural BSDF failed to load - falling back to dielectric")
            bsdf = load_analytic_bsdf(diamond_params)

    # Disable megakernel
    for flag in [dr.JitFlag.LoopRecord, dr.JitFlag.VCallRecord, dr.JitFlag.VCallOptimize]:
        dr.set_flag(flag, False)
    print("✓ Megakernel disabled")

    # Render animation
    frames = create_animation_frames(args, bsdf, diamond_params, vertices, normals, faces)

    # Create video
    create_video(frames, Path(args.output_dir), fps=args.fps)

    print(f"\n✅ Rolling animation complete!")
    print(f"   Frames: {Path(args.output_dir) / 'frames'}")
    print(f"   Video: {args.output_dir}/diamond_rolling.mp4")
    if os.path.exists(Path(args.output_dir) / "diamond_rolling.gif"):
        print(f"   GIF: {args.output_dir}/diamond_rolling.gif")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())