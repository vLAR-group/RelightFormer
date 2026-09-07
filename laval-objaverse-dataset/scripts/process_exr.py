#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
import numpy as np
import OpenEXR
import Imath

# Scaling factors
INDOOR_SCALE = 50
OUTDOOR_SCALE = 300

def is_indoor_path(path: Path) -> bool:
    """
    Determine if the EXR file belongs to an indoor scene based on path.
    Customize this logic based on your actual folder naming convention.
    """
    path_str = str(path).lower()
    # Add more keywords if needed
    indoor_keywords = ['indoor', 'interior', 'room', 'house', 'building']
    outdoor_keywords = ['outdoor', 'exterior', 'sky', 'landscape', 'street']
    
    # If any indoor keyword appears and no outdoor keyword, treat as indoor
    has_indoor = any(kw in path_str for kw in indoor_keywords)
    has_outdoor = any(kw in path_str for kw in outdoor_keywords)
    
    if has_outdoor:
        return False
    if has_indoor:
        return True
    # Default to outdoor if unsure (safer for HDR lighting)
    return False

def read_exr(filepath):
    """Read EXR file and return as numpy array (H, W, C) in float32."""
    exr_file = OpenEXR.InputFile(str(filepath))
    header = exr_file.header()
    dw = header['dataWindow']
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1

    channels = []
    for ch_name in ['R', 'G', 'B', 'A']:
        if ch_name in header['channels']:
            channels.append(ch_name)
    
    if not channels:
        channel_names = list(header['channels'].keys())
        channels = channel_names[:3]

    pixel_data = []
    for ch in channels:
        data = exr_file.channel(ch, Imath.PixelType(Imath.PixelType.FLOAT))
        arr = np.frombuffer(data, dtype=np.float32).reshape(h, w)
        pixel_data.append(arr)
    
    img = np.stack(pixel_data, axis=-1)  # (H, W, C)
    return img

def write_exr(filepath, img):
    """Write numpy array (H, W, C) to EXR file."""
    h, w = img.shape[:2]
    channels = {}
    channel_names = ['R', 'G', 'B', 'A'][:img.shape[2]]
    
    for i, name in enumerate(channel_names):
        channels[name] = img[:, :, i].astype(np.float32).tobytes()
    
    exr = OpenEXR.OutputFile(str(filepath), OpenEXR.Header(w, h))
    exr.writePixels(channels)
    exr.close()

def rotate_equirectangular(img, shift_pixels):
    return np.roll(img, shift=shift_pixels, axis=1)

def resize_image(img, target_size):
    """Resize using OpenCV if available, else fallback to nearest neighbor."""
    h_old, w_old = img.shape[:2]
    w_new, h_new = target_size

    try:
        import cv2
        return cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        print("Warning: OpenCV not found. Using nearest-neighbor (low quality).")
        y_idx = (np.arange(h_new) * h_old / h_new).astype(int)
        x_idx = (np.arange(w_new) * w_old / w_new).astype(int)
        if img.ndim == 3:
            return img[np.ix_(y_idx, x_idx, np.arange(img.shape[2]))]
        else:
            return img[np.ix_(y_idx, x_idx)]

def main():
    parser = argparse.ArgumentParser(description="Generate rotated/resized EXR envmaps with indoor/outdoor scaling.")
    parser.add_argument("--input_path", type=str, default='./laval-objaverse-dataset/laval/src', help="Input root directory containing .exr files")
    parser.add_argument("--output_path", type=str, default='./laval-objaverse-dataset/laval/preprocessed', help="Output root directory")
    parser.add_argument("--num_versions", type=int, default=16, help="Number of rotated versions")
    parser.add_argument("--width", type=int, default=1024, help="Target width")
    parser.add_argument("--height", type=int, default=512, help="Target height")
    args = parser.parse_args()

    input_root = Path(args.input_path)
    output_root = Path(args.output_path)
    target_size = (args.width, args.height)

    if not input_root.exists():
        print(f"Error: Input path '{input_root}' does not exist.", file=sys.stderr)
        sys.exit(1)

    exr_files = list(input_root.rglob("*.exr"))
    if not exr_files:
        print(f"No .exr files found under '{input_root}'.")
        return

    print(f"Found {len(exr_files)} EXR files. Processing...\n")

    for i, src_path in enumerate(exr_files, 1):
        try:
            print(f"[{i}/{len(exr_files)}] {src_path.relative_to(input_root)}")

            # Determine scale based on path
            if is_indoor_path(src_path):
                scale = INDOOR_SCALE
                scene_type = "indoor"
            else:
                scale = OUTDOOR_SCALE
                scene_type = "outdoor"

            # Compute output path
            rel_path = src_path.relative_to(input_root)
            rel_dir = rel_path.parent
            filename_clean = src_path.stem.replace(' ', '-').replace('_', '-')
            new_subdir = output_root / rel_dir / filename_clean
            new_subdir.mkdir(parents=True, exist_ok=True)

            # Process image
            img = read_exr(src_path)
            img_resized = resize_image(img, target_size)
            img_scaled = img_resized * scale  # Apply scale

            H, W = img_scaled.shape[:2]
            for idx in range(args.num_versions):
                shift_px = int(round((idx / args.num_versions) * W))
                rotated = rotate_equirectangular(img_scaled, shift_px)
                out_file = new_subdir / f"{idx}.exr"
                write_exr(out_file, rotated)

            print(f"  → {scene_type} (scale={scale})")

        except Exception as e:
            print(f"Error processing {src_path}: {e}", file=sys.stderr)
            continue

    print("\n✅ All done!")

if __name__ == "__main__":
    main()