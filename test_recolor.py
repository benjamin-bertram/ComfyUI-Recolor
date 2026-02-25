#!/usr/bin/env python3
"""
Standalone Product Recoloring Script
=====================================
Test the LAB recoloring approach independently of ComfyUI.

Usage:
    python test_recolor.py --image product.png --target_rgb 66,17,34

    # Multi-zone with auto segmentation:
    python test_recolor.py --image jacket.png --zones 3 --target_rgbs "140,137,189;200,193,219;54,45,77"

    # Batch from config:
    python test_recolor.py --image pant.png --config colorway_configs.json --article article_selection_2_d4t_knit_pant

Requirements:
    pip install scikit-image opencv-python numpy Pillow rembg onnxruntime
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image
from skimage import color as skcolor


def remove_background_get_mask(img_np):
    """
    Extract product mask using rembg (U2-Net based background removal).
    Falls back to simple thresholding if rembg not available.
    """
    try:
        from rembg import remove
        pil_img = Image.fromarray(img_np)
        result = remove(pil_img)
        # Alpha channel = product mask
        result_np = np.array(result)
        if result_np.shape[2] == 4:
            mask = result_np[:, :, 3].astype(np.float32) / 255.0
        else:
            mask = np.ones(img_np.shape[:2], dtype=np.float32)
        return mask
    except ImportError:
        print("⚠️  rembg not installed. Using GrabCut fallback.")
        print("   Install with: pip install rembg onnxruntime")
        return grabcut_mask(img_np)


def grabcut_mask(img_np):
    """Fallback mask extraction using GrabCut."""
    mask = np.zeros(img_np.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    h, w = img_np.shape[:2]
    rect = (int(w * 0.05), int(h * 0.05), int(w * 0.9), int(h * 0.9))
    
    cv2.grabCut(img_np, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask_out = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.float32)
    return mask_out


def auto_segment_zones(img_np, product_mask, num_zones=3):
    """
    K-Means clustering in LAB space to find color zones within the product.
    Returns list of zone masks and their average RGB values.
    """
    mask_bool = product_mask > 0.5
    
    img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    pixels_lab = img_lab[mask_bool].reshape(-1, 3).astype(np.float32)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 0.5)
    _, labels, centers = cv2.kmeans(
        pixels_lab, num_zones, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    
    label_map = np.full(img_np.shape[:2], -1, dtype=np.int32)
    label_map[mask_bool] = labels.flatten()
    
    zone_masks = []
    zone_info = []
    
    for i in range(num_zones):
        zone_mask = (label_map == i).astype(np.float32)
        
        # Clean up with morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        zone_mask = cv2.morphologyEx(zone_mask, cv2.MORPH_CLOSE, kernel)
        zone_mask = cv2.morphologyEx(zone_mask, cv2.MORPH_OPEN, kernel)
        
        zone_masks.append(zone_mask)
        
        # Average color
        zone_pixels = img_np[label_map == i]
        if len(zone_pixels) > 0:
            avg = zone_pixels.mean(axis=0).astype(int)
            pct = len(zone_pixels) / mask_bool.sum() * 100
            zone_info.append({
                "zone": i,
                "avg_rgb": [int(avg[0]), int(avg[1]), int(avg[2])],
                "pixel_pct": round(pct, 1)
            })
            print(f"   Zone {i}: avg RGB({avg[0]}, {avg[1]}, {avg[2]}) — {pct:.1f}% of product")
    
    return zone_masks, zone_info


def recolor_lab(img_np, mask, target_r, target_g, target_b,
                luminance_blend=0.3, edge_feather=3):
    """
    Core recoloring function using LAB color space.
    
    Strategy:
    - Convert to LAB
    - Shift a* and b* channels to match target color
    - Optionally blend L* channel
    - Preserve all luminance variation (shadows, texture, folds)
    """
    img_float = img_np.astype(np.float64) / 255.0
    
    # Feather mask edges
    if edge_feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), edge_feather)
    mask = np.clip(mask, 0, 1)
    
    # Convert to LAB
    img_lab = skcolor.rgb2lab(img_float)
    
    # Target color in LAB
    target_norm = np.array([[[target_r / 255.0, target_g / 255.0, target_b / 255.0]]])
    target_lab = skcolor.rgb2lab(target_norm.astype(np.float64))[0, 0]
    target_L, target_a, target_b_ch = target_lab
    
    out_lab = img_lab.copy()
    
    mask_bool = mask > 0.1
    if mask_bool.sum() == 0:
        return img_np
    
    # Current color stats in masked region
    current_L_mean = img_lab[mask_bool, 0].mean()
    current_a_mean = img_lab[mask_bool, 1].mean()
    current_b_mean = img_lab[mask_bool, 2].mean()
    
    # Shift a* and b* to target (preserving variation)
    a_shift = target_a - current_a_mean
    out_lab[:, :, 1] = img_lab[:, :, 1] + a_shift * mask
    
    b_shift = target_b_ch - current_b_mean
    out_lab[:, :, 2] = img_lab[:, :, 2] + b_shift * mask
    
    # Blend luminance
    if luminance_blend > 0:
        L_shift = (target_L - current_L_mean) * luminance_blend
        out_lab[:, :, 0] = img_lab[:, :, 0] + L_shift * mask
    
    # Clamp
    out_lab[:, :, 0] = np.clip(out_lab[:, :, 0], 0, 100)
    out_lab[:, :, 1] = np.clip(out_lab[:, :, 1], -128, 127)
    out_lab[:, :, 2] = np.clip(out_lab[:, :, 2], -128, 127)
    
    # Back to RGB
    out_rgb = skcolor.lab2rgb(out_lab)
    out_rgb = (np.clip(out_rgb, 0, 1) * 255).astype(np.uint8)
    
    return out_rgb


def process_single(img_np, product_mask, target_rgb, luminance_blend=0.3, feather=3):
    """Process a single-zone recolor."""
    r, g, b = target_rgb
    return recolor_lab(img_np, product_mask, r, g, b, luminance_blend, feather)


def process_multizone(img_np, product_mask, zones_config, num_zones=3,
                      luminance_blend=0.3, feather=3):
    """Process multi-zone recolor with auto segmentation."""
    print(f"\n🔍 Auto-segmenting into {num_zones} zones...")
    zone_masks, zone_info = auto_segment_zones(img_np, product_mask, num_zones)
    
    result = img_np.copy()
    for zone in zones_config:
        idx = zone["mask_index"]
        if idx < len(zone_masks):
            print(f"   🎨 Recoloring zone {idx} ({zone.get('label', '')}) → "
                  f"RGB({zone['r']}, {zone['g']}, {zone['b']})")
            result = recolor_lab(result, zone_masks[idx],
                                 zone["r"], zone["g"], zone["b"],
                                 luminance_blend, feather)
    
    return result, zone_masks


def create_comparison_grid(original, recolored_variants, names, max_per_row=4):
    """Create a comparison grid image."""
    h, w = original.shape[:2]
    
    # Add original at start
    all_images = [original] + recolored_variants
    all_names = ["Original"] + names
    
    n = len(all_images)
    cols = min(n, max_per_row)
    rows = (n + cols - 1) // cols
    
    # Scale down for grid
    thumb_w = min(w, 400)
    thumb_h = int(h * thumb_w / w)
    label_h = 40
    
    grid = np.ones((rows * (thumb_h + label_h), cols * thumb_w, 3), dtype=np.uint8) * 255
    
    for i, (img, name) in enumerate(zip(all_images, all_names)):
        row = i // cols
        col = i % cols
        
        thumb = cv2.resize(img, (thumb_w, thumb_h))
        y = row * (thumb_h + label_h)
        x = col * thumb_w
        grid[y:y + thumb_h, x:x + thumb_w] = thumb
        
        # Label
        cv2.putText(grid, name,
                    (x + 10, y + thumb_h + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return grid


def main():
    parser = argparse.ArgumentParser(description="Product Photo Recoloring")
    parser.add_argument("--image", required=True, help="Source product image")
    parser.add_argument("--target_rgb", help="Target RGB, e.g. '66,17,34'")
    parser.add_argument("--zones", type=int, default=1, help="Number of color zones")
    parser.add_argument("--target_rgbs", help="Multi-zone targets, semicolon-separated")
    parser.add_argument("--config", help="JSON config file path")
    parser.add_argument("--article", help="Article key in config file")
    parser.add_argument("--luminance_blend", type=float, default=0.3)
    parser.add_argument("--feather", type=int, default=3)
    parser.add_argument("--output_dir", default="./output")
    
    args = parser.parse_args()
    
    # Load image
    print(f"📷 Loading image: {args.image}")
    img = np.array(Image.open(args.image).convert("RGB"))
    
    # Get product mask
    print("🔍 Extracting product mask...")
    product_mask = remove_background_get_mask(img)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save mask preview
    mask_preview = (product_mask * 255).astype(np.uint8)
    Image.fromarray(mask_preview).save(os.path.join(args.output_dir, "mask_preview.png"))
    print(f"   ✅ Mask saved to {args.output_dir}/mask_preview.png")
    
    if args.config and args.article:
        # Batch mode from config
        print(f"\n📋 Loading config: {args.config}")
        with open(args.config) as f:
            config = json.load(f)
        
        article = config[args.article]
        print(f"   Article: {article['description']}")
        print(f"   Zones: {article['zones']}")
        print(f"   Colorways: {len(article['colorways'])}")
        
        num_zones = len(article["zones"])
        variants = []
        names = []
        
        for cw in article["colorways"]:
            print(f"\n🎨 Processing colorway: {cw['name']} ({cw['season']})")
            
            if num_zones > 1:
                result, _ = process_multizone(
                    img, product_mask, cw["zones"], num_zones,
                    args.luminance_blend, args.feather
                )
            else:
                zone = cw["zones"][0]
                result = process_single(
                    img, product_mask, (zone["r"], zone["g"], zone["b"]),
                    args.luminance_blend, args.feather
                )
            
            # Save individual result
            out_path = os.path.join(args.output_dir, f"{cw['name']}.png")
            Image.fromarray(result).save(out_path)
            print(f"   ✅ Saved: {out_path}")
            
            variants.append(result)
            names.append(f"{cw['name']} ({cw['season']})")
        
        # Create comparison grid
        grid = create_comparison_grid(img, variants, names)
        grid_path = os.path.join(args.output_dir, "comparison_grid.png")
        Image.fromarray(grid).save(grid_path)
        print(f"\n✅ Comparison grid saved: {grid_path}")
    
    elif args.target_rgb:
        # Single target
        rgb = [int(x.strip()) for x in args.target_rgb.split(",")]
        print(f"\n🎨 Recoloring to RGB({rgb[0]}, {rgb[1]}, {rgb[2]})")
        result = process_single(img, product_mask, rgb, args.luminance_blend, args.feather)
        out_path = os.path.join(args.output_dir, "recolored.png")
        Image.fromarray(result).save(out_path)
        print(f"✅ Saved: {out_path}")
    
    elif args.target_rgbs:
        # Multi-zone
        rgb_sets = []
        for s in args.target_rgbs.split(";"):
            parts = [int(x.strip()) for x in s.split(",")]
            rgb_sets.append({"mask_index": len(rgb_sets), "r": parts[0], "g": parts[1], "b": parts[2]})
        
        result, zone_masks = process_multizone(
            img, product_mask, rgb_sets, args.zones,
            args.luminance_blend, args.feather
        )
        
        out_path = os.path.join(args.output_dir, "recolored_multizone.png")
        Image.fromarray(result).save(out_path)
        print(f"\n✅ Saved: {out_path}")
        
        # Save zone masks
        for i, zm in enumerate(zone_masks):
            zm_path = os.path.join(args.output_dir, f"zone_mask_{i}.png")
            Image.fromarray((zm * 255).astype(np.uint8)).save(zm_path)
    
    else:
        # Just analyze - show zones
        print("\n🔍 Analyzing product zones (no target specified)...")
        if args.zones > 1:
            zone_masks, zone_info = auto_segment_zones(img, product_mask, args.zones)
            
            # Save color-coded zone map
            colors = [
                [255, 0, 0], [0, 255, 0], [0, 0, 255],
                [255, 255, 0], [255, 0, 255], [0, 255, 255],
                [128, 0, 255], [255, 128, 0]
            ]
            zone_viz = img.copy().astype(np.float32)
            for i, zm in enumerate(zone_masks):
                c = np.array(colors[i % len(colors)], dtype=np.float32)
                zone_viz = zone_viz + (zm[:, :, np.newaxis] * c * 0.4)
            zone_viz = np.clip(zone_viz, 0, 255).astype(np.uint8)
            
            viz_path = os.path.join(args.output_dir, "zone_visualization.png")
            Image.fromarray(zone_viz).save(viz_path)
            print(f"\n✅ Zone visualization saved: {viz_path}")
            print(f"   Zone info: {json.dumps(zone_info, indent=2)}")
        else:
            print("   Use --zones N to detect color zones")
            print("   Use --target_rgb R,G,B for single-zone recolor")
            print("   Use --config + --article for batch processing")


if __name__ == "__main__":
    main()
