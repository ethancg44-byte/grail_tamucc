#!/usr/bin/env python3
"""Run segmentation model on real microgreen photos and generate presentation figure."""

import cv2
import numpy as np
import torch
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
import matplotlib.pyplot as plt
from pathlib import Path

# --- Config ---
CHECKPOINT = "checkpoints/run33_cetversky_stemsamp_768/best_model.pth"
PHOTO_DIR = Path("microgreen_photos_jpg")
OUTPUT_DIR = Path("exports/microgreen_results")
INPUT_SIZE = 768

# Iterative stem growing parameters
STEM_HSV_LOW = np.array([25, 30, 50])    # green-yellow stems (25 cuts brown soil)
STEM_HSV_HIGH = np.array([60, 160, 210])
EDGE_PROXIMITY_PX = 5       # stem candidates must be near a Canny edge
GROW_KERNEL_W = 11           # dilation kernel width
GROW_KERNEL_H = 21           # dilation kernel height (downward-biased)
GROW_ANCHOR_Y = 5            # anchor near top -> 3:1 downward bias
MAX_GROW_ITERS = 60
MIN_NEW_PIXELS = 5
MAX_STEM_WIDTH_PX = 20       # erode components wider than this

SELECTED_PHOTOS = [
    "4480402392920505571.jpg",     # clean BG, great leaf+stem
    "338695527286679407.jpg",      # strong detection, clear
    "4524876660580090774.jpg",     # clean, good classification
    "8334210251397774536.jpg",     # very clean, no BG noise
]

# ImageNet normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Color map (RGB for matplotlib)
COLOR_MAP_RGB = {
    0: (0, 0, 0),           # BACKGROUND — black
    1: (0, 200, 0),         # LEAF — green
    2: (255, 140, 0),       # STEM — orange
}


def postprocess_stems(img_rgb_resized, probs):
    """Iterative stem growing from leaf regions using color + edge cues.

    Args:
        img_rgb_resized: RGB image resized to INPUT_SIZE x INPUT_SIZE (uint8)
        probs: softmax probabilities, shape (3, H, W)

    Returns:
        pred_mask: uint8 array (H, W) with 0=BG, 1=LEAF, 2=STEM
    """
    h, w = probs.shape[1], probs.shape[2]

    # Base predictions
    pred_mask = np.argmax(probs, axis=0).astype(np.uint8)
    leaf_mask = (pred_mask == 1)
    known_stem = (pred_mask == 2)

    # --- Pre-compute stem candidates (color + edge, not leaf) ---
    img_bgr = cv2.cvtColor(img_rgb_resized, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    color_ok = cv2.inRange(hsv, STEM_HSV_LOW, STEM_HSV_HIGH) > 0

    # Edge proximity: Canny + dilate to get "near edge" mask
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_dilated = cv2.dilate(edges, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * EDGE_PROXIMITY_PX + 1, 2 * EDGE_PROXIMITY_PX + 1)))
    near_edge = edge_dilated > 0

    # Green-dominance: plant tissue has G > R, soil doesn't
    green_dominant = img_rgb_resized[:, :, 1] > img_rgb_resized[:, :, 0]

    stem_candidates = color_ok & near_edge & green_dominant & ~leaf_mask

    # --- Downward-biased dilation kernel ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (GROW_KERNEL_W, GROW_KERNEL_H))
    anchor = (GROW_KERNEL_W // 2, GROW_ANCHOR_Y)

    # --- Iterative growing ---
    seed = (leaf_mask | known_stem).astype(np.uint8)
    for _ in range(MAX_GROW_ITERS):
        dilated = cv2.dilate(seed, kernel, anchor=anchor)
        frontier = (dilated > 0) & (seed == 0)
        new_stem = frontier & stem_candidates
        n_new = int(new_stem.sum())
        if n_new < MIN_NEW_PIXELS:
            break
        known_stem |= new_stem
        seed = (known_stem | leaf_mask).astype(np.uint8)

    # --- Cleanup: keep only stem components connected to leaves ---
    stem_u8 = known_stem.astype(np.uint8)
    # Dilate leaves slightly to bridge small gaps for connectivity check
    leaf_dilated = cv2.dilate(leaf_mask.astype(np.uint8),
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n_labels, labels = cv2.connectedComponents(stem_u8)
    for lbl in range(1, n_labels):
        component = (labels == lbl)
        # Keep only if this component touches a dilated leaf region
        if not np.any(component & (leaf_dilated > 0)):
            known_stem[component] = False

    # Morphological close to fill tiny gaps
    known_stem = cv2.morphologyEx(known_stem.astype(np.uint8),
                                  cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))) > 0

    # Width filter: erode away overly wide blobs
    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (MAX_STEM_WIDTH_PX, MAX_STEM_WIDTH_PX))
    too_wide = cv2.erode(known_stem.astype(np.uint8), erode_k) > 0
    known_stem = known_stem & ~too_wide

    # Assemble final mask
    final_mask = np.zeros((h, w), dtype=np.uint8)
    final_mask[leaf_mask] = 1
    final_mask[known_stem] = 2
    return final_mask


def preprocess(img_bgr, input_size):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    img_float = img_resized.astype(np.float32) / 255.0
    img_norm = (img_float - IMAGENET_MEAN) / IMAGENET_STD
    return img_norm.transpose(2, 0, 1)[np.newaxis, ...]


def colorize_mask_rgb(mask):
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for label_id, rgb in COLOR_MAP_RGB.items():
        color[mask == label_id] = rgb
    return color


def overlay_rgb(img_rgb, mask_color, alpha=0.5):
    """Overlay colored mask on image, keeping original colors where background."""
    img_resized = cv2.resize(img_rgb, (mask_color.shape[1], mask_color.shape[0]))
    # Only blend where mask is non-black (i.e., plant regions)
    mask_nonzero = (mask_color.sum(axis=2) > 0).astype(np.float32)[..., np.newaxis]
    blended = img_resized.astype(np.float32) * (1 - alpha * mask_nonzero) + \
              mask_color.astype(np.float32) * alpha * mask_nonzero
    return np.clip(blended, 0, 255).astype(np.uint8)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load PyTorch model
    print(f"Loading PyTorch checkpoint: {CHECKPOINT}")
    model = lraspp_mobilenet_v3_large(num_classes=3)
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()

    # Process selected photos
    results = []
    for fname in SELECTED_PHOTOS:
        img_path = PHOTO_DIR / fname
        if not img_path.exists():
            print(f"  WARNING: {fname} not found, skipping")
            continue

        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Run inference
        input_tensor = preprocess(img_bgr, INPUT_SIZE)
        tensor = torch.from_numpy(input_tensor).float()
        with torch.no_grad():
            probs = torch.softmax(model(tensor)["out"], dim=1)[0].numpy()

        # Iterative stem growing post-processing
        img_resized = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE))
        pred_mask = postprocess_stems(img_resized, probs)

        # Colorize and overlay
        mask_color = colorize_mask_rgb(pred_mask)
        overlay_img = overlay_rgb(img_rgb, mask_color)

        results.append({
            "name": fname,
            "original": img_rgb,
            "mask": mask_color,
            "overlay": overlay_img,
        })

        # Save individual overlay
        cv2.imwrite(
            str(OUTPUT_DIR / f"{Path(fname).stem}_overlay.png"),
            cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR),
        )
        print(f"  Processed {fname}")

    if not results:
        print("No images processed!")
        return

    # --- Generate presentation figure ---
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(18, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Original Photo", "Segmentation Mask", "Overlay"]

    for i, r in enumerate(results):
        # Original (resize for display)
        h_orig, w_orig = r["original"].shape[:2]
        scale = INPUT_SIZE / max(h_orig, w_orig)
        display_orig = cv2.resize(r["original"], (int(w_orig * scale), int(h_orig * scale)))

        axes[i, 0].imshow(display_orig)
        axes[i, 1].imshow(r["mask"])
        axes[i, 2].imshow(r["overlay"])

        for j in range(3):
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(col_titles[j], fontsize=16, fontweight="bold", pad=10)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=(0, 0, 0), edgecolor="white", label="Background"),
        Patch(facecolor=(0, 200/255, 0), edgecolor="white", label="Leaf"),
        Patch(facecolor=(1, 140/255, 0), edgecolor="white", label="Stem"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=14, frameon=True, fancybox=True, shadow=True,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("LRASPP-MobileNetV3 Segmentation on Real Microgreen Photos",
                 fontsize=20, fontweight="bold", y=1.01)
    plt.tight_layout()

    out_path = OUTPUT_DIR / "microgreen_segmentation_demo.png"
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nPresentation figure saved: {out_path}")

    # Also save a compact 2-photo version for a single slide
    if len(results) >= 2:
        fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
        for i in range(2):
            r = results[i]
            h_orig, w_orig = r["original"].shape[:2]
            scale = INPUT_SIZE / max(h_orig, w_orig)
            display_orig = cv2.resize(r["original"], (int(w_orig * scale), int(h_orig * scale)))

            axes2[i, 0].imshow(display_orig)
            axes2[i, 1].imshow(r["mask"])
            axes2[i, 2].imshow(r["overlay"])
            for j in range(3):
                axes2[i, j].axis("off")
                if i == 0:
                    axes2[i, j].set_title(col_titles[j], fontsize=16, fontweight="bold", pad=10)

        fig2.legend(handles=legend_elements, loc="lower center", ncol=3,
                    fontsize=14, frameon=True, fancybox=True, shadow=True,
                    bbox_to_anchor=(0.5, -0.02))
        fig2.suptitle("LRASPP-MobileNetV3 Segmentation on Real Microgreen Photos",
                      fontsize=20, fontweight="bold", y=1.01)
        plt.tight_layout()
        out2 = OUTPUT_DIR / "microgreen_segmentation_slide.png"
        fig2.savefig(str(out2), dpi=200, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"Slide-ready figure saved: {out2}")


if __name__ == "__main__":
    main()
