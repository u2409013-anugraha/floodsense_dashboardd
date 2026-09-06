"""
FloodSense - Flood Detection using Sentinel-1 SAR Backscatter Thresholding
----------------------------------------------------------------------------
Compares a before-flood and during-flood SAR image (VV band, GeoTIFF,
exported from SNAP after calibration + speckle filter + terrain correction)
and produces a binary flood extent map.

Requirements:
    pip install rasterio numpy matplotlib --break-system-packages
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# STEP 1: Set your file paths
# Replace these with the actual GeoTIFF files you export from SNAP
# (File -> Export -> GeoTIFF, after calibration + speckle filter + terrain correction)
# ---------------------------------------------------------------------------
BEFORE_PATH = "before_flood_alappuzha.tif"   # Aug 9 image, VV band
AFTER_PATH  = "during_flood_alappuzha.tif"                            # Aug 21 image, VV band

# Threshold in dB. Water typically has backscatter below about -17 to -15 dB.
# You may need to tune this after visually inspecting your histogram (Step 4 below).
THRESHOLD_DB = -17.0


def load_band(path, band_index=1):
    """
    Load a GeoTIFF band and return the array + metadata.

    IMPORTANT: your exported GeoTIFF may contain BOTH VV and VH bands
    (since SNAP calibration often keeps both by default), not just one.
    This function prints how many bands it found so you can check you're
    reading the right one. VV is usually listed before VH, but always
    verify against SNAP's Band names (check the 'Bands' folder for each
    product in Product Explorer, e.g. Sigma0_VV vs Sigma0_VH).
    """
    with rasterio.open(path) as src:
        print(f"'{path}' has {src.count} band(s): {src.descriptions}")
        data = src.read(band_index).astype(np.float32)
        meta = src.meta.copy()
        meta.update(count=1)  # we'll only ever write 1 band back out
    return data, meta


def to_db(linear_array):
    """
    Convert linear backscatter (sigma naught) to decibels.
    If your SNAP export was already in dB, skip this step.
    """
    # Avoid log(0) errors
    linear_array = np.where(linear_array <= 0, 1e-6, linear_array)
    return 10 * np.log10(linear_array)


def detect_water(db_array, threshold=THRESHOLD_DB):
    """Return a binary mask: 1 = water, 0 = not water."""
    return (db_array < threshold).astype(np.uint8)


def main():
    # NOTE: band_index=1 assumes VV is the first band in your GeoTIFF.
    # Check the printed band info below when you run this script -- if it
    # looks wrong (e.g. VH is actually first), change band_index to 2.
    print("Loading before-flood image...")
    before_raw, meta = load_band(BEFORE_PATH, band_index=2)

    print("Loading during-flood image...")
    after_raw, _ = load_band(AFTER_PATH, band_index=2)

    # --- STEP 2: Convert to dB (skip this line if already in dB from SNAP) ---
    before_db = to_db(before_raw)
    after_db = to_db(after_raw)

    # --- STEP 3: Classify water in each image ---
    water_before = detect_water(before_db)
    water_after = detect_water(after_db)

    # --- STEP 4 (recommended): Inspect the histogram to pick a good threshold ---
    plt.figure(figsize=(8, 4))
    plt.hist(after_db.flatten(), bins=100, range=(-30, 5))
    plt.axvline(THRESHOLD_DB, color="red", linestyle="--", label=f"Threshold = {THRESHOLD_DB} dB")
    plt.title("Backscatter histogram (during-flood image)")
    plt.xlabel("Backscatter (dB)")
    plt.ylabel("Pixel count")
    plt.legend()
    plt.tight_layout()
    plt.savefig("histogram_check.png", dpi=150)
    print("Saved histogram_check.png -- inspect this to confirm/tune your threshold")

    # --- STEP 5: Compute NEW flood extent (water now, but not before) ---
    flood_map = np.where((water_after == 1) & (water_before == 0), 1, 0).astype(np.uint8)

    # --- STEP 6: Save the flood map as a GeoTIFF (for use in QGIS) ---
    out_meta = meta.copy()
    out_meta.update(dtype=rasterio.uint8, count=1, nodata=0)

    with rasterio.open("flood_map_alappuzha.tif", "w", **out_meta) as dst:
        dst.write(flood_map, 1)

    print("Saved flood_map_alappuzha.tif")

    # --- STEP 7: Quick visual check ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(before_db, cmap="gray", vmin=-25, vmax=0)
    axes[0].set_title("Before Flood (dB)")
    axes[1].imshow(after_db, cmap="gray", vmin=-25, vmax=0)
    axes[1].set_title("During Flood (dB)")
    axes[2].imshow(flood_map, cmap="Blues")
    axes[2].set_title("Detected New Flood Extent")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("flood_detection_result.png", dpi=150)
    print("Saved flood_detection_result.png")

    # --- Summary stats ---
    total_pixels = flood_map.size
    flooded_pixels = flood_map.sum()
    print(f"\nFlooded pixels: {flooded_pixels} / {total_pixels} "
          f"({100 * flooded_pixels / total_pixels:.2f}% of image)")


if __name__ == "__main__":
    main()
