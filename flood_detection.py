import rasterio
import numpy as np
import matplotlib.pyplot as plt
BEFORE_PATH = "before_flood_kochi.tif"   
AFTER_PATH  = "during_flood_kochi.tif"                            
THRESHOLD_DB = -17.0

def load_band(path, band_index=1):
    with rasterio.open(path) as src:
        print(f"'{path}' has {src.count} band(s): {src.descriptions}")
        data = src.read(band_index).astype(np.float32)
        meta = src.meta.copy()
        meta.update(count=1) 
    return data, meta

def to_db(linear_array):
    linear_array = np.where(linear_array <= 0, 1e-6, linear_array)
    return 10 * np.log10(linear_array)

def detect_water(db_array, threshold=THRESHOLD_DB):
    return (db_array < threshold).astype(np.uint8)

def main():
    print("Loading before-flood image...")
    before_raw, meta = load_band(BEFORE_PATH, band_index=2)

    print("Loading during-flood image...")
    after_raw, _ = load_band(AFTER_PATH, band_index=2)

    before_db = to_db(before_raw)
    after_db = to_db(after_raw)

    water_before = detect_water(before_db)
    water_after = detect_water(after_db)

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

    flood_map = np.where((water_after == 1) & (water_before == 0), 1, 0).astype(np.uint8)
    out_meta = meta.copy()
    out_meta.update(dtype=rasterio.uint8, count=1, nodata=0)

    with rasterio.open("flood_map.tif", "w", **out_meta) as dst:
        dst.write(flood_map, 1)

    print("Saved flood_map.tif")
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
    total_pixels = flood_map.size
    flooded_pixels = flood_map.sum()
    print(f"\nFlooded pixels: {flooded_pixels} / {total_pixels} "
          f"({100 * flooded_pixels / total_pixels:.2f}% of image)")

if __name__ == "__main__":
    main()
