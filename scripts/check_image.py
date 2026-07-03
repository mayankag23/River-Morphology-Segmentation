from pathlib import Path
import rasterio

image = Path("outputs/demo_dataset/scenes/demo_scene/image.tif")

with rasterio.open(image) as ds:
    print("Width :", ds.width)
    print("Height:", ds.height)
    print("Bands :", ds.count)
    print("CRS   :", ds.crs)