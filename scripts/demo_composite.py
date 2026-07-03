# from pathlib import Path
# import sys

# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# sys.path.insert(0, str(PROJECT_ROOT))
# # from src.core.config import Config
# # from src.gee.collections import LandsatCollectionBuilder
# # from src.gee.preprocessing import LandsatPreprocessor
# # from src.gee.features import SpectralFeatureGenerator
# # from src.export.exporter import DatasetExporter
# # from src.patches.generator import PatchGenerator
# # from src.labels.manager import LabelManager
# # from src.dataset.assembler import DatasetAssembler

# # config = Config("config/config.yaml")

# # print("=== River Morphology Segmentation Demo ===")

# # # 1. Build collection
# # collection = (
# #     LandsatCollectionBuilder(config)
# #     .with_point(
# #         latitude=28.6139,
# #         longitude=77.2090,
# #         buffer_meters=5000,
# #     )
# #     .with_date_range(
# #         "2023-01-01",
# #         "2023-12-31",
# #     )
# #     .build()
# # )

# # print("✓ Collection created")

# # # 2. Preprocess
# # preprocessor = LandsatPreprocessor(config)
# # processed = preprocessor.process(collection)

# # print("✓ Preprocessing complete")

# # # 3. Composite
# # composite = preprocessor.build_composite(processed)

# # print("✓ Composite generated")

# # # 4. Features
# # generator = SpectralFeatureGenerator(config)
# # features = generator.generate(composite)

# # print("✓ Spectral features generated")

# # # 5. Export
# # exporter = DatasetExporter(config)
# # export_result = exporter.export(features)

# # print("✓ GeoTIFF exported")

# # # 6. Generate patches
# # patch_generator = PatchGenerator(config)
# # patch_result = patch_generator.generate(export_result)

# # print("✓ Patches generated")

# # # Modules 9–10 require ground-truth labels, so they can only be demonstrated
# # # once label masks are available.

# from src.core.config import Config
# from src.gee.client import EarthEngineClient
# from src.gee.collections import LandsatCollectionBuilder
# from src.gee.preprocessing import LandsatPreprocessor

# config = Config("config/config.yaml")

# client = EarthEngineClient(config)
# client.initialize()

# print("=" * 60)
# print("River Morphology Segmentation Demo")
# print("=" * 60)

# start = input("Start date (YYYY-MM-DD): ")
# end = input("End date   (YYYY-MM-DD): ")

# builder = (
#     LandsatCollectionBuilder(client, config)
#         .with_date_range(start, end)
#         .with_aoi_from_config()          # Uses AOI already in config.yaml
#         .with_cloud_cover_from_config()
#         .with_auto_sensors()
# )

# print("\nBuilding Landsat collection...\n")

# collection = builder.build(validate_not_empty=True)

# print("✓ Collection successfully created")
# print(collection)

# print("\nRunning preprocessing...\n")

# preprocessor = LandsatPreprocessor(client, config)

# processed = preprocessor.process(collection)

# print("✓ Preprocessing complete")
# print(processed)

# # builder = (
# #     LandsatCollectionBuilder(client, config)
# #         .with_date_range_from_config()
# #         .with_aoi_from_config()
# #         .with_cloud_cover_from_config()
# #         .with_auto_sensors()
# # )

# # collection = builder.build(validate_not_empty=True)

# # print(collection)

# # preprocessor = LandsatPreprocessor(client, config)

# # processed = preprocessor.process(collection)

# # print(processed)

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import Config
from src.gee.client import EarthEngineClient
from src.gee.collections import LandsatCollectionBuilder
from src.gee.preprocessing import LandsatPreprocessor


print("=" * 70)
print("River Morphology Segmentation Demo")
print("=" * 70)

config = Config("config/config.yaml")

client = EarthEngineClient(config)
client.initialize()

print("\nEnter acquisition dates\n")

# start_date = input("Start Date (YYYY-MM-DD): ").strip()
# end_date = input("End Date   (YYYY-MM-DD): ").strip()
start_date = "2023-01-01"
end_date = "2023-12-31"

print("\nBuilding Landsat collection...\n")

builder = (
    LandsatCollectionBuilder(client, config)
    .with_date_range(start_date, end_date)
    .with_aoi_from_config()
    .with_cloud_cover_from_config()
    .with_auto_sensors()
)

collection_result = builder.build(validate_not_empty=True)

print("=" * 70)
print("COLLECTION SUMMARY")
print("=" * 70)

for line in collection_result.summary_lines():
    print(line)

print("\nRunning preprocessing...\n")

preprocessor = LandsatPreprocessor(client, config)

processed_result = preprocessor.process(collection_result)

print("=" * 70)
print("PREPROCESSING SUMMARY")
print("=" * 70)

if hasattr(processed_result, "summary_lines"):
    for line in processed_result.summary_lines():
        print(line)
else:
    print(processed_result)

print("\nDemo finished successfully.")