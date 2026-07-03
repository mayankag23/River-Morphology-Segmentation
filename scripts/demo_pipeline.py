"""
River Morphology Segmentation
End-to-End Demonstration Pipeline

Modules Covered (Part 1)
------------------------
✓ Module 1 : Configuration
✓ Module 2 : Bootstrap
✓ Module 3 : Earth Engine Client
✓ Module 4 : Landsat Collection Builder
✓ Module 5 : Landsat Preprocessing

Author : Mayank
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import Config
from src.gee.client import EarthEngineClient
from src.gee.collections import LandsatCollectionBuilder
from src.gee.preprocessing import LandsatPreprocessor


# --------------------------------------------------------
# Pretty Printing
# --------------------------------------------------------

LINE = "=" * 72
SUBLINE = "-" * 72


def title(text: str):
    print()
    print(LINE)
    print(text)
    print(LINE)


def section(text: str):
    print()
    print(SUBLINE)
    print(text)
    print(SUBLINE)


def ok(msg: str):
    print(f"[OK] {msg}")


def info(msg: str):
    print(f"[*] {msg}")


# --------------------------------------------------------
# Main
# --------------------------------------------------------

def main():

    overall_start = time.time()

    title("River Morphology Segmentation Demo Pipeline")

    # ----------------------------------------------------
    # Module 1
    # ----------------------------------------------------

    section("Module 1 - Configuration")

    config = Config("config/config.yaml")

    ok("Configuration Loaded")
    # info(f"Project : {config.project_name}")

    # ----------------------------------------------------
    # Module 2 & 3
    # ----------------------------------------------------

    section("Module 2 / 3 - Earth Engine")

    client = EarthEngineClient(config)

    client.initialize()

    ok("Earth Engine Initialized")

    # ----------------------------------------------------
    # User Input
    # ----------------------------------------------------

    section("Input Parameters")

    # start_date = input(
    #     "Start Date [2023-01-01] : "
    # ).strip() or "2023-01-01"

    # end_date = input(
    #     "End Date   [2023-12-31] : "
    # ).strip() or "2023-12-31"
    start_date = "2023-11-01"
    end_date = "2024-03-31"

    # ----------------------------------------------------
    # Module 4
    # ----------------------------------------------------

    section("Module 4 - Building Landsat Collection")

    builder = (
        LandsatCollectionBuilder(client, config)
        .with_aoi_from_config()
        .with_auto_sensors()
        .with_cloud_cover_from_config()
        .with_date_range(start_date, end_date)
    )

    collection_result = builder.build(
        validate_not_empty=True
    )

    ok("Collection Created")

    print()

    for line in collection_result.summary_lines():
        print(line)

    # ----------------------------------------------------
    # Module 5
    # ----------------------------------------------------

    section("Module 5 - Landsat Preprocessing")

    preprocessor = LandsatPreprocessor(
        client,
        config,
    )

    processed_result = preprocessor.process(
        collection_result
    )

    ok("Preprocessing Completed")

    print()

    for line in processed_result.summary_lines():
        print(line)

    # ----------------------------------------------------
    # Store results for next modules
    # ----------------------------------------------------

#     pipeline = {
#         "config": config,
#         "client": client,
#         "collection": collection_result,
#         "processed": processed_result,
#         "start_time": overall_start,
#     }

#     return pipeline


# if __name__ == "__main__":

#     pipeline = main()

    return {
        "config": config,
        "client": client,
        "processed": processed_result,
        "start_time": overall_start,
    }


if __name__ == "__main__":

    pipeline = main()

    config = pipeline["config"]
    client = pipeline["client"]
    processed_result = pipeline["processed"]

from src.gee.composite import LandsatCompositor
from src.gee.features import SpectralFeatureGenerator
from src.export.exporter import DatasetExporter

section("Module 6 - Composite Generation")

compositor = LandsatCompositor(
    client,
    config,
)

composite_result = compositor.build_composite(
    processed_result
)

ok("Composite Generated")

print()

for line in composite_result.summary_lines():
    print(line)


section("Module 6 - Spectral Feature Engineering")

feature_generator = SpectralFeatureGenerator(
    client,
    config,
)

feature_stack_result = feature_generator.generate(
    composite_result
)

ok("Spectral Features Generated")

print()

for line in feature_stack_result.summary_lines():
    print(line)

section("Module 7 - Dataset Export")

output_dir = Path("outputs/demo_dataset")

exporter = DatasetExporter(
    client,
    config,
)

export_result = exporter.export(
    feature_stack_result=feature_stack_result,
    output_dir=output_dir,
    scene_id="demo_scene",
    append_to_manifest=False,
)

ok("Dataset Export Complete")

print()

for line in export_result.summary_lines():
    print(line)

print()

print("Dataset Root")

print(export_result.dataset_root)

print()

print("Scene Directory")

print(export_result.scene_dir)

print()

print("GeoTIFF")

print(export_result.image_path)

print()

print("Metadata")

print(export_result.metadata_path)

print()

print("Version")

print(export_result.version_path)

# ==========================================================
# Module 8 - Patch Generation
# ==========================================================

from src.patches.generator import PatchGenerator

section("Module 8 - Patch Generation")

patch_output_dir = Path("outputs/demo_patches")

patch_generator = PatchGenerator(config)

patch_result = patch_generator.generate(
    dataset_export_result=export_result,
    output_dir=patch_output_dir,
    append_to_manifest=False,
)

ok("Patch Generation Completed")

print()

for line in patch_result.summary_lines():
    print(line)

print()

print("Patch Generation Summary")
print(SUBLINE)

for line in patch_result.summary_lines():
    print(line)