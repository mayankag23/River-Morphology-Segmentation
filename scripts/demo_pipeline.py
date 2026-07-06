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

#################################################
# ==========================================================
# Module 9 - Pseudo Label Generation
# ==========================================================

from src.labels.generator import PseudoLabelGenerator
from src.labels.renderer import DemoRenderer
from src.labels.schema import ClassDefinition, ClassSchema

section("Module 9 - Pseudo Label Generation")

schema = ClassSchema(
    classes=(
        ClassDefinition(0, "background", (128, 128, 128)),
        ClassDefinition(1, "water", (0, 119, 190)),
        ClassDefinition(2, "sand", (255, 200, 87)),
        ClassDefinition(3, "vegetation", (34, 139, 34)),
    )
)

generator = PseudoLabelGenerator.from_config(
    config,
    schema,
)

# patch_dir = patch_result.scene_patches_dir

# patch_path = patch_dir / "demo_scene_r000_c000.tif"

# mask_path = patch_dir / "demo_scene_r000_c000_mask.tif"

# label_result = generator.generate(
#     patch_path=patch_path,
#     patch_id="demo_scene_r000_c000",
#     output_path=mask_path,
# )


patch_dir = patch_result.scene_patches_dir

patch_files = sorted(patch_dir.glob("*.tif"))

# Ignore mask files if they already exist
patch_files = [p for p in patch_files if "_mask" not in p.stem]

patch_path = patch_files[0]

mask_path = patch_dir / f"{patch_path.stem}_mask.tif"
label_result = generator.generate(
    patch_path=patch_path,
    patch_id=patch_path.stem,
    output_mask_path=mask_path,
)

ok("Pseudo Labels Generated")

print()

print("Patch ID              :", label_result.patch_id)
print("Mask Path             :", label_result.mask_path)
print("Mask Confidence       :", f"{label_result.mask_confidence:.3f}")
print("Quality Score         :", f"{label_result.quality_score:.3f}")
print("Acceptable            :", label_result.is_acceptable)
print("Classes Present       :", label_result.num_classes_present)
print("Valid Pixel Ratio     :", f"{label_result.valid_pixel_ratio:.2%}")
print("Unclassified Ratio    :", f"{label_result.unclassified_ratio:.2%}")
print("Spectral Indices Used :", ", ".join(label_result.spectral_indices_used))
print("CRS                   :", label_result.crs)

if label_result.issues:
    print("Issues                :", ", ".join(label_result.issues))
else:
    print("Issues                : None")

# ==========================================================
# Demo Visualization
# ==========================================================

section("Visualization")

renderer = DemoRenderer(
    output_dir=Path("outputs/demo_visualizations"),
)

# Generate only the visualizations that don't require a confidence map
renderer.render_rgb(patch_path)

renderer.render_pseudo_labels(mask_path)

renderer.render_overlay(
    image_path=patch_path,
    mask_path=mask_path,
)

renderer.render_patch_gallery(
    patch_directory=patch_result.scene_patches_dir,
)

print()

ok("Visualizations Generated")

print()

print("Generated Visualizations")
print(SUBLINE)

viz_dir = Path("outputs/demo_visualizations")

for file in sorted(viz_dir.glob("*.png")):
    print(file.name)

print()

print("Saved to")

print(viz_dir.resolve())