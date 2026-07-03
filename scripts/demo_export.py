from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import Config
from src.gee.client import EarthEngineClient
from src.gee.collections import LandsatCollectionBuilder
from src.gee.preprocessing import LandsatPreprocessor
from src.gee.composite import LandsatCompositor
from src.gee.features import SpectralFeatureGenerator
from src.export.exporter import DatasetExporter


def main():

    print("=" * 70)
    print("Module 7 Demo - Dataset Export")
    print("=" * 70)

    config = Config("config/config.yaml")

    client = EarthEngineClient(config)
    client.initialize()

    start = "2023-01-01"
    end = "2023-12-31"

    print("\nBuilding Collection...")

    collection = (
        LandsatCollectionBuilder(client, config)
        .with_aoi_from_config()
        .with_auto_sensors()
        .with_cloud_cover_from_config()
        .with_date_range(start, end)
        .build(validate_not_empty=True)
    )

    print("[OK] Collection Built")

    preprocessor = LandsatPreprocessor(client, config)

    processed = preprocessor.process(collection)

    print("[OK] Preprocessing Complete")

    compositor = LandsatCompositor(client, config)

    composite = compositor.build_composite(processed)

    print("[OK] Composite Generated")

    generator = SpectralFeatureGenerator(client, config)

    feature_stack = generator.generate(composite)

    print("[OK] Spectral Features Generated")

    exporter = DatasetExporter(client, config)

    output_dir = Path("outputs/demo_dataset")

    export_result = exporter.export(
        feature_stack_result=feature_stack,
        output_dir=output_dir,
        scene_id="demo_scene",
        append_to_manifest=False,
    )

    print("\nExport Summary")
    print("-" * 70)

    for line in export_result.summary_lines():
        print(line)

    print("\nGenerated Files")

    print("Dataset Root :", export_result.dataset_root)
    print("Scene Folder :", export_result.scene_dir)
    print("Image        :", export_result.image_path)
    print("Metadata     :", export_result.metadata_path)
    print("Version File :", export_result.version_path)

    print("\nOperations")

    for op in export_result.operations_log:
        print(" -", op)

    print("\nDemo Complete")


if __name__ == "__main__":
    main()