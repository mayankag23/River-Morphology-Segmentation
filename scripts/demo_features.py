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


def main():

    print("=" * 70)
    print("Module 6 Demo - Spectral Feature Engineering")
    print("=" * 70)

    config = Config("config/config.yaml")

    client = EarthEngineClient(config)
    client.initialize()

    start_date = "2023-01-01"
    end_date = "2023-12-31"

    print("\nBuilding Landsat Collection...")

    collection = (
        LandsatCollectionBuilder(client, config)
        .with_aoi_from_config()
        .with_auto_sensors()
        .with_cloud_cover_from_config()
        .with_date_range(start_date, end_date)
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

    print("\nFeature Stack Summary")
    print("-" * 70)

    for line in feature_stack.summary_lines():
        print(line)

    print("\nBands Available")

    for band in feature_stack.all_band_names:
        print(f"  • {band}")

    print("\nIndices Computed")

    for feature in feature_stack.features_computed:
        print(f" - {feature}")

    print("\nTotal Bands :", len(feature_stack.all_band_names))

    print("\nDemo Complete")


if __name__ == "__main__":
    main()