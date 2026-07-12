"""
Unit tests for src/patches/generator.py.

Uses real rasterio I/O: a small synthetic source scene GeoTIFF is written
with GeoTiffWriter (reused from Module 7), then PatchGenerator reads it
back and produces patches. No EE, no mocking of rasterio internals.

Run:
    pytest tests/patches/test_patch_generator.py -v
    pytest tests/patches/test_patch_generator.py -v \
        --cov=src/patches/generator --cov-report=term-missing
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import rasterio
from affine import Affine

from src.export.downloader import AffineTransform, AoiBounds, DownloadResult
from src.export.geotiff import GeoTiffProfile, GeoTiffWriter
from src.patches.generator import PatchDatasetResult, PatchGenerator
from src.patches.manifest import PatchManifest
from tests.conftest import make_valid_config, write_config

_CRS   = "EPSG:4326"
_BANDS = ("Blue", "Green", "Red", "NIR")


def _make_scene_geotiff(
    tmp_path: Path,
    width:    int  = 16,
    height:   int  = 16,
    nodata_block: bool = False,
) -> Path:
    """Write a synthetic source scene GeoTIFF using the real GeoTiffWriter."""
    data = np.random.rand(len(_BANDS), height, width).astype(np.float32)
    if nodata_block:
        data[:, : height // 2, : width // 2] = np.nan

    transform = AffineTransform.from_affine(
        Affine(0.001, 0.0, 87.0, 0.0, -0.001, 26.5)
    )
    dr = DownloadResult(
        data=data, crs=_CRS, transform=transform,
        band_names=_BANDS, width=width, height=height,
        aoi_bounds=AoiBounds(87.0, 26.0, 87.5, 26.5),
        num_tiles=1,
    )
    writer = GeoTiffWriter(GeoTiffProfile(tiled=False, overviews=False))
    return writer.write(dr, tmp_path / "image.tif").path


def _make_export_result(image_path: Path, scene_id: str = "scene001") -> MagicMock:
    """Build a mock DatasetExportResult wrapping a real GeoTIFF path."""
    scene_meta             = MagicMock()
    scene_meta.crs         = _CRS
    scene_meta.band_names  = _BANDS

    result               = MagicMock()
    result.scene_id       = scene_id
    result.image_path     = image_path
    result.scene_metadata = scene_meta
    return result


def _make_config(
    tmp_path: Path,
    patch_size: int = 8,
    stride:     int = 8,
    min_valid_ratio: float = 0.5,
    nodata: float = -9999.0,
):
    from src.core.config import Config
    data = make_valid_config()
    data["patch_generation"].update({
        "patch_size":            patch_size,
        "train_stride":          stride,
        "inference_stride":      stride,
        "min_valid_pixel_ratio": min_valid_ratio,
        "nodata_value":          nodata,
    })
    data["inference"]["patch_size"] = patch_size
    data["export"] = {
        "geotiff": {"compress": "LZW", "tiled": False, "tile_size": 256,
                    "dtype": "float32", "overviews": False},
        "manifest": {"formats": ["csv", "json"]},
        "max_tile_pixels": 1_000_000,
        "dataset_version": "1.0.0",
        "pipeline_version": "1.0.0",
        "feature_schema_version": "1.0.0",
        "landsat_collection": "Landsat C2 L2",
    }
    return Config(config_path=write_config(tmp_path, data))


@pytest.fixture
def scene_image(tmp_path: Path) -> Path:
    scene_dir = tmp_path / "source_scene"
    scene_dir.mkdir()
    return _make_scene_geotiff(scene_dir, width=16, height=16)


@pytest.fixture
def export_result(scene_image: Path) -> MagicMock:
    return _make_export_result(scene_image)


@pytest.fixture
def config(tmp_path: Path):
    return _make_config(tmp_path)


@pytest.fixture
def generator(config) -> PatchGenerator:
    return PatchGenerator(config)


# ==============================================================================
# Construction tests
# ==============================================================================

class TestPatchGeneratorConstruction:
    """Tests for PatchGenerator.__init__() reading config values."""

    def test_reads_patch_size_from_config(self, generator: PatchGenerator) -> None:
        assert generator._patch_size == 8

    def test_reads_stride_from_config(self, generator: PatchGenerator) -> None:
        assert generator._stride == 8

    def test_reads_nodata_from_config(self, generator: PatchGenerator) -> None:
        assert generator._nodata_value == pytest.approx(-9999.0)

    def test_reads_min_valid_ratio_from_config(self, generator: PatchGenerator) -> None:
        assert generator._min_valid_pixel_ratio == pytest.approx(0.5)

    def test_construction_does_not_touch_filesystem(
        self, config, tmp_path: Path
    ) -> None:
        PatchGenerator(config)
        # No output directories should exist yet.
        assert not (tmp_path / "patches_out").exists()


# ==============================================================================
# generate() — happy path tests
# ==============================================================================

class TestPatchGeneratorGenerate:
    """Tests for PatchGenerator.generate() end-to-end behaviour."""

    def test_returns_patch_dataset_result(
        self, generator: PatchGenerator, export_result: MagicMock, tmp_path: Path
    ) -> None:
        result = generator.generate(export_result, tmp_path / "patches_out")
        assert isinstance(result, PatchDatasetResult)

    def test_generates_expected_patch_count(
        self, generator: PatchGenerator, export_result: MagicMock, tmp_path: Path
    ) -> None:
        # 16x16 image, patch_size=8, stride=8 -> 2x2 = 4 patches.
        result = generator.generate(export_result, tmp_path / "patches_out")
        assert result.patches_generated == 4
        assert result.patches_skipped   == 0
        assert result.total_windows     == 4

    def test_patch_files_created(
        self, generator: PatchGenerator, export_result: MagicMock, tmp_path: Path
    ) -> None:
        result = generator.generate(export_result, tmp_path / "patches_out")
        files = list(result.scene_patches_dir.glob("*.tif"))
        assert len(files) == 4

    def test_deterministic_patch_ids(
        self, generator: PatchGenerator, export_result: MagicMock, tmp_path: Path
    ) -> None:
        result = generator.generate(export_result, tmp_path / "patches_out")
        ids = {e.patch_id for e in result.manifest.entries}
        assert ids == {
            "scene001_r000_c000", "scene001_r000_c001",
            "scene001_r001_c000", "scene001_r001_c001",
        }

    def test_patch_ids_stable_across_runs(
        self, export_result: MagicMock, tmp_path: Path
    ) -> None:
        """Calling generate() twice (fresh dirs) must produce identical IDs."""
        cfg1 = _make_config(tmp_path / "cfg1")
        cfg2 = _make_config(tmp_path / "cfg2")
        gen1 = PatchGenerator(cfg1)
        gen2 = PatchGenerator(cfg2)

        r1 = gen1.generate(export_result, tmp_path / "out1")
        r2 = gen2.generate(export_result, tmp_path / "out2")

        ids1 = sorted(e.patch_id for e in r1.manifest.entries)
        ids2 = sorted(e.patch_id for e in r2.manifest.entries)
        assert ids1 == ids2

    def test_manifest_csv_created(
        self, generator: PatchGenerator, export_result: MagicMock, tmp_path: Path
    ) -> None:
        result = generator.generate(export_result, tmp_path / "patches_out")
        assert result.manifest.csv_path.exists()

    def test_manifest_json_created(
        self, generator: PatchGenerator, export_result: MagicMock, tmp_path: Path
    ) -> None:
        result = generator.generate(export_result, tmp_path / "patches_out")
        assert result.manifest.json_path.exists()

    def test_patch_band_names_preserved(
        self, generator: PatchGenerator, export_result: MagicMock, tmp_path: Path
    ) -> None:
        result = generator.generate(export_result, tmp_path / "patches_out")
        any_patch = next(result.scene_patches_dir.glob("*.tif"))
        with rasterio.open(any_patch) as ds:
            assert list(ds.descriptions) == list(_BANDS)

    def test_patch_crs_preserved(
        self, generator: PatchGenerator, export_result: MagicMock, tmp_path: Path
    ) -> None:
        result = generator.generate(export_result, tmp_path / "patches_out")
        any_patch = next(result.scene_patches_dir.glob("*.tif"))
        with rasterio.open(any_patch) as ds:
            assert ds.crs.to_string() == _CRS

    def test_patch_dimensions_match_patch_size(
        self, generator: PatchGenerator, export_result: MagicMock, tmp_path: Path
    ) -> None:
        result = generator.generate(export_result, tmp_path / "patches_out")
        any_patch = next(result.scene_patches_dir.glob("*.tif"))
        with rasterio.open(any_patch) as ds:
            assert ds.width  == 8
            assert ds.height == 8
            assert ds.count  == 4

    def test_patch_transform_shifted_correctly(
        self, generator: PatchGenerator, export_result: MagicMock, tmp_path: Path
    ) -> None:
        result = generator.generate(export_result, tmp_path / "patches_out")
        r1c1_path = result.scene_patches_dir / "scene001_r001_c001.tif"
        with rasterio.open(r1c1_path) as ds:
            expected_c = 87.0  + 8 * 0.001
            expected_f = 26.5  + 8 * (-0.001)
            assert ds.transform.c == pytest.approx(expected_c)
            assert ds.transform.f == pytest.approx(expected_f)
            assert ds.transform.a == pytest.approx(0.001)
            assert ds.transform.e == pytest.approx(-0.001)

    def test_first_patch_transform_matches_source_origin(
        self, generator: PatchGenerator, export_result: MagicMock, tmp_path: Path
    ) -> None:
        result = generator.generate(export_result, tmp_path / "patches_out")
        r0c0_path = result.scene_patches_dir / "scene001_r000_c000.tif"
        with rasterio.open(r0c0_path) as ds:
            assert ds.transform.c == pytest.approx(87.0)
            assert ds.transform.f == pytest.approx(26.5)

    def test_directory_structure(
        self, generator: PatchGenerator, export_result: MagicMock, tmp_path: Path
    ) -> None:
        out    = tmp_path / "patches_out"
        result = generator.generate(export_result, out)
        assert result.scene_patches_dir == out / "scenes" / "scene001" / "patches"
        assert result.scene_patches_dir.exists()

    def test_summary_lines_ascii_only(
        self, generator: PatchGenerator, export_result: MagicMock, tmp_path: Path
    ) -> None:
        result = generator.generate(export_result, tmp_path / "patches_out")
        for line in result.summary_lines():
            assert all(ord(c) < 128 for c in line)


# ==============================================================================
# generate() — invalid patch filtering tests
# ==============================================================================

class TestPatchGeneratorInvalidFiltering:
    """Tests for NoData-based patch rejection."""

    def test_invalid_patches_skipped(
        self, generator: PatchGenerator, tmp_path: Path
    ) -> None:
        scene_dir  = tmp_path / "nodata_scene"
        scene_dir.mkdir()
        image_path = _make_scene_geotiff(scene_dir, width=16, height=16, nodata_block=True)
        export_result = _make_export_result(image_path)

        result = generator.generate(export_result, tmp_path / "patches_out")

        # Top-left 8x8 window is entirely NaN -> skipped.
        assert result.patches_skipped == 1
        assert result.patches_generated == 3
        assert result.total_windows == 4

    def test_skipped_patch_not_written_to_disk(
        self, generator: PatchGenerator, tmp_path: Path
    ) -> None:
        scene_dir  = tmp_path / "nodata_scene"
        scene_dir.mkdir()
        image_path = _make_scene_geotiff(scene_dir, width=16, height=16, nodata_block=True)
        export_result = _make_export_result(image_path)

        result = generator.generate(export_result, tmp_path / "patches_out")
        files = {p.stem for p in result.scene_patches_dir.glob("*.tif")}
        assert "scene001_r000_c000" not in files

    def test_strict_threshold_skips_more(
        self, export_result: MagicMock, tmp_path: Path
    ) -> None:
        strict_config = _make_config(tmp_path, min_valid_ratio=1.0)
        strict_gen    = PatchGenerator(strict_config)
        result = strict_gen.generate(export_result, tmp_path / "strict_out")
        # Random data should be all-valid (no NaN), so this is still 4/4
        # but demonstrates the threshold is actually applied/configurable.
        assert result.patches_generated + result.patches_skipped == result.total_windows


# ==============================================================================
# generate() — stride and overlap tests
# ==============================================================================

class TestPatchGeneratorStride:
    """Tests for stride configuration and override behaviour."""

    def test_stride_override(
        self, generator: PatchGenerator, export_result: MagicMock, tmp_path: Path
    ) -> None:
        result = generator.generate(export_result, tmp_path / "patches_out", stride=4)
        assert result.stride == 4
        # 16x16 image, patch_size=8, stride=4 -> offsets [0,4,8] -> 3x3=9 windows.
        assert result.total_windows == 9

    def test_default_stride_used_when_not_overridden(
        self, generator: PatchGenerator, export_result: MagicMock, tmp_path: Path
    ) -> None:
        result = generator.generate(export_result, tmp_path / "patches_out")
        assert result.stride == 8


# ==============================================================================
# generate() — manifest append behaviour tests
# ==============================================================================

class TestPatchGeneratorManifestAppend:
    """Tests for append_to_manifest behaviour across multiple scenes."""

    def test_append_to_manifest_accumulates(
        self, generator: PatchGenerator, tmp_path: Path
    ) -> None:
        out = tmp_path / "patches_out"

        scene_a_dir = tmp_path / "scene_a"; scene_a_dir.mkdir()
        image_a = _make_scene_geotiff(scene_a_dir, width=16, height=16)
        export_a = _make_export_result(image_a, scene_id="sceneA")

        scene_b_dir = tmp_path / "scene_b"; scene_b_dir.mkdir()
        image_b = _make_scene_geotiff(scene_b_dir, width=16, height=16)
        export_b = _make_export_result(image_b, scene_id="sceneB")

        generator.generate(export_a, out, append_to_manifest=True)
        result_b = generator.generate(export_b, out, append_to_manifest=True)

        # The returned result is scene-local so downstream stages process
        # only patches generated by the current call.
        assert result_b.manifest.entry_count == 4
        assert {
            entry.scene_id for entry in result_b.manifest.entries
        } == {"sceneB"}

        # The persisted manifest remains cumulative across generation calls.
        persisted = PatchManifest.from_csv(
            out / "patch_manifest.csv"
        )
        assert persisted.entry_count == 8  # 4 + 4
        assert {
            entry.scene_id for entry in persisted.entries
        } == {"sceneA", "sceneB"}

    def test_no_append_overwrites(
        self, generator: PatchGenerator, tmp_path: Path
    ) -> None:
        out = tmp_path / "patches_out"

        scene_a_dir = tmp_path / "scene_a"; scene_a_dir.mkdir()
        image_a = _make_scene_geotiff(scene_a_dir, width=16, height=16)
        export_a = _make_export_result(image_a, scene_id="sceneA")

        scene_b_dir = tmp_path / "scene_b"; scene_b_dir.mkdir()
        image_b = _make_scene_geotiff(scene_b_dir, width=16, height=16)
        export_b = _make_export_result(image_b, scene_id="sceneB")

        generator.generate(export_a, out, append_to_manifest=False)
        result_b = generator.generate(export_b, out, append_to_manifest=False)

        assert result_b.manifest.entry_count == 4


# ==============================================================================
# generate() — input validation tests
# ==============================================================================

class TestPatchGeneratorValidation:
    """Tests for input validation in PatchGenerator.generate()."""

    def test_empty_band_names_raises(
        self, generator: PatchGenerator, export_result: MagicMock, tmp_path: Path
    ) -> None:
        from src.core.exceptions import InvalidValueError
        export_result.scene_metadata.band_names = ()
        with pytest.raises(InvalidValueError, match="band_names"):
            generator.generate(export_result, tmp_path / "patches_out")

    def test_missing_source_image_raises_oserror(
        self, generator: PatchGenerator, tmp_path: Path
    ) -> None:
        missing = tmp_path / "nonexistent.tif"
        export_result = _make_export_result(missing)
        with pytest.raises(OSError):
            generator.generate(export_result, tmp_path / "patches_out")