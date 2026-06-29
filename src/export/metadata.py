"""
Scene metadata generation and persistence for the export pipeline.

SceneMetadata is an immutable, fully JSON-serializable record describing one
exported scene. All nested fields use frozen dataclasses so the entire object
can be serialized with dataclasses.asdict() and reconstructed with
SceneMetadata(**data).

MetadataWriter constructs SceneMetadata from a FeatureStackResult and a
DownloadResult, traverses the result chain with getattr-based safe access,
and persists/loads the record as UTF-8 JSON.

Module 8 (Patch Generation) loads SceneMetadata to recover band names, CRS,
transform, and processing history without re-reading the GeoTIFF.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.export.downloader import AffineTransform, AoiBounds, DownloadResult

if TYPE_CHECKING:
    from src.core.config import Config
    from src.gee.feature_stack import FeatureStackResult

__all__ = [
    "SceneMetadata",
    "MetadataWriter",
    "METADATA_SCHEMA_VERSION",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)
METADATA_SCHEMA_VERSION: str = "1.0"


# ==============================================================================
# SceneMetadata
# ==============================================================================

@dataclass(frozen=True)
class SceneMetadata:
    """
    Immutable scene metadata record persisted alongside each GeoTIFF.

    All fields are Python primitives or frozen dataclasses so that
    dataclasses.asdict(metadata) always produces a JSON-serializable dict.

    Attributes:
        schema_version:    Metadata schema version string.
        scene_id:          Unique scene identifier.
        export_timestamp:  ISO 8601 UTC timestamp of export.
        aoi:               AoiBounds (min_lon, min_lat, max_lon, max_lat).
        crs:               Output CRS, e.g. "EPSG:4326".
        transform:         AffineTransform coefficients.
        width:             Raster width in pixels.
        height:            Raster height in pixels.
        resolution_meters: Nominal pixel size in metres.
        num_bands:         Total band count in the GeoTIFF.
        band_names:        Ordered tuple of all band names.
        composite_bands:   Source composite band names.
        spectral_indices:  Names of computed spectral indices.
        composite_method:  Compositing method ("median", "mean", etc.).
        sensors:           Landsat sensor names (e.g. ("L8", "L9")).
        start_date:        Collection date-range start (YYYY-MM-DD).
        end_date:          Collection date-range end (YYYY-MM-DD).
        cloud_cover_limit: Max cloud cover threshold applied.
        operations_applied: Preprocessing operations in order.
        num_tiles:         GEE download tiles assembled into the image.
    """

    schema_version:    str
    scene_id:          str
    export_timestamp:  str
    aoi:               AoiBounds
    crs:               str
    transform:         AffineTransform
    width:             int
    height:            int
    resolution_meters: float
    num_bands:         int
    band_names:        tuple[str, ...]
    composite_bands:   tuple[str, ...]
    spectral_indices:  tuple[str, ...]
    composite_method:  str
    sensors:           tuple[str, ...]
    start_date:        str
    end_date:          str
    cloud_cover_limit: float
    operations_applied: tuple[str, ...]
    num_tiles:         int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable plain dict (tuples become lists)."""
        return dataclasses.asdict(self)


# ==============================================================================
# MetadataWriter
# ==============================================================================

class MetadataWriter:
    """
    Generates SceneMetadata from pipeline results and persists it as JSON.

    Safely traverses FeatureStackResult -> CompositeResult ->
    ProcessedCollectionResult -> CollectionResult using getattr with
    defaults, so mocked objects in tests produce sensible values.

    Args:
        config: Fully initialized Config object.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._logger: logging.Logger = logging.getLogger(__name__)

    def generate(
        self,
        scene_id:             str,
        feature_stack_result: FeatureStackResult,
        download_result:      DownloadResult,
    ) -> SceneMetadata:
        """
        Build SceneMetadata from the feature stack and downloaded image data.

        Args:
            scene_id:             Unique identifier for this scene.
            feature_stack_result: From SpectralFeatureGenerator.generate().
            download_result:      From EarthEngineDownloader.download().

        Returns:
            Populated, immutable SceneMetadata.
        """
        composite  = getattr(feature_stack_result, "source_composite", None)
        processed  = getattr(composite, "source_result", None)
        collection = getattr(processed,  "source_result", None)

        sensors    = self._extract_sensors(collection)
        start, end, cloud = self._extract_dates(collection)
        method     = self._extract_method(composite)
        ops        = self._extract_ops(processed)

        band_names       = tuple(getattr(feature_stack_result, "all_band_names",       ()))
        composite_bands  = tuple(getattr(feature_stack_result, "composite_band_names", ()))
        spectral_indices = tuple(getattr(feature_stack_result, "features_computed",    ()))

        resolution = float(
            getattr(getattr(self._config, "satellite", None), "resolution_meters", 30.0)
        )

        meta = SceneMetadata(
            schema_version=METADATA_SCHEMA_VERSION,
            scene_id=scene_id,
            export_timestamp=datetime.now(timezone.utc).isoformat(),
            aoi=download_result.aoi_bounds,
            crs=download_result.crs,
            transform=download_result.transform,
            width=download_result.width,
            height=download_result.height,
            resolution_meters=resolution,
            num_bands=len(band_names),
            band_names=band_names,
            composite_bands=composite_bands,
            spectral_indices=spectral_indices,
            composite_method=method,
            sensors=sensors,
            start_date=start,
            end_date=end,
            cloud_cover_limit=cloud,
            operations_applied=ops,
            num_tiles=download_result.num_tiles,
        )

        self._logger.debug(
            "SceneMetadata generated: scene=%s, bands=%d",
            scene_id, len(band_names),
        )
        return meta

    def save(self, metadata: SceneMetadata, path: Path) -> Path:
        """
        Serialize SceneMetadata to a JSON file.

        Args:
            metadata: SceneMetadata to persist.
            path:     Destination path. Parent directory must exist.

        Returns:
            Resolved absolute path to the written file.
        """
        path = Path(path).resolve()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(metadata.to_dict(), fh, indent=2, ensure_ascii=True)
        self._logger.info("Scene metadata written: %s", path.name)
        return path

    @staticmethod
    def load(path: Path) -> SceneMetadata:
        """
        Load SceneMetadata from a JSON file.

        Args:
            path: Path to metadata JSON written by save().

        Returns:
            Reconstructed SceneMetadata.

        Raises:
            FileNotFoundError: path does not exist.
            ValueError:        JSON structure is invalid or missing fields.
        """
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)

            # Reconstruct nested frozen dataclasses from plain dicts.
            raw["aoi"]       = AoiBounds(**raw["aoi"])
            raw["transform"] = AffineTransform(**raw["transform"])

            # JSON lists -> tuples for frozen dataclass tuple fields.
            for key in (
                "band_names", "composite_bands", "spectral_indices",
                "sensors", "operations_applied",
            ):
                if key in raw and isinstance(raw[key], list):
                    raw[key] = tuple(raw[key])

            return SceneMetadata(**raw)
        except (TypeError, KeyError) as exc:
            raise ValueError(
                f"Metadata file has unexpected structure: {path}\nError: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Private extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_sensors(collection: Any) -> tuple[str, ...]:
        sensors = getattr(collection, "sensors", [])
        return tuple(
            s.name if hasattr(s, "name") else str(s)
            for s in sensors
        )

    @staticmethod
    def _extract_dates(collection: Any) -> tuple[str, str, float]:
        return (
            str(getattr(collection, "start_date",        "")),
            str(getattr(collection, "end_date",          "")),
            float(getattr(collection, "cloud_cover_limit", 20.0)),
        )

    @staticmethod
    def _extract_method(composite: Any) -> str:
        method = getattr(composite, "method", None)
        if method is None:
            return "unknown"
        return str(getattr(method, "value", method))

    @staticmethod
    def _extract_ops(processed: Any) -> tuple[str, ...]:
        return tuple(getattr(processed, "operations_applied", ()))