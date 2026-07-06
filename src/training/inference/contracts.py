"""
Public data contracts for the Inference Pipeline Framework (Module 16).

Contract chain:
    TrainingResult            (Module 14) ──┐
    EvaluationResult          (Module 15) ──┤──> InferenceEngine.predict() ──> InferenceResult
    TransformPipelineResult   (Module 12) ──┤
    Config                                ──┘

InferenceResult is the immutable public output consumed by:
    Module 17 (River Morphology Analytics)
    Downstream geospatial analysis tools

Design invariants
-----------------
- All public contracts are frozen dataclasses.
- No torch types at module level (lazy import policy).
- Probability and confidence maps are stored as numpy arrays to remain
  torch-free and directly consumable by rasterio / numpy pipelines.
- All temporal and geospatial metadata fields from Module 11 SampleMetadata
  and Module 12 TransformSample are preserved without filtering.
- InferenceResult carries the complete prediction provenance so Module 17
  can reconstruct the full analysis chain.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "InferenceConfig",
    "SamplePrediction",
    "InferenceResult",
    "CheckpointMetadata",
]


# ==============================================================================
# InferenceConfig
# ==============================================================================

@dataclass(frozen=True)
class InferenceConfig:
    """
    Immutable inference configuration.

    Attributes:
        checkpoint_path:     Explicit path to a .pt checkpoint. When empty,
                             "best" or "latest" is derived from checkpoint_dir.
        checkpoint_strategy: "best", "latest", or "explicit".
        checkpoint_dir:      Directory containing checkpoints (from Module 14).
        device:              Compute device: "cpu", "cuda", "cuda:0", etc.
        batch_size:          Batch size for dataset inference.
        num_workers:         DataLoader worker count.
        mixed_precision:     Enable AMP float16 inference on CUDA.
        deterministic:       Enable deterministic CUDA algorithms.
        probability_mode:    "softmax" (multi-class) or "sigmoid" (binary/multi-label).
        confidence_strategy: "max_probability" or "entropy".
        output_dir:          Directory for exported predictions.
        export_numpy:        Save predicted masks as .npy files.
        export_geotiff:      Save predictions as GeoTIFF (requires rasterio).
        export_png:          Save predicted masks as PNG images.
        postprocess:         Enable post-processing pipeline.
        fill_holes:          Fill small holes in predicted masks.
        min_object_size:     Minimum connected-component size (pixels). 0=disabled.
        morph_open_size:     Morphological opening kernel size. 0=disabled.
        morph_close_size:    Morphological closing kernel size. 0=disabled.
        ignore_index:        Pixel index to exclude from confidence/stats.
        seed:                RNG seed for deterministic inference.
        pin_memory:          Pin DataLoader memory for GPU inference.
    """

    checkpoint_path:      str   = ""
    checkpoint_strategy:  str   = "best"
    checkpoint_dir:       str   = "checkpoints"
    device:               str   = "cpu"
    batch_size:           int   = 8
    num_workers:          int   = 4
    mixed_precision:      bool  = False
    deterministic:        bool  = False
    probability_mode:     str   = "softmax"
    confidence_strategy:  str   = "max_probability"
    output_dir:           str   = "predictions"
    export_numpy:         bool  = True
    export_geotiff:       bool  = False
    export_png:           bool  = False
    postprocess:          bool  = False
    fill_holes:           bool  = False
    min_object_size:      int   = 0
    morph_open_size:      int   = 0
    morph_close_size:     int   = 0
    ignore_index:         int   = 255
    seed:                 int   = 42
    pin_memory:           bool  = False

    @classmethod
    def from_config(cls, config: Any) -> InferenceConfig:
        """Build InferenceConfig from config.inference."""
        inf_cfg = getattr(config, "inference", None)
        if inf_cfg is None:
            return cls()
        return cls(
            checkpoint_path     = str(getattr(inf_cfg,  "checkpoint_path",     "")),
            checkpoint_strategy = str(getattr(inf_cfg,  "checkpoint_strategy", "best")),
            checkpoint_dir      = str(getattr(inf_cfg,  "checkpoint_dir",      "checkpoints")),
            device              = str(getattr(inf_cfg,  "device",              "cpu")),
            batch_size          = int(getattr(inf_cfg,  "batch_size",          8)),
            num_workers         = int(getattr(inf_cfg,  "num_workers",         4)),
            mixed_precision     = bool(getattr(inf_cfg, "mixed_precision",     False)),
            deterministic       = bool(getattr(inf_cfg, "deterministic",       False)),
            probability_mode    = str(getattr(inf_cfg,  "probability_mode",    "softmax")),
            confidence_strategy = str(getattr(inf_cfg,  "confidence_strategy", "max_probability")),
            output_dir          = str(getattr(inf_cfg,  "output_dir",          "predictions")),
            export_numpy        = bool(getattr(inf_cfg, "export_numpy",        True)),
            export_geotiff      = bool(getattr(inf_cfg, "export_geotiff",      False)),
            export_png          = bool(getattr(inf_cfg, "export_png",          False)),
            postprocess         = bool(getattr(inf_cfg, "postprocess",         False)),
            fill_holes          = bool(getattr(inf_cfg, "fill_holes",          False)),
            min_object_size     = int(getattr(inf_cfg,  "min_object_size",     0)),
            morph_open_size     = int(getattr(inf_cfg,  "morph_open_size",     0)),
            morph_close_size    = int(getattr(inf_cfg,  "morph_close_size",    0)),
            ignore_index        = int(getattr(inf_cfg,  "ignore_index",        255)),
            seed                = int(getattr(inf_cfg,  "seed",                42)),
            pin_memory          = bool(getattr(inf_cfg, "pin_memory",          False)),
        )


# ==============================================================================
# CheckpointMetadata
# ==============================================================================

@dataclass(frozen=True)
class CheckpointMetadata:
    """
    Immutable metadata extracted from a loaded checkpoint.

    Attributes:
        checkpoint_path: Absolute path to the loaded checkpoint file.
        checkpoint_version: Version string from the checkpoint payload.
        epoch:           Epoch at which this checkpoint was saved.
        train_loss:      Training loss at the saved epoch.
        val_loss:        Validation loss at the saved epoch.
        architecture:    Model architecture name from the checkpoint.
        num_classes:     Number of segmentation classes recorded.
        in_channels:     Number of input spectral channels recorded.
    """

    checkpoint_path:     str
    checkpoint_version:  str
    epoch:               int
    train_loss:          float
    val_loss:            float
    architecture:        str
    num_classes:         int
    in_channels:         int

    def as_dict(self) -> dict:
        return {
            "checkpoint_path":    self.checkpoint_path,
            "checkpoint_version": self.checkpoint_version,
            "epoch":              self.epoch,
            "train_loss":         round(self.train_loss, 6),
            "val_loss":           round(self.val_loss,   6),
            "architecture":       self.architecture,
            "num_classes":        self.num_classes,
            "in_channels":        self.in_channels,
        }


# ==============================================================================
# SamplePrediction
# ==============================================================================

@dataclass
class SamplePrediction:
    """
    Per-sample prediction output.  NOT frozen (arrays are mutable numpy arrays).

    Attributes:
        sample_id:         Unique patch identifier.
        predicted_mask:    (H, W) uint8 numpy array — predicted class IDs.
        probabilities:     (C, H, W) float32 — per-class probability maps.
        confidence:        (H, W) float32 — scalar confidence per pixel.
        logits:            (C, H, W) float32 — raw model logits (pre-softmax).
                           None when not retained to save memory.
        acquisition_date:  YYYY-MM-DD representative imagery date.
        season:            Resolved season name (e.g. "monsoon").
        hydrological_year: Resolved hydrological year integer.
        sensor:            Comma-separated sensor names.
        river_name:        River name, or empty string.
        reach_id:          River reach identifier.
        basin_id:          Drainage basin identifier.
        aoi_id:            Area-of-interest identifier.
        patch_path:        Absolute path to the source patch GeoTIFF.
        mask_path:         Absolute path to the source mask GeoTIFF.
        scene_id:          Source scene identifier.
        year:              Calendar year.
        month:             Calendar month [1, 12].
        metadata:          Free-form provenance dict.
        exported_paths:    Absolute paths of all exported files for this sample.
    """

    sample_id:          str
    predicted_mask:     np.ndarray          # (H, W) uint8
    probabilities:      np.ndarray          # (C, H, W) float32
    confidence:         np.ndarray          # (H, W) float32
    logits:             np.ndarray | None   = None
    acquisition_date:   str                 = ""
    season:             str                 = ""
    hydrological_year:  int                 = 0
    sensor:             str                 = ""
    river_name:         str                 = ""
    reach_id:           str                 = ""
    basin_id:           str                 = ""
    aoi_id:             str                 = ""
    patch_path:         str                 = ""
    mask_path:          str                 = ""
    scene_id:           str                 = ""
    year:               int                 = 0
    month:              int                 = 0
    metadata:           dict                = field(default_factory=dict)
    exported_paths:     list[str]           = field(default_factory=list)

    def summary_dict(self) -> dict:
        """Return a JSON-serializable summary (no large arrays)."""
        return {
            "sample_id":         self.sample_id,
            "acquisition_date":  self.acquisition_date,
            "season":            self.season,
            "hydrological_year": self.hydrological_year,
            "sensor":            self.sensor,
            "river_name":        self.river_name,
            "reach_id":          self.reach_id,
            "basin_id":          self.basin_id,
            "aoi_id":            self.aoi_id,
            "patch_path":        self.patch_path,
            "mask_shape":        list(self.predicted_mask.shape),
            "prob_shape":        list(self.probabilities.shape),
            "confidence_mean":   round(float(self.confidence.mean()), 6),
            "confidence_min":    round(float(self.confidence.min()),  6),
            "confidence_max":    round(float(self.confidence.max()),  6),
            "exported_paths":    self.exported_paths,
        }


# ==============================================================================
# InferenceResult
# ==============================================================================

@dataclass(frozen=True)
class InferenceResult:
    """
    Immutable public output of InferenceEngine.predict().

    Module 17 (River Morphology Analytics) consumes this object.

    Attributes:
        predictions:        Tuple of SamplePrediction, one per inferred sample.
        num_samples:        Total number of samples inferred.
        architecture:       Model architecture name.
        num_classes:        Number of segmentation classes.
        class_names:        Ordered class names.
        checkpoint_meta:    CheckpointMetadata of the loaded checkpoint.
        inference_config:   InferenceConfig used for this run.
        device_used:        Actual device used ("cpu" or "cuda:N").
        total_inference_s:  Total wall-clock seconds for all inference.
        per_sample_ms:      Mean inference time per sample in milliseconds.
        operations_log:     Ordered log of engine steps.
        mean_confidence:    Dataset-level mean confidence (across all pixels).
        class_pixel_counts: Dict class_name -> total predicted pixel count.
    """

    predictions:         tuple[SamplePrediction, ...]
    num_samples:         int
    architecture:        str
    num_classes:         int
    class_names:         tuple[str, ...]
    checkpoint_meta:     CheckpointMetadata
    inference_config:    InferenceConfig
    device_used:         str
    total_inference_s:   float
    per_sample_ms:       float
    operations_log:      tuple[str, ...]
    mean_confidence:     float
    class_pixel_counts:  dict[str, int]

    def summary_lines(self) -> list[str]:
        """Return ASCII-formatted summary lines."""
        return [
            f"  architecture:       {self.architecture}",
            f"  num_classes:        {self.num_classes}",
            f"  num_samples:        {self.num_samples}",
            f"  device_used:        {self.device_used}",
            f"  checkpoint_epoch:   {self.checkpoint_meta.epoch}",
            f"  mean_confidence:    {self.mean_confidence:.4f}",
            f"  total_inference_s:  {self.total_inference_s:.2f}",
            f"  per_sample_ms:      {self.per_sample_ms:.2f}",
        ]

    def as_dict(self) -> dict:
        """Return a JSON-serializable summary dict (no large arrays)."""
        return {
            "num_samples":        self.num_samples,
            "architecture":       self.architecture,
            "num_classes":        self.num_classes,
            "class_names":        list(self.class_names),
            "device_used":        self.device_used,
            "total_inference_s":  round(self.total_inference_s, 3),
            "per_sample_ms":      round(self.per_sample_ms,     3),
            "mean_confidence":    round(self.mean_confidence,   6),
            "class_pixel_counts": self.class_pixel_counts,
            "checkpoint":         self.checkpoint_meta.as_dict(),
            "operations_log":     list(self.operations_log),
            "sample_summaries":   [p.summary_dict() for p in self.predictions],
        }
