"""
Public data contracts for the Segmentation Model Framework (Module 13).

These dataclasses form the stable interface between Module 13 (models),
Module 14 (Training Engine), Module 15 (Evaluation), and Module 16 (Inference).
They must not be modified without strong justification.

Contract chain:
    TransformPipelineResult (Module 12)
        |
        v
    ModelConfig  ------>  SegmentationModel.forward(x) --> logits tensor
        |
        v
    ModelResult  (public output of ModelFactory.build())
        |
        v
    Training Engine (Module 14)

Design invariants
-----------------
- ModelConfig is frozen and fully config-driven. No field has a hardcoded
  semantic value; all defaults are documented so callers know what they get.
- ModelResult wraps the model object behind a stable, inspectable contract.
  Module 14 never instantiates models directly; it receives a ModelResult.
- Logits are NEVER post-processed inside the model. softmax / sigmoid /
  argmax are the responsibility of Module 14 (training) and Module 16
  (inference).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ModelConfig",
    "ModelResult",
    "EncoderConfig",
    "DecoderConfig",
]


# ==============================================================================
# EncoderConfig
# ==============================================================================

@dataclass(frozen=True)
class EncoderConfig:
    """
    Immutable encoder configuration.

    Attributes:
        filters:       Ordered filter counts per encoder stage.
                       Length determines the encoder depth.
                       Example: (32, 64, 128, 256, 512) -> 5-stage encoder.
        dropout_rate:  Spatial dropout rate applied after each encoder block
                       [0.0, 1.0]. 0.0 disables dropout.
        norm_type:     Normalisation layer type: "batch", "instance", or "none".
        act_type:      Activation function type: "relu", "leaky_relu", or "gelu".
        pool_type:     Downsampling method: "max" or "avg".
    """

    filters:      tuple[int, ...]   = (32, 64, 128, 256, 512)
    dropout_rate: float             = 0.0
    norm_type:    str               = "batch"
    act_type:     str               = "relu"
    pool_type:    str               = "max"

    @classmethod
    def from_config(cls, config: Any) -> EncoderConfig:
        """
        Build EncoderConfig from config.model.encoder.

        Returns defaults when the section is absent.
        """
        model_cfg   = getattr(config, "model", None)
        enc_cfg     = getattr(model_cfg, "encoder", None)
        if enc_cfg is None:
            return cls()
        raw_filters = getattr(enc_cfg, "filters", (32, 64, 128, 256, 512))
        return cls(
            filters      = tuple(int(f) for f in raw_filters),
            dropout_rate = float(getattr(enc_cfg, "dropout_rate", 0.0)),
            norm_type    = str(getattr(enc_cfg,   "norm_type",    "batch")),
            act_type     = str(getattr(enc_cfg,   "act_type",     "relu")),
            pool_type    = str(getattr(enc_cfg,   "pool_type",    "max")),
        )


# ==============================================================================
# DecoderConfig
# ==============================================================================

@dataclass(frozen=True)
class DecoderConfig:
    """
    Immutable decoder configuration.

    Attributes:
        dropout_rate:       Dropout applied inside decoder blocks [0.0, 1.0].
        norm_type:          Normalisation layer: "batch", "instance", or "none".
        act_type:           Activation: "relu", "leaky_relu", or "gelu".
        deep_supervision:   When True, each decoder scale emits an auxiliary
                            segmentation head and outputs are averaged.
        upsample_mode:      Upsampling strategy: "bilinear" or "nearest".
    """

    dropout_rate:     float = 0.0
    norm_type:        str   = "batch"
    act_type:         str   = "relu"
    deep_supervision: bool  = False
    upsample_mode:    str   = "bilinear"

    @classmethod
    def from_config(cls, config: Any) -> DecoderConfig:
        """Build DecoderConfig from config.model.decoder."""
        model_cfg = getattr(config, "model", None)
        dec_cfg   = getattr(model_cfg, "decoder", None)
        if dec_cfg is None:
            return cls()
        return cls(
            dropout_rate     = float(getattr(dec_cfg, "dropout_rate",     0.0)),
            norm_type        = str(getattr(dec_cfg,   "norm_type",        "batch")),
            act_type         = str(getattr(dec_cfg,   "act_type",         "relu")),
            deep_supervision = bool(getattr(dec_cfg,  "deep_supervision", False)),
            upsample_mode    = str(getattr(dec_cfg,   "upsample_mode",    "bilinear")),
        )


# ==============================================================================
# ModelConfig
# ==============================================================================

@dataclass(frozen=True)
class ModelConfig:
    """
    Immutable top-level model configuration.

    This is the single source of truth for model construction.
    ModelFactory.build() consumes exactly one ModelConfig and produces
    exactly one ModelResult.

    Attributes:
        architecture:   Registered model name: "unetplusplus", "unet", etc.
        in_channels:    Number of input spectral bands (never assume 3).
        num_classes:    Number of segmentation output classes.
        encoder:        EncoderConfig controlling the feature extraction path.
        decoder:        DecoderConfig controlling the reconstruction path.
        init_seed:      RNG seed for deterministic weight initialization.
                        None disables seed pinning.
        pretrained:     Reserved for future pretrained-weight loading.
                        Currently ignored; included for forward compatibility.
    """

    architecture: str           = "unetplusplus"
    in_channels:  int           = 12
    num_classes:  int           = 4
    encoder:      EncoderConfig = field(default_factory=EncoderConfig)
    decoder:      DecoderConfig = field(default_factory=DecoderConfig)
    init_seed:    int | None    = None
    pretrained:   bool          = False

    @classmethod
    def from_config(cls, config: Any) -> ModelConfig:
        """
        Build ModelConfig from config.model.

        All fields fall back to class-level defaults when absent from config.

        Args:
            config: Fully initialized Config object.

        Returns:
            Frozen ModelConfig.
        """
        model_cfg = getattr(config, "model", None)
        if model_cfg is None:
            return cls()
        raw_seed = getattr(model_cfg, "init_seed", None)
        return cls(
            architecture = str(getattr(model_cfg,  "architecture", "unetplusplus")),
            in_channels  = int(getattr(model_cfg,  "in_channels",  12)),
            num_classes  = int(getattr(model_cfg,  "num_classes",  4)),
            encoder      = EncoderConfig.from_config(config),
            decoder      = DecoderConfig.from_config(config),
            init_seed    = int(raw_seed) if raw_seed is not None else None,
            pretrained   = bool(getattr(model_cfg, "pretrained",   False)),
        )


# ==============================================================================
# ModelResult
# ==============================================================================

@dataclass(frozen=True)
class ModelResult:
    """
    Immutable public output of ModelFactory.build().

    Module 14 (Training Engine) receives a ModelResult and never instantiates
    models directly. This contract ensures that Module 14 can inspect model
    properties (parameter count, architecture name, configuration) without
    coupling to any concrete model class.

    Attributes:
        model:              The instantiated SegmentationModel. Type is Any to
                            avoid importing torch at the contracts module level.
                            At runtime it is always a torch.nn.Module subclass.
        architecture:       Registered architecture name (e.g. "unetplusplus").
        in_channels:        Number of input spectral channels.
        num_classes:        Number of output segmentation classes.
        num_parameters:     Total parameter count (trainable + frozen).
        num_trainable:      Trainable parameter count.
        deep_supervision:   True when the model emits auxiliary outputs.
        config:             The ModelConfig used to build this model.
        operations_log:     Ordered log of construction steps.
    """

    model:             Any
    architecture:      str
    in_channels:       int
    num_classes:       int
    num_parameters:    int
    num_trainable:     int
    deep_supervision:  bool
    config:            ModelConfig
    operations_log:    tuple[str, ...]

    def summary_lines(self) -> list[str]:
        """Return ASCII-formatted summary lines (no Unicode box-drawing)."""
        return [
            f"  architecture:      {self.architecture}",
            f"  in_channels:       {self.in_channels}",
            f"  num_classes:       {self.num_classes}",
            f"  num_parameters:    {self.num_parameters:,}",
            f"  num_trainable:     {self.num_trainable:,}",
            f"  deep_supervision:  {self.deep_supervision}",
            f"  encoder_filters:   {self.config.encoder.filters}",
            f"  decoder_dropout:   {self.config.decoder.dropout_rate}",
            f"  norm_type:         {self.config.encoder.norm_type}",
            f"  init_seed:         {self.config.init_seed}",
        ]
