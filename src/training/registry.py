"""
Transform registry for the Data Transformation and Augmentation Pipeline
(Module 12).

TransformRegistry mirrors the registry pattern established by RuleRegistry
(Module 9) and LabelStrategyRegistry.  Every augmentation class is registered
by name; TransformRegistry.create_pipeline() reads the config and assembles a
ComposedTransform from the enabled entries.

Adding a new transform requires only:
    1. Subclass SegmentationTransform.
    2. Decorate with @TransformRegistry.register or call
       TransformRegistry.register_external(MyTransform).
    3. Add the transform to config.training.augmentation.

No existing code needs modification.

All eight built-in transforms are registered at module import time by being
referenced in the _BUILTIN_TRANSFORMS list at the bottom of this file.  This
design avoids circular imports: augmentation.py imports transform.py (for
SegmentationTransform), registry.py imports both.
"""

from __future__ import annotations

import logging
from typing import Any

from src.training.augmentation import (
    BrightnessTransform,
    ContrastTransform,
    GaussianNoiseTransform,
    HorizontalFlipTransform,
    RandomCropTransform,
    RandomScaleTransform,
    Rotate90Transform,
    VerticalFlipTransform,
)
from src.training.normalization import NormalizationTransform
from src.training.transform import (
    ComposedTransform,
    IdentityTransform,
    SegmentationTransform,
)

__all__ = ["TransformRegistry"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class TransformRegistry:
    """
    Central registry for SegmentationTransform plugins.

    TransformPipeline queries this registry to build train / validation / test
    transform pipelines from config without knowing which transforms exist.

    Usage -- built-in transforms register themselves at import time:

        @TransformRegistry.register
        class HorizontalFlipTransform(SegmentationTransform): ...

    Usage -- external plugin registration at runtime:

        TransformRegistry.register_external(MyCustomTransform)

    Config structure expected under config.training.augmentation:

        training:
          augmentation:
            horizontal_flip:
              enabled: true
              probability: 0.5
            vertical_flip:
              enabled: true
              probability: 0.5
            rotate_90:
              enabled: true
              probability: 0.5
            brightness:
              enabled: false
              probability: 0.3
              max_delta: 0.05
            contrast:
              enabled: false
              probability: 0.3
              contrast_range: 0.1
            gaussian_noise:
              enabled: false
              probability: 0.3
              std: 0.02
            random_crop:
              enabled: false
              probability: 0.5
              crop_height: 224
              crop_width: 224
            random_scale:
              enabled: false
              probability: 0.3
              min_scale: 0.75
              max_scale: 1.25
    """

    _registered: dict[str, type[SegmentationTransform]] = {}

    @classmethod
    def register(
        cls,
        transform_class: type[SegmentationTransform],
    ) -> type[SegmentationTransform]:
        """
        Class decorator.  Registers transform_class by its _NAME class attribute.
        """
        name = transform_class._NAME  # type: ignore[attr-defined]
        cls._registered[name] = transform_class
        _LOGGER.debug("TransformRegistry: registered '%s'", name)
        return transform_class

    @classmethod
    def register_external(cls, transform_class: type[SegmentationTransform]) -> None:
        """
        Imperatively register an external transform class.

        Allows runtime plugin injection without the decorator syntax.
        """
        name = transform_class._NAME  # type: ignore[attr-defined]
        cls._registered[name] = transform_class
        _LOGGER.debug("TransformRegistry: registered external '%s'", name)

    @classmethod
    def registered_names(cls) -> tuple[str, ...]:
        """Names of all registered transforms."""
        return tuple(cls._registered.keys())

    @classmethod
    def clear(cls) -> None:
        """
        Unregister all transforms.

        Intended for test isolation ONLY.  Do NOT call in production code.
        """
        cls._registered.clear()

    @classmethod
    def create_pipeline(
        cls,
        config:             Any,
        normalization_stats: Any = None,  # NormalizationStatistics or None
        augmentation_only:   bool = False,
    ) -> ComposedTransform:
        """
        Build a ComposedTransform from config.training.augmentation.

        The assembled pipeline has this order:
            1. NormalizationTransform (always first, if stats provided)
            2. Enabled augmentation transforms in registration order

        Transforms are added only when both:
            - transform_cfg.enabled == True
            - The transform name is registered in the registry

        Args:
            config:               Fully initialized Config object.
            normalization_stats:  NormalizationStatistics to prepend as the
                                  first transform.  When None, no normalization
                                  is inserted (used for val/test when caller
                                  passes stats separately).
            augmentation_only:    When True, skip normalization even if stats
                                  is provided (used for validation/test
                                  pipelines that share the same normalization
                                  stats but skip augmentation).

        Returns:
            ComposedTransform containing all enabled transforms.
        """
        transforms: list[SegmentationTransform] = []

        if normalization_stats is not None and not augmentation_only:
            transforms.append(NormalizationTransform(normalization_stats))

        aug_cfg = getattr(
            getattr(config, "training", None), "augmentation", None
        )

        for name, transform_cls in cls._registered.items():
            entry_cfg = getattr(aug_cfg, name, None) if aug_cfg is not None else None
            if entry_cfg is None:
                continue
            enabled = bool(getattr(entry_cfg, "enabled", False))
            if not enabled:
                continue

            try:
                transform = cls._build_transform(name, transform_cls, entry_cfg)
                transforms.append(transform)
                _LOGGER.debug(
                    "TransformRegistry.create_pipeline: added '%s'", name
                )
            except Exception as exc:
                _LOGGER.error(
                    "TransformRegistry.create_pipeline: failed to build "
                    "'%s': %s", name, exc,
                )
                raise

        if not transforms:
            transforms.append(IdentityTransform())

        return ComposedTransform(transforms)

    @classmethod
    def create_inference_pipeline(
        cls,
        normalization_stats: Any,
    ) -> ComposedTransform:
        """
        Build a normalization-only pipeline for inference (no augmentation).

        Args:
            normalization_stats: NormalizationStatistics to apply.

        Returns:
            ComposedTransform with only NormalizationTransform.
        """
        return ComposedTransform([NormalizationTransform(normalization_stats)])

    # ------------------------------------------------------------------
    # Private factory
    # ------------------------------------------------------------------

    @staticmethod
    def _build_transform(
        name:           str,
        transform_cls:  type[SegmentationTransform],
        entry_cfg:      Any,
    ) -> SegmentationTransform:
        """
        Instantiate a transform from its config entry.

        Each transform reads only the parameters it needs from entry_cfg.
        All parameters must originate from config (never hardcoded here).
        """
        if name == "horizontal_flip":
            return HorizontalFlipTransform(
                probability = float(getattr(entry_cfg, "probability", 0.5)),
            )

        if name == "vertical_flip":
            return VerticalFlipTransform(
                probability = float(getattr(entry_cfg, "probability", 0.5)),
            )

        if name == "rotate_90":
            num_rot = getattr(entry_cfg, "num_rotations", None)
            return Rotate90Transform(
                probability   = float(getattr(entry_cfg, "probability", 0.5)),
                num_rotations = int(num_rot) if num_rot is not None else None,
            )

        if name == "brightness":
            return BrightnessTransform(
                probability = float(getattr(entry_cfg, "probability", 0.3)),
                max_delta   = float(getattr(entry_cfg, "max_delta",    0.05)),
            )

        if name == "contrast":
            return ContrastTransform(
                probability    = float(getattr(entry_cfg, "probability",    0.3)),
                contrast_range = float(getattr(entry_cfg, "contrast_range", 0.1)),
            )

        if name == "gaussian_noise":
            return GaussianNoiseTransform(
                probability = float(getattr(entry_cfg, "probability", 0.3)),
                std         = float(getattr(entry_cfg, "std",         0.02)),
            )

        if name == "random_crop":
            return RandomCropTransform(
                probability  = float(getattr(entry_cfg, "probability",  0.5)),
                crop_height  = int(getattr(entry_cfg,   "crop_height",  224)),
                crop_width   = int(getattr(entry_cfg,   "crop_width",   224)),
            )

        if name == "random_scale":
            return RandomScaleTransform(
                probability = float(getattr(entry_cfg, "probability", 0.3)),
                min_scale   = float(getattr(entry_cfg, "min_scale",   0.75)),
                max_scale   = float(getattr(entry_cfg, "max_scale",   1.25)),
            )

        # External / unknown transform: attempt generic construction.
        _LOGGER.warning(
            "TransformRegistry._build_transform: no known constructor for "
            "'%s'; attempting transform_cls(config=entry_cfg).", name,
        )
        return transform_cls(entry_cfg)  # type: ignore[call-arg]


# ==============================================================================
# Register all built-in transforms
# ==============================================================================

TransformRegistry.register(HorizontalFlipTransform)
TransformRegistry.register(VerticalFlipTransform)
TransformRegistry.register(Rotate90Transform)
TransformRegistry.register(BrightnessTransform)
TransformRegistry.register(ContrastTransform)
TransformRegistry.register(GaussianNoiseTransform)
TransformRegistry.register(RandomCropTransform)
TransformRegistry.register(RandomScaleTransform)
