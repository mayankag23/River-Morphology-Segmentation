"""
Spectral classification rules for the pseudo-label generation pipeline
(Module 9).

Design principles
-----------------
1. Evidence-based scoring
   Each rule uses a sigmoid function to convert spectral index values into
   per-pixel evidence in (0, 1). The sigmoid produces continuous, smooth
   evidence near threshold boundaries rather than a hard cutoff. The
   protected helper _soft_threshold() is the stable public interface;
   internally it delegates to _sigmoid_evidence().

2. Plugin-based RuleRegistry
   RuleEngine no longer hardcodes the four rule classes. Every concrete rule
   class decorated with @RuleRegistry.register is discovered automatically by
   RuleEngine.from_config(). Future rules (ShadowRule, MudRule,
   FloodedVegetationRule, ...) can be added by:
       (a) Subclassing ClassificationRule and decorating with
           @RuleRegistry.register, or
       (b) Calling RuleRegistry.register_external(MyRule) at runtime.

3. Seasonal interface preparation
   ClassificationRule.apply() accepts an optional ClassificationContext.
   All current rules ignore it. Future rules can inspect
   context.season, context.hydrological_year, context.sensor, etc.
   without requiring architecture changes.

4. BackgroundRule is always registered last
   Registration order is guaranteed by Python 3.7+ insertion-ordered dicts.
   The four built-in rules are decorated in this order: Water, Sand,
   Vegetation, Background. RuleEngine.from_config() returns them in
   registration order, so engine.rule_names[-1] == "background" is always
   satisfied.

5. Improved SandRule
   Optionally incorporates NDBI, NDMI (inverted), SAVI (inverted), and
   optical brightness. All additional bands default to zero weight so the
   original four-index behaviour is preserved when not configured.

All thresholds, weights, and minimum confidence values come exclusively
from Config. No spectral threshold is hardcoded.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.labels.contracts import ClassificationContext, RuleResult, SpectralBandData

__all__ = [
    "ClassificationRule",
    "RuleRegistry",
    "WaterRule",
    "SandRule",
    "VegetationRule",
    "BackgroundRule",
    "RuleEngine",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)
_EPS: float = 1e-8


# ==============================================================================
# RuleRegistry
# ==============================================================================

class RuleRegistry:
    """
    Central registry for ClassificationRule plugins.

    RuleEngine.from_config() queries this registry rather than hardcoding
    which rules exist. Any rule class decorated with @RuleRegistry.register
    is automatically discovered by RuleEngine.from_config().

    Registration order determines the order rules appear in apply_all()
    output. BackgroundRule must always be registered last.

    Usage -- built-in rules register themselves at module import time:

        @RuleRegistry.register
        class WaterRule(ClassificationRule): ...

    Usage -- external plugin registration at runtime:

        from src.labels.rules import RuleRegistry
        RuleRegistry.register_external(MyCustomRule)
    """

    _registered: dict[str, type[ClassificationRule]] = {}

    @classmethod
    def register(
        cls, rule_class: type[ClassificationRule]
    ) -> type[ClassificationRule]:
        """
        Class decorator. Registers rule_class by its _CLASS_NAME.

        Args:
            rule_class: Concrete ClassificationRule subclass to register.

        Returns:
            rule_class unchanged (allows use as a decorator).
        """
        name = rule_class._CLASS_NAME  # type: ignore[attr-defined]
        cls._registered[name] = rule_class
        _LOGGER.debug("RuleRegistry: registered '%s'", name)
        return rule_class

    @classmethod
    def register_external(cls, rule_class: type[ClassificationRule]) -> None:
        """
        Imperatively register an external rule class.

        Allows runtime plugin injection without the decorator syntax, e.g.
        when the rule class is defined in a third-party package.
        """
        name = rule_class._CLASS_NAME  # type: ignore[attr-defined]
        cls._registered[name] = rule_class
        _LOGGER.debug("RuleRegistry: registered external '%s'", name)

    @classmethod
    def create_enabled(cls, config: Any) -> list[ClassificationRule]:
        """
        Instantiate all registered rules and return only the enabled ones.

        Rules are returned in registration order. BackgroundRule (the
        fallback) must always be registered last so that engine.rule_names[-1]
        == "background" is always satisfied.

        Args:
            config: Fully initialized Config object.

        Returns:
            List of enabled ClassificationRule instances in registration order.
        """
        instances = [rule_cls(config) for rule_cls in cls._registered.values()]
        enabled = [r for r in instances if r.is_enabled]
        _LOGGER.debug(
            "RuleRegistry.create_enabled: %d/%d rules enabled: %s",
            len(enabled), len(instances), [r.class_name for r in enabled],
        )
        return enabled

    @classmethod
    def registered_names(cls) -> tuple[str, ...]:
        """Return names of all registered rule classes (enabled or not)."""
        return tuple(cls._registered.keys())

    @classmethod
    def clear(cls) -> None:
        """
        Unregister all rules.

        Intended for test isolation ONLY. Do NOT call in production code.
        """
        cls._registered.clear()


# ==============================================================================
# Abstract base
# ==============================================================================

class ClassificationRule(ABC):
    """
    Abstract interface for all spectral classification rules.

    Each subclass votes for one class across all pixels in a patch, returning
    a per-pixel evidence-based confidence map.

    Subclasses should:
        - Define _CLASS_ID: int (class-level)
        - Define _CLASS_NAME: str (class-level)
        - Be decorated with @RuleRegistry.register to be auto-discovered.
    """

    @property
    @abstractmethod
    def class_id(self) -> int:
        """Integer class label this rule votes for."""

    @property
    @abstractmethod
    def class_name(self) -> str:
        """Human-readable class name."""

    @property
    @abstractmethod
    def is_enabled(self) -> bool:
        """True if this rule is active (reads from config)."""

    @abstractmethod
    def apply(
        self,
        band_data: SpectralBandData,
        context:   ClassificationContext | None = None,
    ) -> RuleResult:
        """
        Apply this rule to one patch.

        Args:
            band_data: SpectralBandData with all available band arrays.
            context:   Optional ClassificationContext for future seasonal or
                       sensor-specific adaptation. All current rules ignore
                       this parameter.

        Returns:
            RuleResult with per-pixel sigmoid evidence map.
        """

    # ------------------------------------------------------------------
    # Protected interface -- stable name, sigmoid implementation
    # ------------------------------------------------------------------

    @staticmethod
    def _soft_threshold(
        values:    np.ndarray,
        threshold: float,
        scale:     float,
        direction: str,
    ) -> np.ndarray:
        """
        Convert band values to per-pixel evidence in (0, 1) via sigmoid.

        This is the stable protected interface. Tests call this method
        directly. Internally it delegates to _sigmoid_evidence().

        Args:
            values:    (H, W) float32 band array. NaN inputs -> 0.0 evidence.
            threshold: Decision boundary value.
            scale:     Width of the soft transition zone (e.g. 0.30).
            direction: "greater" -- evidence -> 1 for values >> threshold.
                       "less"    -- evidence -> 1 for values << threshold.

        Returns:
            (H, W) float32 evidence array in (0.0, 1.0).
        """
        return ClassificationRule._sigmoid_evidence(values, threshold, scale, direction)

    @staticmethod
    def _sigmoid_evidence(
        values:    np.ndarray,
        threshold: float,
        scale:     float,
        direction: str,
    ) -> np.ndarray:
        """
        Sigmoid implementation of the evidence function.

        Produces continuous, differentiable evidence that correctly models
        spectral measurement uncertainty near the threshold boundary.

        Behavior:
            direction="greater": evidence -> 1 for values >> threshold,
                                 evidence = 0.5 exactly at threshold,
                                 evidence -> 0 for values << threshold.
            direction="less":    inverse of the above.

        Args:
            values:    (H, W) float32 band array. NaN -> 0.0 evidence.
            threshold: Decision boundary value.
            scale:     Controls steepness of the sigmoid transition.
            direction: "greater" or "less".

        Returns:
            (H, W) float32 evidence array in (0.0, 1.0).
        """
        k = 1.25 / max(float(scale), _EPS)
        if direction == "greater":
            exponent = -k * (values.astype(np.float64) - threshold)
        else:  # "less"
            exponent = -k * (threshold - values.astype(np.float64))

        # Clip exponent to avoid float64 overflow (exp > ~709 overflows).
        exponent_clipped = np.clip(exponent, -88.0, 88.0)
        raw = (1.0 / (1.0 + np.exp(exponent_clipped))).astype(np.float32)
        raw[np.isnan(values)] = 0.0
        return raw

    def _weighted_evidence(
        self,
        band_data:   SpectralBandData,
        conditions:  list[tuple[str, float, str, float, float]],
    ) -> tuple[np.ndarray, list[str], list[str]]:
        """
        Compute weighted-average evidence across multiple band conditions.

        Conditions with weight <= 0 or absent bands are silently skipped.

        Args:
            band_data:  SpectralBandData.
            conditions: List of (band_name, threshold, direction, weight, scale).

        Returns:
            Tuple of:
                evidence_map: (H, W) float32 weighted evidence in (0, 1).
                bands_used:   Band names that contributed.
                bands_missing: Band names that were absent from band_data.
        """
        h, w    = band_data.height, band_data.width
        accum   = np.zeros((h, w), dtype=np.float32)
        total_w = 0.0
        used:    list[str] = []
        missing: list[str] = []

        for band_name, threshold, direction, weight, scale in conditions:
            if weight <= 0.0:
                continue
            if band_name not in band_data.bands:
                missing.append(band_name)
                continue
            arr      = band_data.bands[band_name]
            evidence = self._soft_threshold(arr, threshold, scale, direction)
            accum   += evidence * weight
            total_w += weight
            used.append(band_name)

        if total_w < _EPS:
            return np.zeros((h, w), dtype=np.float32), used, missing

        return (accum / total_w).astype(np.float32), used, missing


# ==============================================================================
# WaterRule
# ==============================================================================

@RuleRegistry.register
class WaterRule(ClassificationRule):
    """
    Classifies pixels as Water (class_id=1).

    Evidence combines:
        MNDWI (primary)     -- modified NDWI; best for turbid river water.
        NDWI                -- open water indicator.
        AWEI_nsh            -- automated water extraction, no shadow.
        AWEI_sh             -- automated water extraction, with shadow.
        NDVI (inverted)     -- high vegetation -> low water evidence.

    All thresholds and weights are read from config.labels.rules.water.
    context parameter accepted but currently ignored.
    """

    _CLASS_ID:   int = 1
    _CLASS_NAME: str = "water"

    def __init__(self, config: Any) -> None:
        rules_cfg = getattr(getattr(config, "labels", None), "rules", None)
        w = getattr(rules_cfg, "water", None)

        self._enabled          = bool(getattr(w, "enabled",            True))
        self._mndwi_thr        = float(getattr(w, "mndwi_threshold",    0.2))
        self._ndwi_thr         = float(getattr(w, "ndwi_threshold",     0.0))
        self._awei_nsh_thr     = float(getattr(w, "awei_nsh_threshold", 0.0))
        self._awei_sh_thr      = float(getattr(w, "awei_sh_threshold",  0.0))
        self._ndvi_max_thr     = float(getattr(w, "ndvi_max_threshold", 0.05))
        self._mndwi_weight     = float(getattr(w, "mndwi_weight",       0.40))
        self._ndwi_weight      = float(getattr(w, "ndwi_weight",        0.20))
        self._awei_nsh_weight  = float(getattr(w, "awei_nsh_weight",    0.15))
        self._awei_sh_weight   = float(getattr(w, "awei_sh_weight",     0.05))
        self._ndvi_weight      = float(getattr(w, "ndvi_weight",        0.20))
        self._scale            = float(getattr(w, "confidence_scale",   0.30))
        self._min_confidence   = float(getattr(w, "min_confidence",     0.40))

    @property
    def class_id(self) -> int:
        return self._CLASS_ID

    @property
    def class_name(self) -> str:
        return self._CLASS_NAME

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def apply(
        self,
        band_data: SpectralBandData,
        context:   ClassificationContext | None = None,
    ) -> RuleResult:
        conditions = [
            ("MNDWI",    self._mndwi_thr,    "greater", self._mndwi_weight,    self._scale),
            ("NDWI",     self._ndwi_thr,     "greater", self._ndwi_weight,     self._scale),
            ("AWEI_nsh", self._awei_nsh_thr, "greater", self._awei_nsh_weight, self._scale),
            ("AWEI_sh",  self._awei_sh_thr,  "greater", self._awei_sh_weight,  self._scale),
            ("NDVI",     self._ndvi_max_thr, "less",    self._ndvi_weight,     self._scale),
        ]
        evidence, used, missing = self._weighted_evidence(band_data, conditions)
        pixel_mask = evidence >= self._min_confidence
        evidence   = np.where(pixel_mask, evidence, 0.0).astype(np.float32)
        return RuleResult(
            class_id=self._CLASS_ID, class_name=self._CLASS_NAME,
            confidence=evidence, pixel_mask=pixel_mask,
            bands_used=tuple(used), bands_missing=tuple(missing),
        )


# ==============================================================================
# SandRule
# ==============================================================================

@RuleRegistry.register
class SandRule(ClassificationRule):
    """
    Classifies pixels as Sand / bare sediment (class_id=2).

    Core evidence:
        MNDWI (inverted) -- sand is NOT water (high SWIR1).
        BSI              -- bare soil index.
        NDVI (inverted)  -- sand has little vegetation.
        NDWI (inverted)  -- sand has no water signal.

    Extended evidence (all default to weight=0, preserving v1 behaviour):
        NDBI             -- built-up / bare surface.
        NDMI (inverted)  -- dry sand has low moisture.
        SAVI (inverted)  -- soil-adjusted; low for bare sand.
        brightness       -- mean optical brightness (Blue+Green+Red+NIR).

    All thresholds and weights come from config.labels.rules.sand.
    context parameter accepted but currently ignored.
    """

    _CLASS_ID:   int = 2
    _CLASS_NAME: str = "sand"

    def __init__(self, config: Any) -> None:
        rules_cfg = getattr(getattr(config, "labels", None), "rules", None)
        s = getattr(rules_cfg, "sand", None)

        self._enabled          = bool(getattr(s, "enabled",              True))
        self._mndwi_max_thr    = float(getattr(s, "mndwi_max_threshold",  0.0))
        self._bsi_thr          = float(getattr(s, "bsi_threshold",         0.0))
        self._ndvi_max_thr     = float(getattr(s, "ndvi_max_threshold",   0.25))
        self._ndwi_max_thr     = float(getattr(s, "ndwi_max_threshold",    0.0))
        self._mndwi_weight     = float(getattr(s, "mndwi_weight",         0.30))
        self._bsi_weight       = float(getattr(s, "bsi_weight",            0.35))
        self._ndvi_weight      = float(getattr(s, "ndvi_weight",           0.25))
        self._ndwi_weight      = float(getattr(s, "ndwi_weight",           0.10))
        self._scale            = float(getattr(s, "confidence_scale",      0.30))
        self._min_confidence   = float(getattr(s, "min_confidence",        0.30))

        # Extended indices -- zero defaults preserve original behaviour.
        self._ndbi_thr          = float(getattr(s, "ndbi_threshold",       0.0))
        self._ndbi_weight       = float(getattr(s, "ndbi_weight",           0.0))
        self._ndmi_max_thr      = float(getattr(s, "ndmi_max_threshold",    0.0))
        self._ndmi_weight       = float(getattr(s, "ndmi_weight",           0.0))
        self._savi_max_thr      = float(getattr(s, "savi_max_threshold",   0.15))
        self._savi_weight       = float(getattr(s, "savi_weight",           0.0))
        self._brightness_thr    = float(getattr(s, "brightness_threshold",  0.2))
        self._brightness_weight = float(getattr(s, "brightness_weight",     0.0))

    @property
    def class_id(self) -> int:
        return self._CLASS_ID

    @property
    def class_name(self) -> str:
        return self._CLASS_NAME

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def apply(
        self,
        band_data: SpectralBandData,
        context:   ClassificationContext | None = None,
    ) -> RuleResult:
        conditions: list[tuple[str, float, str, float, float]] = [
            ("MNDWI", self._mndwi_max_thr, "less",    self._mndwi_weight, self._scale),
            ("BSI",   self._bsi_thr,        "greater", self._bsi_weight,   self._scale),
            ("NDVI",  self._ndvi_max_thr,   "less",    self._ndvi_weight,  self._scale),
            ("NDWI",  self._ndwi_max_thr,   "less",    self._ndwi_weight,  self._scale),
            # Extended -- zero weight skipped automatically in _weighted_evidence.
            ("NDBI",  self._ndbi_thr,        "greater", self._ndbi_weight,  self._scale),
            ("NDMI",  self._ndmi_max_thr,    "less",    self._ndmi_weight,  self._scale),
            ("SAVI",  self._savi_max_thr,    "less",    self._savi_weight,  self._scale),
        ]

        # Optional brightness injection -- computed inline, not stored in GeoTIFF.
        if self._brightness_weight > 0.0:
            band_data = self._inject_brightness(band_data)
            conditions.append(
                ("_brightness", self._brightness_thr, "greater",
                 self._brightness_weight, self._scale)
            )

        evidence, used, missing = self._weighted_evidence(band_data, conditions)
     
        mndwi = band_data.bands.get("MNDWI")
        if mndwi is not None:
            water_mask = mndwi > self._mndwi_max_thr
        else:
            water_mask = np.zeros_like(evidence, dtype=bool)

        pixel_mask = ((evidence >= self._min_confidence) & (~water_mask))
            
        evidence   = np.where(pixel_mask, evidence, 0.0).astype(np.float32)

        # Remove internal brightness band from public report.
        used_clean = [b for b in used if not b.startswith("_")]
        return RuleResult(
            class_id=self._CLASS_ID, class_name=self._CLASS_NAME,
            confidence=evidence, pixel_mask=pixel_mask,
            bands_used=tuple(used_clean), bands_missing=tuple(missing),
        )

    @staticmethod
    def _inject_brightness(band_data: SpectralBandData) -> SpectralBandData:
        """
        Compute optical brightness = mean(Blue, Green, Red, NIR) and inject as
        a temporary "_brightness" band. Returns a shallow copy of band_data
        with the extra band; the source SpectralBandData is NOT mutated.
        """
        optical = [b for b in ("Blue", "Green", "Red", "NIR") if b in band_data.bands]
        if not optical:
            return band_data
        brightness = np.stack(
            [band_data.bands[b] for b in optical], axis=0
        ).mean(axis=0).astype(np.float32)
        new_bands = dict(band_data.bands)
        new_bands["_brightness"] = brightness
        return SpectralBandData(
            bands=new_bands, height=band_data.height, width=band_data.width,
            crs=band_data.crs, transform=band_data.transform,
            band_names=band_data.band_names + ("_brightness",),
        )


# ==============================================================================
# VegetationRule
# ==============================================================================

@RuleRegistry.register
class VegetationRule(ClassificationRule):
    """
    Classifies pixels as Vegetation (class_id=3).

    Evidence combines:
        NDVI  -- primary greenness indicator.
        SAVI  -- soil-adjusted; accounts for sparse cover on sandy floodplains.
        NDMI  -- leaf moisture; distinguishes living from dead vegetation.

    All thresholds and weights come from config.labels.rules.vegetation.
    context parameter accepted but currently ignored.
    """

    _CLASS_ID:   int = 3
    _CLASS_NAME: str = "vegetation"

    def __init__(self, config: Any) -> None:
        rules_cfg = getattr(getattr(config, "labels", None), "rules", None)
        v = getattr(rules_cfg, "vegetation", None)

        self._enabled        = bool(getattr(v, "enabled",           True))
        self._ndvi_thr       = float(getattr(v, "ndvi_threshold",   0.20))
        self._savi_thr       = float(getattr(v, "savi_threshold",   0.10))
        self._ndmi_thr       = float(getattr(v, "ndmi_threshold",   0.00))
        self._ndvi_weight    = float(getattr(v, "ndvi_weight",      0.50))
        self._savi_weight    = float(getattr(v, "savi_weight",      0.30))
        self._ndmi_weight    = float(getattr(v, "ndmi_weight",      0.20))
        self._scale          = float(getattr(v, "confidence_scale", 0.25))
        self._min_confidence = float(getattr(v, "min_confidence",   0.35))

    @property
    def class_id(self) -> int:
        return self._CLASS_ID

    @property
    def class_name(self) -> str:
        return self._CLASS_NAME

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def apply(
        self,
        band_data: SpectralBandData,
        context:   ClassificationContext | None = None,
    ) -> RuleResult:
        conditions = [
            ("NDVI", self._ndvi_thr, "greater", self._ndvi_weight, self._scale),
            ("SAVI", self._savi_thr, "greater", self._savi_weight, self._scale),
            ("NDMI", self._ndmi_thr, "greater", self._ndmi_weight, self._scale),
        ]
        evidence, used, missing = self._weighted_evidence(band_data, conditions)

        ndvi = band_data.bands.get("NDVI")

        if ndvi is not None:
            ndvi_gate = ndvi >= self._ndvi_thr
        else:
            ndvi_gate = np.zeros_like(evidence, dtype=bool)

        pixel_mask = ((evidence >= self._min_confidence) & (ndvi_gate))
        evidence   = np.where(pixel_mask, evidence, 0.0).astype(np.float32)
        return RuleResult(
            class_id=self._CLASS_ID, class_name=self._CLASS_NAME,
            confidence=evidence, pixel_mask=pixel_mask,
            bands_used=tuple(used), bands_missing=tuple(missing),
        )


# ==============================================================================
# BackgroundRule  -- ALWAYS REGISTERED LAST
# ==============================================================================

@RuleRegistry.register
class BackgroundRule(ClassificationRule):
    """
    Assigns Background (class_id=0) as a low-confidence fallback.

    Background receives a uniform, low confidence equal to min_confidence.
    It wins only where all other rules fail to meet their minimum evidence
    thresholds. This prevents large unclassified areas in non-fluvial terrain.

    BackgroundRule is always the last rule registered in RuleRegistry so that
    engine.rule_names[-1] == "background" is a stable invariant.

    All parameters come from config.labels.rules.background.
    context parameter accepted but currently ignored.
    """

    _CLASS_ID:   int = 0
    _CLASS_NAME: str = "background"

    def __init__(self, config: Any) -> None:
        rules_cfg = getattr(getattr(config, "labels", None), "rules", None)
        b = getattr(rules_cfg, "background", None)

        self._enabled        = bool(getattr(b, "enabled",        True))
        self._min_confidence = float(getattr(b, "min_confidence", 0.10))

    @property
    def class_id(self) -> int:
        return self._CLASS_ID

    @property
    def class_name(self) -> str:
        return self._CLASS_NAME

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def apply(
        self,
        band_data: SpectralBandData,
        context:   ClassificationContext | None = None,
    ) -> RuleResult:
        h, w       = band_data.height, band_data.width
        confidence = np.full((h, w), self._min_confidence, dtype=np.float32)
        pixel_mask = np.ones((h, w), dtype=bool)
        return RuleResult(
            class_id=self._CLASS_ID, class_name=self._CLASS_NAME,
            confidence=confidence, pixel_mask=pixel_mask,
            bands_used=(), bands_missing=(),
        )


# ==============================================================================
# RuleEngine
# ==============================================================================

class RuleEngine:
    """
    Applies all enabled ClassificationRules to a SpectralBandData object.

    Plugin-based design: RuleEngine.from_config() queries RuleRegistry for
    enabled rules rather than hardcoding the four concrete classes. Adding a
    new rule only requires decorating it with @RuleRegistry.register.

    Args:
        rules: List of enabled ClassificationRule instances. In production,
               always build via RuleEngine.from_config() which uses the
               registry. The constructor accepts an explicit list to support
               testing with injected rules.
    """

    def __init__(self, rules: list[ClassificationRule]) -> None:
        self._rules = list(rules)
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._logger.debug(
            "RuleEngine initialized with %d rule(s): %s",
            len(self._rules), [r.class_name for r in self._rules],
        )

    @classmethod
    def from_config(cls, config: Any) -> RuleEngine:
        """
        Build a RuleEngine by querying RuleRegistry for all enabled rules.

        Any rule class decorated with @RuleRegistry.register and whose
        corresponding config entry has enabled=True will be included.
        Registration order determines application order in apply_all().

        Args:
            config: Fully initialized Config object.

        Returns:
            RuleEngine with all registered enabled rules.
        """
        rules = RuleRegistry.create_enabled(config)
        return cls(rules)

    def apply_all(
        self,
        band_data: SpectralBandData,
        context:   ClassificationContext | None = None,
    ) -> list[RuleResult]:
        """
        Apply all enabled rules to band_data.

        Args:
            band_data: SpectralBandData for one patch.
            context:   Optional ClassificationContext passed to each rule.
                       Current rules ignore it; future rules may use it.

        Returns:
            List of RuleResult, one per enabled rule, in registration order.
        """
        results: list[RuleResult] = []
        for rule in self._rules:
            result = rule.apply(band_data, context=context)
            results.append(result)
            self._logger.debug(
                "Rule '%s': %d/%d pixels accepted, missing=%s",
                rule.class_name,
                int(result.pixel_mask.sum()),
                band_data.height * band_data.width,
                result.bands_missing,
            )
        return results

    @property
    def num_rules(self) -> int:
        """Number of active rules in this engine."""
        return len(self._rules)

    @property
    def rule_names(self) -> tuple[str, ...]:
        """Names of all active rules in application order."""
        return tuple(r.class_name for r in self._rules)

# """
# Spectral classification rules for the pseudo-label generation pipeline.

# Refinements in this pass:
#     1. Evidence-based scoring:
#        Each rule now uses _sigmoid_evidence() instead of the former piecewise-
#        linear _soft_threshold(). Sigmoid produces continuous evidence scores
#        that reflect spectral uncertainty near threshold boundaries, rather than
#        hard zeros below the threshold and a linear ramp above.

#     2. Plugin-based RuleRegistry:
#        RuleEngine no longer hardcodes the four rule classes. Instead, each rule
#        class registers itself via @RuleRegistry.register. Future rules (e.g.,
#        ShadowRule, MudRule) can be added by:
#          (a) Subclassing ClassificationRule, or
#          (b) Calling RuleRegistry.register_external(MyNewRule) at runtime.
#        RuleEngine.from_config() discovers all registered enabled rules.

#     3. Improved SandRule:
#        Sand evidence now optionally incorporates NDBI, NDMI (inverted),
#        SAVI (inverted), and optical brightness. All additional bands have
#        zero default weights (fully configurable; graceful handling of absent
#        bands preserves original behavior when not configured).

#     4. Seasonal interface preparation:
#        ClassificationRule.apply() now accepts an optional
#        ClassificationContext. All existing rules ignore it. Future rules
#        (e.g., FloodedVegetationRule) can inspect season/hydrological_year
#        without architecture changes.
# """

# from __future__ import annotations

# import logging
# from abc import ABC, abstractmethod
# from typing import Any

# import numpy as np

# from src.labels.contracts import ClassificationContext, RuleResult, SpectralBandData

# __all__ = [
#     "ClassificationRule",
#     "RuleRegistry",
#     "WaterRule",
#     "SandRule",
#     "VegetationRule",
#     "BackgroundRule",
#     "RuleEngine",
# ]

# _LOGGER: logging.Logger = logging.getLogger(__name__)
# _EPS: float = 1e-8


# # ==============================================================================
# # NEW: RuleRegistry
# # ==============================================================================

# class RuleRegistry:
#     """
#     Central registry for ClassificationRule plugins.

#     RuleEngine.from_config() queries this registry rather than hardcoding
#     which rules exist. Any rule class decorated with @RuleRegistry.register
#     is automatically discovered.

#     Usage — built-in rules register themselves at module import time:

#         @RuleRegistry.register
#         class WaterRule(ClassificationRule): ...

#     Usage — external plugin registration at runtime:

#         from src.labels.rules import RuleRegistry
#         RuleRegistry.register_external(MyCustomRule)

#     After registration, RuleEngine.from_config() will include the custom
#     rule without any changes to this file.
#     """

#     _registered: dict[str, type[ClassificationRule]] = {}

#     @classmethod
#     def register(
#         cls, rule_class: type[ClassificationRule]
#     ) -> type[ClassificationRule]:
#         """
#         Class decorator. Registers rule_class by its _CLASS_NAME.

#         Usage:
#             @RuleRegistry.register
#             class WaterRule(ClassificationRule): ...
#         """
#         cls._registered[rule_class._CLASS_NAME] = rule_class  # type: ignore[attr-defined]
#         _LOGGER.debug("RuleRegistry: registered '%s'", rule_class._CLASS_NAME)  # type: ignore[attr-defined]
#         return rule_class

#     @classmethod
#     def register_external(cls, rule_class: type[ClassificationRule]) -> None:
#         """
#         Imperatively register an external rule class.

#         Allows runtime plugin injection without the decorator syntax, e.g.
#         when the rule class is defined in a third-party package.
#         """
#         cls._registered[rule_class._CLASS_NAME] = rule_class  # type: ignore[attr-defined]
#         _LOGGER.debug("RuleRegistry: registered external '%s'", rule_class._CLASS_NAME)  # type: ignore[attr-defined]

#     @classmethod
#     def create_enabled(cls, config: Any) -> list[ClassificationRule]:
#         """
#         Instantiate all registered rules and return only the enabled ones.

#         Rules are instantiated in registration order so that tests can
#         reason about ordering. BackgroundRule (the fallback) should always
#         be registered last.

#         Args:
#             config: Fully initialized Config object.

#         Returns:
#             List of enabled ClassificationRule instances.
#         """
#         instances = [rule_cls(config) for rule_cls in cls._registered.values()]
#         enabled   = [r for r in instances if r.is_enabled]
#         _LOGGER.debug(
#             "RuleRegistry.create_enabled: %d/%d rules enabled: %s",
#             len(enabled), len(instances), [r.class_name for r in enabled],
#         )
#         return enabled

#     @classmethod
#     def registered_names(cls) -> tuple[str, ...]:
#         """Return names of all registered rule classes (enabled or not)."""
#         return tuple(cls._registered.keys())

#     @classmethod
#     def clear(cls) -> None:
#         """
#         Unregister all rules.

#         Intended for test isolation only. Do NOT call in production code.
#         """
#         cls._registered.clear()


# # ==============================================================================
# # Abstract base
# # ==============================================================================

# class ClassificationRule(ABC):
#     """
#     Abstract interface for all spectral classification rules.

#     Each subclass votes for one class across all pixels in a patch,
#     returning a per-pixel evidence-based confidence map.

#     Subclasses should:
#         - Define _CLASS_ID: int
#         - Define _CLASS_NAME: str
#         - Be decorated with @RuleRegistry.register to be auto-discovered.
#     """

#     @property
#     @abstractmethod
#     def class_id(self) -> int:
#         """Integer class label this rule votes for."""

#     @property
#     @abstractmethod
#     def class_name(self) -> str:
#         """Human-readable class name."""

#     @property
#     @abstractmethod
#     def is_enabled(self) -> bool:
#         """True if this rule is active (reads from config)."""

#     @abstractmethod
#     def apply(
#         self,
#         band_data: SpectralBandData,
#         context:   ClassificationContext | None = None,
#     ) -> RuleResult:
#         """
#         Apply this rule to one patch.

#         Args:
#             band_data: SpectralBandData with all available band arrays.
#             context:   Optional ClassificationContext for future seasonal
#                        classification. All current rules ignore this parameter.
#                        Future rules may inspect context.season, context.hydrological_year,
#                        or context.acquisition_date to adjust thresholds seasonally.

#         Returns:
#             RuleResult with per-pixel sigmoid-evidence confidence map.
#         """

#     # ------------------------------------------------------------------
#     # Protected helper — CHANGED: sigmoid replaces piecewise linear
#     # ------------------------------------------------------------------

#     @staticmethod
#     def _sigmoid_evidence(
#         values:    np.ndarray,
#         threshold: float,
#         scale:     float,
#         direction: str,
#     ) -> np.ndarray:
#         """
#         Convert band values to per-pixel evidence in (0, 1) using a sigmoid.

#         Replaces the former piecewise-linear _soft_threshold(). The sigmoid
#         produces continuous, differentiable evidence scores that correctly
#         model spectral measurement uncertainty near the threshold boundary.

#         Behavior:
#             direction="greater": evidence → 1 for values >> threshold;
#                                  evidence = 0.5 exactly at threshold;
#                                  evidence → 0 for values << threshold.
#             direction="less":    inverse of the above.

#             The steepness of the transition is controlled by `scale`.
#             A small scale means a sharp transition; a large scale means gradual.

#         Args:
#             values:    (H, W) float32 band array. NaN → 0.0 evidence.
#             threshold: Decision boundary value.
#             scale:     Width of the soft transition zone (e.g. 0.30).
#             direction: "greater" or "less".

#         Returns:
#             (H, W) float32 evidence array in (0.0, 1.0).
#         """
#         k = 1.0 / max(float(scale), _EPS)
#         if direction == "greater":
#             exponent = -k * (values - threshold)
#         else:  # "less"
#             exponent = -k * (threshold - values)

#         # Clip exponent to avoid overflow (exp(>709) overflows float64).
#         exponent_clipped = np.clip(exponent, -88.0, 88.0)
#         raw = (1.0 / (1.0 + np.exp(exponent_clipped))).astype(np.float32)
#         raw[np.isnan(values)] = 0.0
#         return raw

#     def _weighted_evidence(
#         self,
#         band_data:   SpectralBandData,
#         conditions:  list[tuple[str, float, str, float, float]],
#     ) -> tuple[np.ndarray, list[str], list[str]]:
#         """
#         Compute weighted-average evidence across multiple band conditions.

#         Each condition uses _sigmoid_evidence() instead of the former
#         _soft_threshold(), producing continuous evidence scores.

#         Args:
#             band_data:  SpectralBandData.
#             conditions: List of
#                         (band_name, threshold, direction, weight, scale).

#         Returns:
#             Tuple of:
#                 evidence_map: (H, W) float32 weighted evidence in (0, 1).
#                 bands_used:   Band names that contributed.
#                 bands_missing: Band names that were absent.
#         """
#         h, w    = band_data.height, band_data.width
#         accum   = np.zeros((h, w), dtype=np.float32)
#         total_w = 0.0
#         used:    list[str] = []
#         missing: list[str] = []

#         for band_name, threshold, direction, weight, scale in conditions:
#             if weight <= 0.0:
#                 continue
#             if band_name not in band_data.bands:
#                 missing.append(band_name)
#                 continue
#             arr      = band_data.bands[band_name]
#             evidence = self._sigmoid_evidence(arr, threshold, scale, direction)
#             accum   += evidence * weight
#             total_w += weight
#             used.append(band_name)

#         if total_w < _EPS:
#             return np.zeros((h, w), dtype=np.float32), used, missing

#         return (accum / total_w).astype(np.float32), used, missing


# # ==============================================================================
# # WaterRule
# # ==============================================================================

# @RuleRegistry.register
# class WaterRule(ClassificationRule):
#     """
#     Classifies pixels as Water (class_id=1) using sigmoid-weighted evidence.

#     Evidence combines:
#         MNDWI (primary) + NDWI + AWEI_nsh + AWEI_sh (water-positive indices)
#         + (1 - NDVI) evidence (vegetation-negative contributes to water)

#     All weights and thresholds are configurable.
#     context parameter accepted but currently ignored.
#     """

#     _CLASS_ID:   int = 1
#     _CLASS_NAME: str = "water"

#     def __init__(self, config: Any) -> None:
#         rules_cfg = getattr(getattr(config, "labels", None), "rules", None)
#         w = getattr(rules_cfg, "water", None)

#         self._enabled          = bool(getattr(w, "enabled",            True))
#         self._mndwi_thr        = float(getattr(w, "mndwi_threshold",    0.2))
#         self._ndwi_thr         = float(getattr(w, "ndwi_threshold",     0.0))
#         self._awei_nsh_thr     = float(getattr(w, "awei_nsh_threshold", 0.0))
#         self._awei_sh_thr      = float(getattr(w, "awei_sh_threshold",  0.0))
#         self._ndvi_max_thr     = float(getattr(w, "ndvi_max_threshold", 0.05))
#         self._mndwi_weight     = float(getattr(w, "mndwi_weight",       0.40))
#         self._ndwi_weight      = float(getattr(w, "ndwi_weight",        0.20))
#         self._awei_nsh_weight  = float(getattr(w, "awei_nsh_weight",    0.15))
#         self._awei_sh_weight   = float(getattr(w, "awei_sh_weight",     0.05))
#         self._ndvi_weight      = float(getattr(w, "ndvi_weight",        0.20))
#         self._scale            = float(getattr(w, "confidence_scale",   0.30))
#         self._min_confidence   = float(getattr(w, "min_confidence",     0.40))

#     @property
#     def class_id(self) -> int:
#         return self._CLASS_ID

#     @property
#     def class_name(self) -> str:
#         return self._CLASS_NAME

#     @property
#     def is_enabled(self) -> bool:
#         return self._enabled

#     def apply(
#         self,
#         band_data: SpectralBandData,
#         context:   ClassificationContext | None = None,
#     ) -> RuleResult:
#         conditions = [
#             ("MNDWI",    self._mndwi_thr,    "greater", self._mndwi_weight,    self._scale),
#             ("NDWI",     self._ndwi_thr,     "greater", self._ndwi_weight,     self._scale),
#             ("AWEI_nsh", self._awei_nsh_thr, "greater", self._awei_nsh_weight, self._scale),
#             ("AWEI_sh",  self._awei_sh_thr,  "greater", self._awei_sh_weight,  self._scale),
#             ("NDVI",     self._ndvi_max_thr, "less",    self._ndvi_weight,     self._scale),
#         ]
#         evidence, used, missing = self._weighted_evidence(band_data, conditions)
#         pixel_mask = evidence >= self._min_confidence
#         evidence   = np.where(pixel_mask, evidence, 0.0).astype(np.float32)
#         return RuleResult(
#             class_id=self._CLASS_ID, class_name=self._CLASS_NAME,
#             confidence=evidence, pixel_mask=pixel_mask,
#             bands_used=tuple(used), bands_missing=tuple(missing),
#         )


# # ==============================================================================
# # SandRule  (IMPROVED)
# # ==============================================================================

# @RuleRegistry.register
# class SandRule(ClassificationRule):
#     """
#     Classifies pixels as Sand/bare sediment (class_id=2).

#     Improvements over v1:
#         Added support for NDBI, NDMI (inverted — dry sand has low moisture),
#         SAVI (inverted — sparse vegetation cover), and optical brightness
#         (sand is bright across all optical bands). All new bands default
#         to zero weight so original behaviour is preserved when not configured.

#     Evidence combines (configurable weights):
#         MNDWI_inverted   -- sand is NOT water
#         BSI              -- bare soil indicator
#         NDVI_inverted    -- sand has little vegetation
#         NDWI_inverted    -- sand has no water signal
#         NDBI             -- built-up/bare surface (optional, wt=0)
#         NDMI_inverted    -- dry sand has low moisture (optional, wt=0)
#         SAVI_inverted    -- soil-adjusted; low for sand (optional, wt=0)
#         brightness       -- mean optical brightness high for sand (optional, wt=0)
#     """

#     _CLASS_ID:   int = 2
#     _CLASS_NAME: str = "sand"

#     def __init__(self, config: Any) -> None:
#         rules_cfg = getattr(getattr(config, "labels", None), "rules", None)
#         s = getattr(rules_cfg, "sand", None)

#         self._enabled         = bool(getattr(s, "enabled",            True))
#         self._mndwi_max_thr   = float(getattr(s, "mndwi_max_threshold", 0.0))
#         self._bsi_thr         = float(getattr(s, "bsi_threshold",       0.0))
#         self._ndvi_max_thr    = float(getattr(s, "ndvi_max_threshold",  0.25))
#         self._ndwi_max_thr    = float(getattr(s, "ndwi_max_threshold",  0.0))
#         self._mndwi_weight    = float(getattr(s, "mndwi_weight",        0.30))
#         self._bsi_weight      = float(getattr(s, "bsi_weight",          0.35))
#         self._ndvi_weight     = float(getattr(s, "ndvi_weight",         0.25))
#         self._ndwi_weight     = float(getattr(s, "ndwi_weight",         0.10))
#         self._scale           = float(getattr(s, "confidence_scale",    0.30))
#         self._min_confidence  = float(getattr(s, "min_confidence",      0.30))

#         # NEW: extended indices (zero defaults preserve v1 behaviour)
#         self._ndbi_thr        = float(getattr(s, "ndbi_threshold",      0.0))
#         self._ndbi_weight     = float(getattr(s, "ndbi_weight",         0.0))
#         self._ndmi_max_thr    = float(getattr(s, "ndmi_max_threshold",  0.0))
#         self._ndmi_weight     = float(getattr(s, "ndmi_weight",         0.0))
#         self._savi_max_thr    = float(getattr(s, "savi_max_threshold",  0.15))
#         self._savi_weight     = float(getattr(s, "savi_weight",         0.0))
#         self._brightness_thr  = float(getattr(s, "brightness_threshold", 0.2))
#         self._brightness_weight = float(getattr(s, "brightness_weight", 0.0))

#     @property
#     def class_id(self) -> int:
#         return self._CLASS_ID

#     @property
#     def class_name(self) -> str:
#         return self._CLASS_NAME

#     @property
#     def is_enabled(self) -> bool:
#         return self._enabled

#     def apply(
#         self,
#         band_data: SpectralBandData,
#         context:   ClassificationContext | None = None,
#     ) -> RuleResult:
#         conditions = [
#             ("MNDWI", self._mndwi_max_thr, "less",    self._mndwi_weight,     self._scale),
#             ("BSI",   self._bsi_thr,        "greater", self._bsi_weight,        self._scale),
#             ("NDVI",  self._ndvi_max_thr,   "less",    self._ndvi_weight,       self._scale),
#             ("NDWI",  self._ndwi_max_thr,   "less",    self._ndwi_weight,       self._scale),
#             # Extended — zero weight unless configured; absent bands silently skipped
#             ("NDBI",  self._ndbi_thr,        "greater", self._ndbi_weight,      self._scale),
#             ("NDMI",  self._ndmi_max_thr,    "less",    self._ndmi_weight,      self._scale),
#             ("SAVI",  self._savi_max_thr,    "less",    self._savi_weight,      self._scale),
#         ]

#         # Optical brightness: mean of Blue, Green, Red, NIR if available.
#         # Computed inline to avoid requiring a separate "brightness" band in the GeoTIFF.
#         if self._brightness_weight > 0.0:
#             band_data = self._inject_brightness(band_data)
#             conditions.append(
#                 ("_brightness", self._brightness_thr, "greater",
#                  self._brightness_weight, self._scale)
#             )

#         evidence, used, missing = self._weighted_evidence(band_data, conditions)
#         pixel_mask = evidence >= self._min_confidence
#         evidence   = np.where(pixel_mask, evidence, 0.0).astype(np.float32)

#         # Remove internal brightness band from reported used list
#         used_clean = [b for b in used if not b.startswith("_")]
#         return RuleResult(
#             class_id=self._CLASS_ID, class_name=self._CLASS_NAME,
#             confidence=evidence, pixel_mask=pixel_mask,
#             bands_used=tuple(used_clean), bands_missing=tuple(missing),
#         )

#     @staticmethod
#     def _inject_brightness(band_data: SpectralBandData) -> SpectralBandData:
#         """
#         Compute optical brightness = mean(Blue, Green, Red, NIR) and inject
#         as a temporary "_brightness" band. Returns a shallow copy of band_data
#         with the extra band; the source SpectralBandData is NOT mutated.
#         """
#         optical = [b for b in ("Blue", "Green", "Red", "NIR") if b in band_data.bands]
#         if not optical:
#             return band_data
#         brightness = np.stack(
#             [band_data.bands[b] for b in optical], axis=0
#         ).mean(axis=0).astype(np.float32)
#         new_bands = dict(band_data.bands)
#         new_bands["_brightness"] = brightness
#         return SpectralBandData(
#             bands=new_bands, height=band_data.height, width=band_data.width,
#             crs=band_data.crs, transform=band_data.transform,
#             band_names=band_data.band_names + ("_brightness",),
#         )


# # ==============================================================================
# # VegetationRule
# # ==============================================================================

# @RuleRegistry.register
# class VegetationRule(ClassificationRule):
#     """
#     Classifies pixels as Vegetation (class_id=3).
#     context parameter accepted but currently ignored.
#     """

#     _CLASS_ID:   int = 3
#     _CLASS_NAME: str = "vegetation"

#     def __init__(self, config: Any) -> None:
#         rules_cfg = getattr(getattr(config, "labels", None), "rules", None)
#         v = getattr(rules_cfg, "vegetation", None)

#         self._enabled        = bool(getattr(v, "enabled",           True))
#         self._ndvi_thr       = float(getattr(v, "ndvi_threshold",   0.20))
#         self._savi_thr       = float(getattr(v, "savi_threshold",   0.10))
#         self._ndmi_thr       = float(getattr(v, "ndmi_threshold",   0.00))
#         self._ndvi_weight    = float(getattr(v, "ndvi_weight",      0.50))
#         self._savi_weight    = float(getattr(v, "savi_weight",      0.30))
#         self._ndmi_weight    = float(getattr(v, "ndmi_weight",      0.20))
#         self._scale          = float(getattr(v, "confidence_scale", 0.25))
#         self._min_confidence = float(getattr(v, "min_confidence",   0.35))

#     @property
#     def class_id(self) -> int:
#         return self._CLASS_ID

#     @property
#     def class_name(self) -> str:
#         return self._CLASS_NAME

#     @property
#     def is_enabled(self) -> bool:
#         return self._enabled

#     def apply(
#         self,
#         band_data: SpectralBandData,
#         context:   ClassificationContext | None = None,
#     ) -> RuleResult:
#         conditions = [
#             ("NDVI", self._ndvi_thr, "greater", self._ndvi_weight, self._scale),
#             ("SAVI", self._savi_thr, "greater", self._savi_weight, self._scale),
#             ("NDMI", self._ndmi_thr, "greater", self._ndmi_weight, self._scale),
#         ]
#         evidence, used, missing = self._weighted_evidence(band_data, conditions)
#         pixel_mask = evidence >= self._min_confidence
#         evidence   = np.where(pixel_mask, evidence, 0.0).astype(np.float32)
#         return RuleResult(
#             class_id=self._CLASS_ID, class_name=self._CLASS_NAME,
#             confidence=evidence, pixel_mask=pixel_mask,
#             bands_used=tuple(used), bands_missing=tuple(missing),
#         )


# # ==============================================================================
# # BackgroundRule  (always registered last — lowest-priority fallback)
# # ==============================================================================

# @RuleRegistry.register
# class BackgroundRule(ClassificationRule):
#     """
#     Assigns Background (class_id=0) as a low-confidence fallback.
#     context parameter accepted but currently ignored.
#     """

#     _CLASS_ID:   int = 0
#     _CLASS_NAME: str = "background"

#     def __init__(self, config: Any) -> None:
#         rules_cfg = getattr(getattr(config, "labels", None), "rules", None)
#         b = getattr(rules_cfg, "background", None)

#         self._enabled        = bool(getattr(b, "enabled",        True))
#         self._min_confidence = float(getattr(b, "min_confidence", 0.10))

#     @property
#     def class_id(self) -> int:
#         return self._CLASS_ID

#     @property
#     def class_name(self) -> str:
#         return self._CLASS_NAME

#     @property
#     def is_enabled(self) -> bool:
#         return self._enabled

#     def apply(
#         self,
#         band_data: SpectralBandData,
#         context:   ClassificationContext | None = None,
#     ) -> RuleResult:
#         h, w       = band_data.height, band_data.width
#         confidence = np.full((h, w), self._min_confidence, dtype=np.float32)
#         pixel_mask = np.ones((h, w), dtype=bool)
#         return RuleResult(
#             class_id=self._CLASS_ID, class_name=self._CLASS_NAME,
#             confidence=confidence, pixel_mask=pixel_mask,
#             bands_used=(), bands_missing=(),
#         )


# # ==============================================================================
# # RuleEngine  (updated: uses RuleRegistry)
# # ==============================================================================

# class RuleEngine:
#     """
#     Applies all enabled ClassificationRules to a SpectralBandData object.

#     Plugin-based design: RuleEngine queries RuleRegistry for enabled rules
#     rather than hardcoding WaterRule, SandRule, etc. Adding a new rule
#     only requires decorating it with @RuleRegistry.register.

#     Args:
#         rules: List of enabled ClassificationRule instances. In production,
#                always build via RuleEngine.from_config() which uses the
#                registry. The constructor accepts an explicit list to support
#                testing with injected rules.
#     """

#     def __init__(self, rules: list[ClassificationRule]) -> None:
#         self._rules = list(rules)
#         self._logger: logging.Logger = logging.getLogger(__name__)
#         self._logger.debug(
#             "RuleEngine initialized with %d rule(s): %s",
#             len(self._rules), [r.class_name for r in self._rules],
#         )

#     @classmethod
#     def from_config(cls, config: Any) -> RuleEngine:
#         """
#         Build a RuleEngine by querying RuleRegistry for all enabled rules.

#         Any rule class decorated with @RuleRegistry.register and whose
#         config.labels.rules.<name>.enabled is True will be included.
#         The order of rules in the registry determines application order
#         in apply_all() and therefore conflict resolution priority when
#         using highest_confidence strategy.

#         Args:
#             config: Fully initialized Config object.

#         Returns:
#             RuleEngine with all registered enabled rules.
#         """
#         rules = RuleRegistry.create_enabled(config)
#         return cls(rules)

#     def apply_all(
#         self,
#         band_data: SpectralBandData,
#         context:   ClassificationContext | None = None,
#     ) -> list[RuleResult]:
#         """
#         Apply all enabled rules to band_data.

#         Args:
#             band_data: SpectralBandData for one patch.
#             context:   Optional ClassificationContext passed to each rule.
#                        Current rules ignore it; future rules may use it.

#         Returns:
#             List of RuleResult, one per enabled rule.
#         """
#         results = []
#         for rule in self._rules:
#             result = rule.apply(band_data, context=context)
#             results.append(result)
#             self._logger.debug(
#                 "Rule '%s': %d/%d pixels accepted, missing=%s",
#                 rule.class_name,
#                 int(result.pixel_mask.sum()),
#                 band_data.height * band_data.width,
#                 result.bands_missing,
#             )
#         return results

#     @property
#     def num_rules(self) -> int:
#         """Number of active rules in this engine."""
#         return len(self._rules)

#     @property
#     def rule_names(self) -> tuple[str, ...]:
#         """Names of all active rules in application order."""
#         return tuple(r.class_name for r in self._rules)