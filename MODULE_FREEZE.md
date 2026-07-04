# MODULE_FREEZE.md

> River Morphology Segmentation
>
> Module Stability & Architectural Contract
>
> Version: 1.0

---

# Purpose

This document defines the architectural contracts of every completed module.

Once a module is marked **FROZEN**, its public API becomes part of the repository contract.

Future development must build on top of these modules instead of modifying them.

Breaking changes should only occur if:

- a confirmed bug exists
- an architectural redesign has been explicitly approved
- backward compatibility can be preserved

This document should always be read before implementing a new module.

---

# Repository Status

| Module | Title | Status |
|---------|-------|--------|
| Module 1 | Configuration System | FROZEN |
| Module 2 | Bootstrap & Directory Management | FROZEN |
| Module 3 | Google Earth Engine Client | FROZEN |
| Module 4 | Landsat Collection Builder | FROZEN |
| Module 5 | Landsat Image Preprocessing | FROZEN |
| Module 6 | Spectral Feature Generation | FROZEN |
| Module 7 | Dataset Export | FROZEN |
| Module 8 | Patch Generation | FROZEN |
| Module 9 | Label Generation | UNDER REDESIGN |
| Module 10 | Dataset Assembly | FROZEN |
| Module 11 | PyTorch Dataset & DataLoader | FROZEN |

---

# Global Repository Rules

The following rules apply to the entire repository.

- Use Python 3.11+
- Absolute imports only
- ASCII characters only in Python source
- No print()
- Use project logging
- Use project exceptions
- Use Config for all configurable values
- No duplicated logic
- Preserve backward compatibility
- Use immutable data contracts
- Maintain high unit-test coverage
- PEP8 compliant
- Complete type hints
- SOLID principles
- Production-quality code only

---

# Module 1

## Title

Configuration System

---

## Status

FROZEN

---

## Responsibilities

- Configuration loading
- YAML parsing
- Configuration validation
- Logging configuration
- Environment configuration

---

## Public Components

Examples:

Config

ConfigurationValidator

Logging Configuration

---

## Input

Configuration files

---

## Output

Typed configuration objects

---

## Future Modules MUST

Reuse Config.

Never introduce independent configuration systems.

Never hardcode values.

---

## Future Modules MUST NOT

- hardcode thresholds
- hardcode paths
- create additional configuration loaders

---

# Module 2

## Title

Bootstrap & Directory Management

---

## Status

FROZEN

---

## Responsibilities

Application startup

Directory creation

Environment validation

Logging initialization

---

## Public Components

Bootstrap

DirectoryManager

---

## Future Modules MUST

Reuse the existing startup sequence.

---

## Future Modules MUST NOT

Create directories independently.

Modify application startup.

Replace bootstrap flow.

---

# Module 3

## Title

Google Earth Engine

---

## Status

FROZEN

---

## Responsibilities

Authentication

Health checks

Retry logic

Earth Engine communication

---

## Public Components

EarthEngineClient

AuthManager

HealthChecker

---

## Future Modules MUST

Always use EarthEngineClient.

---

## Future Modules MUST NOT

Import Earth Engine directly.

Duplicate retry logic.

Implement independent authentication.

Expose raw Earth Engine exceptions.

---

# Module 4

## Title

Landsat Collection Builder

---

## Status

FROZEN

---

## Responsibilities

AOI validation

Collection creation

Date validation

Cloud filtering

Sensor validation

---

## Public Components

LandsatCollectionBuilder

---

## Future Modules MUST

Use LandsatCollectionBuilder.

---

## Future Modules MUST NOT

Construct Earth Engine collections manually.

Hardcode collection IDs.

Duplicate validation logic.

---

# Module 5

## Title

Landsat Preprocessing

---

## Status

FROZEN

---

## Responsibilities

Cloud masking

Shadow masking

Band harmonization

Scaling

Composite generation

---

## Public Components

LandsatPreprocessor

BandHarmonizer

CompositeGenerator

---

## Future Modules MUST

Reuse preprocessing pipeline.

Use harmonized band names only.

---

## Future Modules MUST NOT

Reference raw Landsat bands.

Duplicate cloud masking.

Duplicate scaling.

Create manual composites.

---

# Module 6

## Title

Spectral Feature Generation

---

## Status

FROZEN

---

## Responsibilities

- Spectral index computation
- Feature stack generation
- Feature registry management
- Harmonized feature creation

---

## Public Components

Examples:

- FeatureRegistry
- FeatureStackAssembler
- FeatureStackResult

---

## Input

Preprocessed Landsat composite

---

## Output

FeatureStackResult

---

## Future Modules MUST

- Reuse `FeatureRegistry`
- Reuse `FeatureStackAssembler`
- Use harmonized band names only
- Consume `FeatureStackResult`

---

## Future Modules MUST NOT

- Compute indices manually
- Duplicate feature calculations
- Hardcode index names
- Use raw Landsat band names

---

# Module 7

## Title

Dataset Export

---

## Status

FROZEN

---

## Responsibilities

- GeoTIFF export
- Metadata generation
- Manifest generation
- Dataset versioning

---

## Public Components

Examples:

- DatasetExporter
- MetadataWriter
- DatasetManifestManager
- DatasetVersionManager

---

## Input

FeatureStackResult

---

## Output

DatasetExportResult

---

## Future Modules MUST

Reuse DatasetExporter.

Preserve:

- CRS
- Affine Transform
- Band order
- Band names
- float32 feature stacks

---

## Future Modules MUST NOT

- Write GeoTIFFs directly
- Generate metadata manually
- Create manifests manually
- Create version.json manually

---

# Module 8

## Title

Patch Generation

---

## Status

FROZEN

---

## Responsibilities

- Sliding window patch generation
- Patch validation
- Patch metadata
- Patch manifests

---

## Public Components

Examples:

- PatchGenerator
- PatchDatasetResult

---

## Input

DatasetExportResult

---

## Output

PatchDatasetResult

---

## Future Modules MUST

Consume PatchDatasetResult.

Reuse existing patch metadata.

---

## Future Modules MUST NOT

- Regenerate image patches
- Modify patch geometry
- Duplicate patch generation logic

---

# Module 9

## Title

Label Generation

---

## Status

UNDER ARCHITECTURAL REDESIGN

Public API remains stable.

---

## Current Public Contract

LabelDatasetResult

---

## Approved Future Architecture

Feature Stack

↓

SpectralClassificationEngine

↓

RuleEngine

↓

ConflictResolver

↓

MorphologyProcessor

↓

QualityAssessment

↓

ConfidenceEstimator

↓

MetadataGenerator

↓

StatisticsGenerator

↓

ManifestManager

↓

VersionManager

↓

LabelDatasetResult

---

## Future Modules MUST

Consume only:

LabelDatasetResult

Do not depend on internal implementation.

---

## Future Modules MUST NOT

Assume labels are manually created.

Assume threshold logic.

Depend on internal rule implementations.

---

# Module 10

## Title

Dataset Assembly

---

## Status

FROZEN

---

## Responsibilities

- Image/label pairing
- Dataset validation
- Dataset splitting
- Leakage prevention
- Dataset statistics

---

## Public Components

Examples:

- DatasetAssembler
- DatasetSplitter
- DatasetValidator
- DataLeakageDetector
- TrainingDatasetResult

---

## Input

PatchDatasetResult

+

LabelDatasetResult

---

## Output

TrainingDatasetResult

---

## Future Modules MUST

Consume TrainingDatasetResult.

Respect existing train/validation/test split logic.

---

## Future Modules MUST NOT

Modify split strategy.

Duplicate leakage detection.

Create independent dataset assembly logic.

---

# Module 11

## Title

PyTorch Dataset & DataLoader

---

## Status

FROZEN

---

## Responsibilities

- Manifest reading
- GeoTIFF loading
- Runtime validation
- Tensor conversion
- Dataset creation
- DataLoader creation

---

## Public Components

Examples:

- RiverMorphologyDataset
- DatasetReader
- GeoTIFFLoader
- DataLoaderFactory
- Transform
- IdentityTransform
- TorchDatasetResult

---

## Input

TrainingDatasetResult

---

## Output

TorchDatasetResult

---

## Future Modules MUST

Reuse:

- RiverMorphologyDataset
- Transform interface
- DataLoaderFactory

Augmentation should implement the existing Transform interface.

---

## Future Modules MUST NOT

Modify Dataset implementation.

Embed model logic.

Embed training logic.

Embed augmentation logic directly into Dataset.

---

# Public Data Contract Freeze

The following result objects are considered stable.

Config

↓

CollectionResult

↓

ProcessedCollectionResult

↓

FeatureStackResult

↓

DatasetExportResult

↓

PatchDatasetResult

↓

LabelDatasetResult

↓

TrainingDatasetResult

↓

TorchDatasetResult

Future modules should extend this chain rather than replacing it.

---

# Repository-Level Architectural Contracts

The following architectural decisions are permanent.

Configuration

- Single configuration source
- Configuration-driven behaviour
- No hardcoded values

Logging

- Centralized logging
- No print()

Exceptions

- Centralized exception hierarchy
- Wrap third-party exceptions

Data

- Immutable public contracts
- Typed dataclasses

Testing

- Unit tests for every module
- ≥90% coverage target
- Mock external services

Documentation

Every completed module should update:

- PROJECT_CONTEXT.md
- ARCHITECTURE.md
- MODULE_FREEZE.md
- AI_HANDOFF.md (if onboarding changes)

---

# Checklist Before Implementing Any Future Module

Before writing code, verify:

- [ ] Existing functionality cannot be reused
- [ ] No frozen public API is modified
- [ ] No duplicated logic is introduced
- [ ] Configuration comes from Config
- [ ] Logging uses project logger
- [ ] Exceptions use project hierarchy
- [ ] Public interfaces remain stable
- [ ] Unit tests are included
- [ ] Documentation is updated

---

# Freeze Policy

A module is considered FROZEN only when:

- Implementation is complete
- Unit tests pass
- Public API reviewed
- Documentation updated
- Architecture reviewed
- GitHub commit completed

Once frozen, future work should extend—not modify—the module whenever possible.

---

# End of Document