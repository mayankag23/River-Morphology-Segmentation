# PROJECT_MODULE_INDEX.md

# River Morphology Segmentation using Multi-Temporal Landsat 8/9 Imagery

**Version:** 1.0.0

**Project Status:** Production Ready (Modules 1–20 Completed)

**Repository Type:** End-to-End Deep Learning Framework

**Primary Language:** Python 3.11+

**Deep Learning Framework:** PyTorch

**Satellite Platform:** Landsat 8 / Landsat 9 Collection 2 Level-2

**Primary Task:** Semantic Segmentation of River Morphology

---

# Table of Contents

1. Project Overview
2. System Objectives
3. Overall Architecture
4. Repository Structure
5. Module Dependency Graph
6. Global Data Flow
7. Public Data Contracts
8. Configuration Mapping
9. Design Principles
10. Technology Stack
11. Development Standards

(Module-specific documentation begins in Part 2.)

---

# 1. Project Overview

## Purpose

This repository implements a complete production-ready framework for automated river morphology analysis using multi-temporal Landsat satellite imagery.

Unlike a research prototype, the project is designed as a modular software engineering system with clearly defined interfaces, deterministic execution, configuration-driven behavior, and reusable components.

The framework performs the complete workflow:

- Satellite image acquisition
- Image preprocessing
- Spectral feature engineering
- GeoTIFF export
- Patch generation
- Automatic pseudo-label generation
- Dataset assembly
- PyTorch dataset preparation
- Data augmentation
- Deep learning model training
- Model evaluation
- Inference on unseen AOIs
- River morphology analysis
- Visualization
- Scientific reporting

The project is intended for research, environmental monitoring, river management, and long-term geomorphological studies.

---

# 2. System Objectives

The framework has the following primary objectives.

## Functional Objectives

- Download Landsat imagery directly from Google Earth Engine.
- Build cloud-free temporal composites.
- Generate spectral feature stacks.
- Export analysis-ready GeoTIFF datasets.
- Produce training patches.
- Automatically generate pseudo-labels without manual annotation.
- Assemble deterministic machine learning datasets.
- Train semantic segmentation models.
- Evaluate model performance using standard segmentation metrics.
- Perform inference on unseen AOIs.
- Quantify river morphology.
- Produce publication-quality figures.
- Generate complete experiment reports.

---

## Software Engineering Objectives

The repository prioritizes:

- Modularity
- Maintainability
- Extensibility
- Reproducibility
- Testability
- Configuration-driven execution
- Strong typing
- Immutable public contracts
- Minimal coupling
- Production readiness

---

# 3. Overall Architecture

The project follows a strictly sequential pipeline.

```

Configuration
│
▼
Bootstrap
│
▼
Google Earth Engine
│
▼
Image Collection
│
▼
Preprocessing
│
▼
Feature Engineering
│
▼
GeoTIFF Export
│
▼
Patch Generation
│
▼
Pseudo Label Generation
│
▼
Dataset Assembly
│
▼
PyTorch Dataset
│
▼
Data Augmentation
│
▼
UNet++
│
▼
Training
│
▼
Evaluation
│
▼
Inference
│
▼
River Morphology Analytics
│
▼
Visualization
│
▼
Reporting

```

Every stage consumes the output contract of the previous stage.

Reverse dependencies are prohibited.

---

# 4. Repository Structure

```

RiverMorphology/

│
├── config/
│
├── docs/
│
├── src/
│   ├── core/
│   ├── gee/
│   ├── export/
│   ├── patches/
│   ├── labels/
│   ├── dataset/
│   ├── training/
│   │   ├── evaluation/
│   │   ├── inference/
│   │   ├── losses/
│   │   ├── models/
│   │   └── trainer/
│   ├── morphology/
│   ├── visualization/
│   ├── reporting/
│   └── pipeline/
│
├── tests/
│
├── outputs/
│
├── checkpoints/
│
├── logs/
│
└── main.py

```

The repository is organized into independent domain packages.

Each package exposes only a small public API.

---

# 5. Module Dependency Graph

| Module | Name | Depends On |
|---------|------|------------|
| Module 1 | Configuration System | None |
| Module 2 | Bootstrap & Directory Management | Module 1 |
| Module 3 | Google Earth Engine Client | Modules 1–2 |
| Module 4 | Landsat Collection Builder | Module 3 |
| Module 5 | Landsat Image Preprocessing | Module 4 |
| Module 6 | Spectral Feature Engineering | Module 5 |
| Module 7 | GeoTIFF Export | Module 6 |
| Module 8 | Patch Generation | Module 7 |
| Module 9 | Pseudo Label Generation | Module 8 |
| Module 10 | Dataset Assembly | Module 9 |
| Module 11 | PyTorch Dataset & DataLoader | Module 10 |
| Module 12 | Data Transformation & Augmentation | Module 11 |
| Module 13 | Segmentation Model Zoo | Module 12 |
| Module 14 | Training Engine | Module 13 |
| Module 15 | Evaluation Framework | Module 14 |
| Module 16 | Inference Pipeline | Modules 14–15 |
| Module 17 | River Morphology Analytics | Module 16 |
| Module 18 | Visualization Framework | Module 17 |
| Module 19 | Reporting Framework | Modules 15–18 |
| Module 20 | End-to-End Pipeline Orchestration | Modules 1–19 |

---

# 6. Global Data Flow

The framework passes strongly typed immutable contracts between modules.

```

Config
│
▼
CollectionResult
│
▼
ProcessedCollectionResult
│
▼
FeatureStackResult
│
▼
DatasetExportResult
│
▼
PatchDatasetResult
│
▼
LabelDatasetResult
│
▼
TrainingDatasetResult
│
▼
TorchDatasetResult
│
▼
TransformPipelineResult
│
▼
ModelResult
│
▼
TrainingResult
│
▼
EvaluationResult
│
▼
InferenceResult
│
▼
RiverMorphologyResult
│
▼
VisualizationResult
│
▼
ReportResult
│
▼
PipelineResult

```

Every contract is immutable.

Each module owns only its own contract.

---

# 7. Public Data Contracts

The following contracts are considered frozen public interfaces.

| Contract | Produced By | Consumed By |
|-----------|-------------|-------------|
| Config | Module 1 | All Modules |
| CollectionResult | Module 4 | Module 5 |
| ProcessedCollectionResult | Module 5 | Module 6 |
| FeatureStackResult | Module 6 | Module 7 |
| DatasetExportResult | Module 7 | Module 8 |
| PatchDatasetResult | Module 8 | Module 9 |
| LabelDatasetResult | Module 9 | Module 10 |
| TrainingDatasetResult | Module 10 | Module 11 |
| TorchDatasetResult | Module 11 | Module 12 |
| TransformPipelineResult | Module 12 | Module 13 |
| ModelResult | Module 13 | Module 14 |
| TrainingResult | Module 14 | Modules 15–16 |
| EvaluationResult | Module 15 | Module 19 |
| InferenceResult | Module 16 | Module 17 |
| RiverMorphologyResult | Module 17 | Modules 18–19 |
| VisualizationResult | Module 18 | Module 19 |
| ReportResult | Module 19 | Module 20 |
| PipelineResult | Module 20 | End User |

These contracts must remain backward compatible.

---

# 8. Configuration Mapping

Each configuration section is owned by exactly one module.

| Configuration Section | Primary Owner |
|-----------------------|---------------|
| project | Module 1 |
| paths | Module 2 |
| gee | Module 3 |
| aoi | Module 4 |
| satellite | Module 4 |
| date_range | Module 4 |
| preprocessing | Module 5 |
| spectral_bands | Module 6 |
| export | Module 7 |
| patch_generation | Module 8 |
| label_generation | Module 9 |
| classes | Module 9 |
| dataset | Module 10 |
| augmentation | Module 12 |
| model | Module 13 |
| loss | Module 14 |
| optimizer | Module 14 |
| scheduler | Module 14 |
| training | Module 14 |
| evaluation | Module 15 |
| inference | Module 16 |
| morphology | Module 17 |
| visualization | Module 18 |
| reporting | Module 19 |
| pipeline | Module 20 |

---

# 9. Design Principles

The repository follows the following engineering principles.

- SOLID architecture.
- Single Responsibility Principle.
- Dependency Injection where appropriate.
- Immutable public contracts.
- Configuration-driven execution.
- No hardcoded paths.
- Deterministic execution.
- Production-grade logging.
- Strong typing.
- High unit-test coverage.
- Backward compatibility after module approval.

Every module exposes one primary public interface.

Internal implementations remain private.

---

# 10. Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| Deep Learning | PyTorch |
| Satellite Platform | Landsat 8/9 Collection 2 Level-2 |
| Remote Sensing | Google Earth Engine |
| Raster IO | Rasterio |
| Numerical Computing | NumPy |
| Scientific Computing | SciPy |
| Data Handling | Pandas |
| Visualization | Matplotlib |
| Image Processing | Pillow |
| Configuration | YAML |
| Testing | Pytest |
| Type Checking | Python Typing |

---

# 11. Development Standards

The following standards apply throughout the repository.

## Coding Standards

- PEP 8 compliant.
- Python 3.11+.
- Full type hints.
- Absolute imports only.
- ASCII source files.
- No wildcard imports.
- No print() statements.
- Logging framework only.

## Testing Standards

- Every public module has unit tests.
- External services are mocked.
- Deterministic tests.
- High code coverage.
- Public contracts verified.

## Architectural Standards

- Public APIs are frozen after approval.
- Output contracts must remain backward compatible.
- No circular dependencies.
- Modules communicate only through contracts.
- Module boundaries must not be violated.

---

**End of Part 1**

The following parts document each module individually, including:

- Features Implemented
- Architecture Decisions
- Public Interfaces
- Source Files
- Test Files
- Input Contracts
- Output Contracts
- Configuration Sections
- Dependency Relationships
- Execution Notes
- Freeze Policy

# Part 2 — Modules 1–5

---

# Module 1 — Configuration System

## Overview

Module 1 establishes the configuration foundation for the entire framework.

Every module in the repository depends directly or indirectly on this module.

It provides centralized configuration loading, validation, nested configuration access, runtime environment configuration, and project-wide exception handling.

No module is permitted to access YAML files directly.

---

## Status

| Property | Value |
|----------|-------|
| Module | 1 |
| Status | APPROVED |
| Completion Date | 27 June 2026 |
| Coverage | ~85% |
| Package | `src/core/` |

---

## Primary Purpose

Provide a single immutable configuration object shared across the entire framework.

---

## Features Implemented

- YAML Configuration Loader
- Recursive Configuration Object
- Configuration Validation
- Required Section Validation
- Nested Attribute Access
- Runtime Configuration
- Environment Variable Support
- Logging Configuration
- Custom Exception Hierarchy
- Configuration Serialization

---

## Public Interfaces

```
Config
load_config()
```

---

## Input

```
config/config.yaml
```

---

## Output Contract

```
Config
```

---

## Consumed By

Modules 2–20

---

## Source Files

```
src/core/

config.py
environment.py
exceptions.py
```

---

## Test Files

```
tests/

test_config.py
test_environment.py
```

---

## Architecture Decisions

- Single configuration source.
- Immutable after loading.
- Strong validation before execution.
- No hardcoded values.
- Fail-fast configuration errors.

---

## Frozen Public API

- Config
- load_config()

---

# Module 2 — Bootstrap & Directory Management

---

## Overview

Module 2 prepares the runtime environment before any processing begins.

It initializes logging, validates the execution environment, creates the project directory structure, and bootstraps the framework.

Every execution begins here.

---

## Status

| Property | Value |
|----------|-------|
| Module | 2 |
| Status | APPROVED |
| Completion Date | 28 June 2026 |

---

## Features Implemented

- Project Bootstrap
- Directory Creation
- Logging Initialization
- Runtime Validation
- Output Directory Management
- Temporary Directory Management

---

## Public Interfaces

```
Bootstrap
DirectoryManager
```

---

## Input

```
Config
```

---

## Output

```
Initialized runtime environment
```

---

## Consumed By

Module 20

---

## Source Files

```
src/core/

bootstrap.py
directories.py
```

---

## Test Files

```
tests/

test_bootstrap.py
test_directories.py
```

---

## Architecture Decisions

- Bootstrap executes only once.
- Directory creation is idempotent.
- Logging initialized before all modules.

---

# Module 3 — Google Earth Engine Client

---

## Overview

Module 3 encapsulates every interaction with Google Earth Engine.

No other package imports or initializes the Earth Engine API directly.

All authentication, initialization, retry logic, and health checks are routed through a single client abstraction. :contentReference[oaicite:1]{index=1}

---

## Status

| Property | Value |
|----------|-------|
| Module | 3 |
| Status | APPROVED |
| Completion Date | 28 June 2026 |
| Coverage | >90% |

---

## Features Implemented

- EarthEngineClient
- Authentication Manager
- Runtime Detection (Local / Colab)
- Health Checker
- Structured Health Reports
- Retry Configuration
- Exponential Backoff
- Transient Error Detection
- Custom GEE Exception Hierarchy

---

## Primary Public Interface

```
EarthEngineClient
```

---

## Internal Components

```
EarthEngineClient

├── AuthManager
├── HealthChecker
├── RetryConfig
└── Transient Error Classifier
```

---

## Input

```
Config
```

---

## Output

```
Authenticated Earth Engine Session
```

---

## Source Files

```
src/gee/

client.py
auth.py
health.py
__init__.py
```

---

## Test Files

```
tests/gee/

test_client.py
test_auth.py
test_health.py
```

---

## Architecture Decisions

- Earth Engine imported only inside src/gee.
- Authentication centralized.
- Automatic retry for transient failures.
- Structured health reporting.

---

## Frozen Public API

```
EarthEngineClient
```

---

# Module 4 — Landsat Collection Builder

---

## Overview

Module 4 constructs Landsat image collections from Google Earth Engine.

It is responsible for sensor selection, AOI filtering, temporal filtering, cloud filtering, and metadata extraction. The builder produces a typed `CollectionResult` contract for downstream preprocessing. :contentReference[oaicite:2]{index=2}

---

## Status

| Property | Value |
|----------|-------|
| Module | 4 |
| Status | APPROVED |
| Completion Date | 29 June 2026 |
| Coverage | >90% |

---

## Features Implemented

- LandsatCollectionBuilder
- Automatic Sensor Selection
- Manual Sensor Selection
- AOI Filtering
- Date Filtering
- Cloud Cover Filtering
- Metadata Extraction
- Collection Statistics
- Temporal Coverage Analysis
- CRS and Scale Extraction

---

## Public Interfaces

```
LandsatCollectionBuilder
MetadataExtractor
```

---

## Internal Flow

```
LandsatCollectionBuilder

↓

Resolve Sensors

↓

Build Collection

↓

Apply Filters

↓

Metadata Extraction

↓

CollectionResult
```

---

## Output Contract

```
CollectionResult
```

---

## Source Files

```
src/gee/

collections.py
filters.py
metadata.py
```

---

## Test Files

```
tests/gee/

test_collections.py
test_filters.py
test_metadata.py
```

---

## Architecture Decisions

- Builder Pattern.
- Automatic Landsat sensor selection.
- Metadata separated from collection construction.
- Pure filtering functions.

---

## Frozen Public API

```
LandsatCollectionBuilder
CollectionResult
MetadataExtractor
```

---

# Module 5 — Landsat Image Preprocessing

---

## Overview

Module 5 converts raw Landsat collections into analysis-ready imagery.

The preprocessing pipeline performs radiometric scaling, QA masking, sensor harmonization, and temporal compositing before producing the processed collection contract consumed by feature engineering. :contentReference[oaicite:3]{index=3}

---

## Status

| Property | Value |
|----------|-------|
| Module | 5 |
| Status | APPROVED |
| Completion Date | 29 June 2026 |
| Coverage | >90% |

---

## Features Implemented

- LandsatPreprocessor
- Surface Reflectance Scaling
- Thermal Band Scaling
- QA Pixel Masking
- Cloud Masking
- Shadow Masking
- Snow Masking
- Water Masking
- Band Harmonization
- Temporal Compositing
- Median Composite
- Mean Composite
- Mosaic Composite
- Medoid Composite
- Percentile Composite

---

## Public Interfaces

```
LandsatPreprocessor
LandsatCompositor
```

---

## Internal Pipeline

```
CollectionResult

↓

Scaling

↓

QA Masking

↓

Band Harmonization

↓

ProcessedCollectionResult

↓

Temporal Composite

↓

CompositeResult
```

---

## Output Contracts

```
ProcessedCollectionResult

CompositeResult
```

---

## Source Files

```
src/gee/

preprocessing.py
masking.py
harmonization.py
composite.py
```

---

## Test Files

```
tests/gee/

test_preprocessing.py
test_masking.py
test_harmonization.py
test_composite.py
```

---

## Architecture Decisions

- Server-side processing in Earth Engine.
- Configurable QA masking.
- Independent compositing strategies.
- Harmonization separated from preprocessing.
- Fully deterministic processing.

---

## Frozen Public API

```
LandsatPreprocessor

LandsatCompositor

ProcessedCollectionResult
```

---

**End of Part 2**

The next section documents Modules 6–10:

- Module 6 — Spectral Feature Engineering
- Module 7 — GeoTIFF Export & Dataset Generation
- Module 8 — Patch Generation Pipeline
- Module 9 — Automatic Pseudo Label Generation
- Module 10 — Dataset Assembly & Quality Control

# Part 3 — Modules 6–10

---

# Module 6 — Spectral Feature Engineering

## Overview

Module 6 transforms the preprocessed Landsat composite into a rich multi-band feature stack suitable for machine learning.

Rather than exposing only the original spectral bands, this module computes a configurable set of spectral indices and assembles them into a unified feature representation.

The resulting feature stack becomes the single source of input features for dataset export.

---

## Status

| Property | Value |
|----------|-------|
| Module | 6 |
| Status | APPROVED |
| Completion Date | 29 June 2026 |
| Coverage | >90% |
| Package | `src/gee/` |

---

## Purpose

Generate a machine-learning-ready feature stack by combining original Landsat bands with derived spectral indices.

---

## Features Implemented

- Spectral Feature Registry
- Feature Stack Builder
- Band Selection
- NDVI
- NDWI
- MNDWI
- SAVI
- NDMI
- NDBI
- BSI
- Custom Feature Registration
- Feature Validation
- Dynamic Feature Selection
- Typed FeatureStackResult

---

## Public Interfaces

```
FeatureRegistry

FeatureStackBuilder

FeatureStackResult
```

---

## Input Contract

```
ProcessedCollectionResult
```

---

## Output Contract

```
FeatureStackResult
```

---

## Source Files

```
src/gee/

features.py
registry.py
indices.py
```

---

## Test Files

```
tests/gee/

test_features.py
test_registry.py
test_indices.py
```

---

## Architecture Decisions

- Registry Pattern for spectral indices.
- Configurable feature selection.
- New indices can be added without modifying existing code.
- Original spectral bands remain available.
- Typed immutable output contract.

---

## Frozen Public API

```
FeatureRegistry

FeatureStackBuilder

FeatureStackResult
```

---

# Module 7 — GeoTIFF Export & Dataset Generation

---

## Overview

Module 7 converts Earth Engine feature stacks into persistent GeoTIFF datasets.

This module represents the transition from cloud-based processing to local machine learning.

It downloads imagery, validates GeoTIFFs, generates metadata, maintains manifests, and versions exported datasets. :contentReference[oaicite:0]{index=0}

---

## Status

| Property | Value |
|----------|-------|
| Module | 7 |
| Status | APPROVED |
| Completion Date | 30 June 2026 |
| Coverage | ~80% |

---

## Purpose

Export FeatureStackResult into reproducible GeoTIFF datasets.

---

## Features Implemented

- DatasetExporter
- EarthEngineDownloader
- GeoTiffWriter
- GeoTiffValidator
- MetadataWriter
- DatasetManifestManager
- DatasetVersionManager
- Scene Metadata
- Dataset Versioning
- Export Validation

---

## Public Interfaces

```
DatasetExporter
```

---

## Internal Pipeline

```
FeatureStackResult

↓

EarthEngineDownloader

↓

GeoTiffWriter

↓

MetadataWriter

↓

GeoTiffValidator

↓

DatasetManifestManager

↓

DatasetVersionManager

↓

DatasetExportResult
```

---

## Input Contract

```
FeatureStackResult
```

---

## Output Contract

```
DatasetExportResult
```

---

## Output Directory

```
output/

version.json

manifest.csv

manifest.json

scenes/

scene_id/

image.tif

metadata.json
```

---

## Source Files

```
src/export/

exporter.py
downloader.py
geotiff.py
metadata.py
manifest.py
validator.py
version.py
```

---

## Test Files

```
tests/export/

test_exporter.py
test_downloader.py
test_geotiff.py
test_metadata.py
test_manifest.py
test_validator.py
test_version.py
```

---

## Architecture Decisions

- Earth Engine separated from GeoTIFF writing.
- Export validation mandatory.
- Manifest generation isolated.
- Dataset versioning ensures reproducibility.
- GeoTIFF writing centralized.

---

## Frozen Public API

```
DatasetExporter

DatasetExportResult
```

---

# Module 8 — Patch Generation Pipeline

---

## Overview

Module 8 converts exported GeoTIFF scenes into fixed-size image patches suitable for deep learning.

This module isolates all raster tiling logic from downstream machine learning modules. :contentReference[oaicite:1]{index=1}

---

## Status

| Property | Value |
|----------|-------|
| Module | 8 |
| Status | APPROVED |
| Completion Date | 30 June 2026 |
| Coverage | >90% |

---

## Purpose

Generate deterministic image patches for model training.

---

## Features Implemented

- PatchGenerator
- PatchTiler
- PatchReader
- PatchValidator
- PatchManifestManager
- Sliding Window Generation
- Overlap Support
- NoData Filtering
- Patch Metadata
- Patch Manifest

---

## Public Interfaces

```
PatchGenerator
```

---

## Internal Pipeline

```
DatasetExportResult

↓

PatchTiler

↓

PatchReader

↓

PatchValidator

↓

GeoTiffWriter

↓

PatchManifestManager

↓

PatchDatasetResult
```

---

## Input Contract

```
DatasetExportResult
```

---

## Output Contract

```
PatchDatasetResult
```

---

## Source Files

```
src/patches/

generator.py
tiler.py
reader.py
validator.py
manifest.py
```

---

## Test Files

```
tests/patches/

test_generator.py
test_tiler.py
test_reader.py
test_validator.py
test_manifest.py
```

---

## Architecture Decisions

- Deterministic patch generation.
- Validation before saving.
- Manifest generated automatically.
- GeoTIFF writer reused from Module 7.
- Patch size fully configuration-driven.

---

## Frozen Public API

```
PatchGenerator

PatchDatasetResult
```

---

# Module 9 — Automatic Pseudo Label Generation

---

## Overview

Module 9 automatically generates semantic segmentation labels directly from spectral information.

Unlike traditional workflows, no manually digitized masks are required.

The module uses a configurable evidence-based classification framework followed by conflict resolution, morphology refinement, confidence estimation, and quality assessment before producing the unchanged LabelDatasetResult contract. :contentReference[oaicite:2]{index=2}

---

## Status

| Property | Value |
|----------|-------|
| Module | 9 |
| Status | APPROVED |
| Completion Date | 1 July 2026 |
| Coverage | >90% |

---

## Purpose

Generate reproducible pseudo-labels without human annotation.

---

## Features Implemented

- LabelManager
- PseudoLabelGenerator
- SpectralClassificationEngine
- RuleRegistry
- WaterRule
- SandRule
- VegetationRule
- BackgroundRule
- ConflictResolver
- MorphologyProcessor
- QualityAssessment
- ConfidenceEstimator
- ReproducibilityMetadata
- Evidence-Based Classification
- Plugin Rule Architecture
- Confidence Maps
- Label Statistics
- Manifest Generation

---

## Public Interfaces

```
LabelManager

LabelDatasetResult
```

---

## Internal Pipeline

```
PatchDatasetResult

↓

SpectralClassificationEngine

↓

RuleRegistry

↓

ConflictResolver

↓

MorphologyProcessor

↓

QualityAssessment

↓

ConfidenceEstimator

↓

PseudoLabelResult

↓

LabelDatasetResult
```

---

## Input Contract

```
PatchDatasetResult
```

---

## Output Contract

```
LabelDatasetResult
```

---

## Source Files

```
src/labels/

manager.py
generator.py
classifier.py
rules.py
conflicts.py
morphology.py
quality.py
confidence.py
statistics.py
manifest.py
validator.py
contracts.py
schema.py
temporal.py
```

---

## Test Files

```
tests/labels/

test_label_manager.py
test_pseudo_label_generator.py
test_label_classifier.py
test_label_rules.py
test_label_conflicts.py
test_label_morphology.py
test_label_quality.py
test_label_confidence.py
test_label_strategy.py
```

---

## Architecture Decisions

- Plugin-based rule engine.
- Evidence-based spectral scoring.
- Automatic conflict resolution.
- Confidence-weighted labels.
- Backward-compatible LabelDatasetResult.
- No human annotation required.

---

## Frozen Public API

```
LabelManager

LabelDatasetResult
```

---

# Module 10 — Dataset Assembly & Quality Control

---

## Overview

Module 10 assembles the final machine learning dataset by combining image patches with pseudo-labels.

It performs validation, dataset splitting, leakage detection, quality analysis, statistics generation, manifest creation, and versioning before producing the training dataset contract. :contentReference[oaicite:3]{index=3}

---

## Status

| Property | Value |
|----------|-------|
| Module | 10 |
| Status | APPROVED |
| Completion Date | 3 July 2026 |
| Coverage | >90% |

---

## Purpose

Produce deterministic train, validation, and test datasets ready for PyTorch.

---

## Features Implemented

- DatasetAssembler
- DatasetSplitter
- DatasetValidator
- DataLeakageDetector
- DatasetStatisticsCalculator
- DatasetQualityAnalyzer
- DatasetManifestManager
- DatasetVersionManager
- Random Split
- Temporal Split
- Spatial Split
- Scene-Level Splitting
- Leakage Detection
- Quality Reports
- Dataset Statistics
- Manifest Generation

---

## Public Interfaces

```
DatasetAssembler

TrainingDatasetResult
```

---

## Internal Pipeline

```
PatchDatasetResult

+

LabelDatasetResult

↓

DatasetValidator

↓

DatasetSplitter

↓

LeakageDetector

↓

Statistics

↓

QualityAnalyzer

↓

Manifest

↓

Version

↓

TrainingDatasetResult
```

---

## Input Contracts

```
PatchDatasetResult

LabelDatasetResult
```

---

## Output Contract

```
TrainingDatasetResult
```

---

## Source Files

```
src/dataset/

assembler.py
splitter.py
validator.py
leakage.py
statistics.py
quality.py
manifest.py
version.py
```

---

## Test Files

```
tests/dataset/

test_dataset_assembler.py
test_dataset_splitter.py
test_dataset_validator.py
test_dataset_leakage.py
test_dataset_statistics.py
test_dataset_quality.py
test_dataset_manifest.py
test_dataset_version.py
```

---

## Architecture Decisions

- Scene-level leakage prevention.
- Configurable split strategies.
- Validation before assembly.
- Modular statistics generation.
- Fully reproducible datasets.

---

## Frozen Public API

```
DatasetAssembler

TrainingDatasetResult
```

---

**End of Part 3**

The next section (Part 4) documents Modules 11–15:

- Module 11 — PyTorch Dataset & DataLoader
- Module 12 — Data Transformation & Augmentation
- Module 13 — Segmentation Model Zoo (UNet++)
- Module 14 — Training Engine
- Module 15 — Evaluation Framework

# Part 4 — Modules 11–15

---

# Module 11 — PyTorch Dataset & DataLoader Pipeline

## Overview

Module 11 bridges the gap between the assembled machine learning dataset and the PyTorch training pipeline.

It converts the dataset generated by Module 10 into efficient PyTorch DataLoaders while handling normalization, lazy loading, batching, sampling, and GPU-ready tensor preparation.

This module is the only component allowed to interact directly with PyTorch DataLoader objects.

---

## Status

| Property | Value |
|----------|-------|
| Module | 11 |
| Status | APPROVED |
| Completion Date | 3 July 2026 |
| Coverage | >95% |
| Package | `src/training/` |

---

## Purpose

Create optimized DataLoaders for semantic segmentation training.

---

## Features Implemented

- RiverMorphologyDataset
- DataLoaderFactory
- DatasetNormalizer
- AugmentationPipeline Interface
- TemporalSampler
- Class Weight Computation
- Lazy Raster Loading
- Multi-worker DataLoader
- GPU-ready Dataset Pipeline
- Batch Generation
- Dataset Metadata Support

---

## Public Interfaces

```
DataLoaderFactory

RiverMorphologyDataset

DataLoaderBundle
```

---

## Internal Pipeline

```
TrainingDatasetResult

↓

DatasetNormalizer

↓

RiverMorphologyDataset

↓

TemporalSampler

↓

DataLoaderFactory

↓

DataLoaderBundle
```

---

## Input Contract

```
TrainingDatasetResult
```

---

## Output Contract

```
DataLoaderBundle
```

---

## Source Files

```
src/training/

dataset.py
dataloader.py
normalizer.py
sampler.py
transforms.py
weights.py
```

---

## Test Files

```
tests/training/

test_dataset.py
test_dataloader.py
test_normalizer.py
test_sampler.py
test_transforms.py
test_weights.py
```

---

## Architecture Decisions

- Lazy loading minimizes memory usage.
- Rasterio performs patch reading.
- Dataset independent of augmentation.
- Normalization computed from training data only.
- DataLoader construction centralized.

---

## Frozen Public API

```
RiverMorphologyDataset

DataLoaderFactory

DataLoaderBundle
```

---

# Module 12 — Data Transformation & Augmentation Pipeline

---

## Overview

Module 12 performs deterministic data augmentation and normalization for semantic segmentation.

All transformations operate jointly on image and mask, ensuring spatial correspondence is preserved.

The module supports arbitrary multispectral imagery and is fully configuration-driven.

---

## Status

| Property | Value |
|----------|-------|
| Module | 12 |
| Status | APPROVED |
| Completion Date | 5 July 2026 |
| Coverage | >90% |
| Package | `src/training/` |

---

## Purpose

Generate reproducible augmented datasets for model training.

---

## Features Implemented

- TransformPipeline
- TransformRegistry
- TransformFactory
- TransformPipelineResult
- BaseTransform
- ComposeTransform
- HorizontalFlipTransform
- VerticalFlipTransform
- Rotate90Transform
- BrightnessTransform
- ContrastTransform
- GaussianNoiseTransform
- DatasetNormalizer
- NormalizationStatistics
- StatisticsCalculator
- DatasetValidator
- Multi-band Support
- Mask-safe Geometric Augmentation
- Deterministic Random Operations
- Configuration-driven Pipeline

---

## Public Interfaces

```
TransformPipeline

TransformRegistry

TransformPipelineResult
```

---

## Internal Pipeline

```
DataLoaderBundle

↓

TransformFactory

↓

TransformPipeline

↓

Normalization

↓

Validation

↓

TransformPipelineResult
```

---

## Input Contract

```
DataLoaderBundle
```

---

## Output Contract

```
TransformPipelineResult
```

---

## Source Files

```
src/training/

augmentation.py
contracts.py
normalization.py
pipeline.py
registry.py
statistics.py
transform.py
validator.py
```

---

## Test Files

```
tests/training/

test_training_augmentation.py
test_training_contracts.py
test_training_normalization.py
test_training_pipeline.py
test_training_registry.py
test_training_statistics.py
test_training_transform.py
test_training_validator.py
```

---

## Architecture Decisions

- Image and mask transformed simultaneously.
- Random operations reproducible.
- Augmentations configurable.
- Independent normalization.
- Registry allows future augmentation plugins.

---

## Frozen Public API

```
TransformPipeline

TransformPipelineResult
```

---

# Module 13 — Segmentation Model Zoo

---

## Overview

Module 13 defines the deep learning model architecture layer.

It provides a registry-based model factory capable of constructing semantic segmentation networks while maintaining a common interface for the training engine.

The primary production architecture is UNet++.

---

## Status

| Property | Value |
|----------|-------|
| Module | 13 |
| Status | APPROVED |
| Completion Date | 6 July 2026 |
| Coverage | >90% |
| Package | `src/training/models/` |

---

## Purpose

Construct semantic segmentation models independent of the training engine.

---

## Features Implemented

- ModelFactory
- ModelRegistry
- BaseSegmentationModel
- UNetEncoder
- UNetPlusPlusDecoder
- UNetPlusPlus
- Deep Supervision
- Weight Initialization
- Deterministic Initialization
- Configurable Encoder Depth
- Configurable Input Channels
- Configurable Output Classes

---

## Public Interfaces

```
ModelFactory

ModelRegistry

ModelResult
```

---

## Internal Pipeline

```
TransformPipelineResult

↓

ModelFactory

↓

ModelRegistry

↓

UNet++

↓

ModelResult
```

---

## Input Contract

```
TransformPipelineResult
```

---

## Output Contract

```
ModelResult
```

---

## Source Files

```
src/training/models/

base.py
factory.py
registry.py
encoder.py
decoder.py
unetplusplus.py
initialization.py
contracts.py
```

---

## Test Files

```
tests/training/models/

test_model_factory.py
test_registry.py
test_encoder.py
test_decoder.py
test_unetplusplus.py
test_initialization.py
```

---

## Architecture Decisions

- Registry Pattern.
- Factory Pattern.
- Backbone independent.
- Deterministic initialization.
- Deep supervision optional.
- Future models can be added without modifying Trainer.

---

## Frozen Public API

```
ModelFactory

ModelResult
```

---

# Module 14 — Training Engine

---

## Overview

Module 14 performs supervised semantic segmentation training.

It coordinates optimization, learning-rate scheduling, checkpointing, loss computation, mixed precision training, and training history.

This module owns the entire optimization process.

---

## Status

| Property | Value |
|----------|-------|
| Module | 14 |
| Status | APPROVED |
| Completion Date | 7 July 2026 |
| Coverage | >90% |
| Package | `src/training/trainer/` |

---

## Purpose

Train semantic segmentation models.

---

## Features Implemented

- TrainingEngine
- Trainer
- EpochRunner
- Mixed Precision Training
- CheckpointManager
- EarlyStopping
- OptimizerFactory
- SchedulerFactory
- LossFactory
- CombinedLoss
- CrossEntropyLoss
- DiceLoss
- Training History
- Gradient Clipping
- Deterministic Training

---

## Public Interfaces

```
TrainingEngine

TrainingResult
```

---

## Internal Pipeline

```
ModelResult

+

TransformPipelineResult

↓

Trainer

↓

Optimizer

↓

Loss

↓

Scheduler

↓

CheckpointManager

↓

TrainingResult
```

---

## Input Contracts

```
ModelResult

TransformPipelineResult
```

---

## Output Contract

```
TrainingResult
```

---

## Source Files

```
src/training/trainer/

engine.py
trainer.py
checkpoint.py
loss.py
optimizer.py
scheduler.py
history.py
```

---

## Test Files

```
tests/training/trainer/

test_engine.py
test_checkpoint.py
test_losses.py
test_optimizer.py
test_scheduler.py
test_history.py
```

---

## Architecture Decisions

- Engine owns training lifecycle.
- Loss functions modular.
- Checkpointing isolated.
- Mixed precision optional.
- Deterministic execution supported.

---

## Frozen Public API

```
TrainingEngine

TrainingResult
```

---

# Module 15 — Evaluation Framework

---

## Overview

Module 15 evaluates trained semantic segmentation models using standard computer vision metrics.

It operates on the held-out test dataset generated by Module 10 and produces quantitative performance reports for each trained checkpoint.

---

## Status

| Property | Value |
|----------|-------|
| Module | 15 |
| Status | APPROVED |
| Completion Date | 8 July 2026 |
| Coverage | >90% |
| Package | `src/training/evaluation/` |

---

## Purpose

Measure model performance.

---

## Features Implemented

- EvaluationEngine
- Evaluator
- MetricRegistry
- IoU
- Dice Score
- Precision
- Recall
- F1 Score
- Pixel Accuracy
- Mean IoU
- Confusion Matrix
- Per-class Metrics
- Metrics Export
- Evaluation Summary

---

## Public Interfaces

```
EvaluationEngine

EvaluationResult
```

---

## Internal Pipeline

```
TrainingResult

↓

Evaluator

↓

Metrics

↓

Confusion Matrix

↓

EvaluationResult
```

---

## Input Contract

```
TrainingResult
```

---

## Output Contract

```
EvaluationResult
```

---

## Source Files

```
src/training/evaluation/

engine.py
evaluator.py
metrics.py
confusion.py
registry.py
summary.py
```

---

## Test Files

```
tests/training/evaluation/

test_eval_engine.py
test_eval_evaluator.py
test_metrics.py
test_confusion.py
test_registry.py
```

---

## Architecture Decisions

- Metrics are plugin-based.
- Evaluation independent of training.
- Per-class statistics generated automatically.
- Results exportable.
- Supports future metrics.

---

## Frozen Public API

```
EvaluationEngine

EvaluationResult
```

---

**End of Part 4**

The next section (Part 5) documents Modules 16–20:

- Module 16 — Inference Pipeline
- Module 17 — River Morphology Analytics
- Module 18 — Visualization Framework
- Module 19 — Reporting Framework
- Module 20 — End-to-End Pipeline Orchestration

# Part 5 — Modules 16–20

---

# Module 16 — Inference Pipeline

## Overview

Module 16 performs inference using a trained semantic segmentation model.

It restores a trained checkpoint, loads unseen image patches, performs forward inference through the segmentation model, applies post-processing, computes confidence maps, and exports prediction artifacts.

Unlike Module 14, this module never updates model weights.

---

## Status

| Property | Value |
|----------|-------|
| Module | 16 |
| Status | APPROVED |
| Completion Date | 9 July 2026 |
| Coverage | >95% |
| Package | `src/training/inference/` |

---

## Purpose

Generate semantic segmentation predictions on unseen imagery.

---

## Features Implemented

- InferenceEngine
- InferenceFactory
- CheckpointLoader
- Predictor
- PredictionExporter
- PostprocessorPipeline
- InferenceValidator
- Batch Inference
- Confidence Map Generation
- Softmax Probability Maps
- Prediction Export
- GeoTIFF Export
- PNG Export
- NumPy Export
- Deterministic Inference
- Multi-device Support

---

## Public Interfaces

```
InferenceEngine

InferenceResult
```

---

## Internal Pipeline

```
TrainingResult
        │
        ▼
CheckpointLoader
        │
        ▼
Predictor
        │
        ▼
PostprocessorPipeline
        │
        ▼
PredictionExporter
        │
        ▼
InferenceResult
```

---

## Input Contract

```
TrainingResult
```

---

## Output Contract

```
InferenceResult
```

---

## Source Files

```
src/training/inference/

engine.py
factory.py
loader.py
predictor.py
postprocessing.py
exporter.py
validator.py
contracts.py
```

---

## Test Files

```
tests/training/inference/

test_inference_engine.py
test_inference_factory.py
test_inference_loader.py
test_inference_predictor.py
test_inference_postprocessing.py
test_inference_exporter.py
test_inference_validator.py
```

---

## Architecture Decisions

- Inference independent from training.
- Checkpoints restored through a dedicated loader.
- Prediction export separated from inference.
- Confidence maps preserved.
- Supports CPU and GPU execution.
- Fully deterministic inference.

---

## Frozen Public API

```
InferenceEngine

InferenceResult
```

---

# Module 17 — River Morphology Analytics

---

## Overview

Module 17 transforms semantic segmentation predictions into quantitative river morphology measurements.

This module performs connected-component analysis, geometric measurements, temporal statistics, uncertainty analysis, and morphology metrics.

It converts raster predictions into scientifically meaningful indicators.

---

## Status

| Property | Value |
|----------|-------|
| Module | 17 |
| Status | APPROVED |
| Completion Date | 10 July 2026 |
| Coverage | >90% |
| Package | `src/morphology/` |

---

## Purpose

Extract river morphology metrics from predicted segmentation maps.

---

## Features Implemented

- MorphologyAnalyzer
- GeometryAnalyzer
- MorphologyStatisticsComputer
- UncertaintyAnalyzer
- TemporalAnalyzer
- ConnectedRegionStats
- ClassRegionMetrics
- GeometryMetrics
- Shape Descriptor Computation
- Confidence-weighted Area
- Fragmentation Analysis
- Region Statistics
- Per-class Statistics
- Pixel Resolution Support
- Confidence Propagation

---

## Public Interfaces

```
MorphologyAnalyzer

RiverMorphologyResult
```

---

## Internal Pipeline

```
InferenceResult
        │
        ▼
GeometryAnalyzer
        │
        ▼
StatisticsComputer
        │
        ▼
TemporalAnalyzer
        │
        ▼
UncertaintyAnalyzer
        │
        ▼
RiverMorphologyResult
```

---

## Input Contract

```
InferenceResult
```

---

## Output Contract

```
RiverMorphologyResult
```

---

## Source Files

```
src/morphology/

contracts.py
geometry.py
statistics.py
analyzer.py
uncertainty.py
temporal.py
validator.py
factory.py
engine.py
```

---

## Test Files

```
tests/morphology/

test_morphology_contracts.py
test_morphology_geometry.py
test_morphology_statistics.py
test_morphology_engine.py
```

---

## Architecture Decisions

- Geometry independent from statistics.
- Connected-component analysis per class.
- Confidence-aware measurements.
- Shape descriptors optional.
- Resolution-aware metrics.

---

## Frozen Public API

```
MorphologyAnalyzer

RiverMorphologyResult
```

---

# Module 18 — Visualization Framework

---

## Overview

Module 18 converts morphology and inference outputs into publication-quality figures.

It supports segmentation rendering, overlays, confidence visualization, comparison plots, timelines, and figure export.

---

## Status

| Property | Value |
|----------|-------|
| Module | 18 |
| Status | APPROVED |
| Completion Date | 11 July 2026 |
| Coverage | >95% |
| Package | `src/visualization/` |

---

## Purpose

Generate scientific visualization products.

---

## Features Implemented

- VisualizationEngine
- MaskRenderer
- OverlayRenderer
- TimelineRenderer
- ComparisonRenderer
- FigureExporter
- ColorRegistry
- VisualizationValidator
- PNG Export
- PDF Export
- SVG Export
- Confidence Overlay
- Temporal Timeline
- Side-by-side Comparison

---

## Public Interfaces

```
VisualizationEngine

VisualizationResult
```

---

## Input Contract

```
RiverMorphologyResult
```

---

## Output Contract

```
VisualizationResult
```

---

## Source Files

```
src/visualization/

contracts.py
renderer.py
overlay.py
timeline.py
comparison.py
colormap.py
exporter.py
validator.py
factory.py
engine.py
```

---

## Test Files

```
tests/visualization/

test_visualization_contracts.py
test_visualization_renderer.py
test_visualization_exporter.py
test_visualization_engine.py
```

---

## Architecture Decisions

- Matplotlib backend.
- Configurable color maps.
- High-resolution exports.
- Headless rendering.
- Independent figure exporters.

---

## Frozen Public API

```
VisualizationEngine

VisualizationResult
```

---

# Module 19 — Reporting Framework

---

## Overview

Module 19 aggregates outputs from previous modules into a comprehensive experiment report.

It records experiment metadata, manages generated artifacts, creates manifests, and exports reports in multiple formats.

---

## Status

| Property | Value |
|----------|-------|
| Module | 19 |
| Status | APPROVED |
| Completion Date | 12 July 2026 |
| Coverage | >90% |
| Package | `src/reporting/` |

---

## Purpose

Generate reproducible experiment reports.

---

## Features Implemented

- ReportEngine
- ReportGenerator
- ReportExporter
- ManifestManager
- ArtifactManager
- ExperimentManager
- ReportValidator
- Markdown Report
- JSON Report
- CSV Report
- PDF Export
- Artifact Inventory
- Experiment Metadata

---

## Public Interfaces

```
ReportEngine

ReportResult
```

---

## Internal Pipeline

```
EvaluationResult
        │
InferenceResult
        │
RiverMorphologyResult
        │
VisualizationResult
        │
        ▼
ReportGenerator
        │
        ▼
ReportExporter
        │
        ▼
ReportResult
```

---

## Input Contracts

```
EvaluationResult

InferenceResult

RiverMorphologyResult

VisualizationResult
```

---

## Output Contract

```
ReportResult
```

---

## Source Files

```
src/reporting/

contracts.py
artifact.py
experiment.py
manifest.py
report.py
exporter.py
validator.py
factory.py
engine.py
```

---

## Test Files

```
tests/reporting/

test_reporting_engine.py
test_reporting_exporter.py
test_reporting_manifest.py
test_reporting_validator.py
```

---

## Architecture Decisions

- Reporting isolated from visualization.
- Experiment metadata centralized.
- Artifact tracking automatic.
- Multiple export formats.
- Reproducible reports.

---

## Frozen Public API

```
ReportEngine

ReportResult
```

---

# Module 20 — End-to-End Pipeline Orchestration

---

## Overview

Module 20 integrates every preceding module into a single executable pipeline.

It provides the command-line interface (CLI), validates configuration, orchestrates execution order, manages per-AOI execution state, and coordinates the end-to-end workflow from satellite imagery acquisition to final reporting.

---

## Status

| Property | Value |
|----------|-------|
| Module | 20 |
| Status | APPROVED |
| Completion Date | 13 July 2026 |
| Coverage | >90% |
| Package | `src/pipeline/` |

---

## Purpose

Provide a unified execution layer for the entire framework.

---

## Features Implemented

- PipelineRunner
- PipelineFactory
- PipelineOrchestrator
- StageRunner
- CLI Interface
- Multi-AOI Execution
- Stage State Management
- Configuration Validation
- Dry-run Mode
- Logging Integration
- Failure Recovery
- PipelineResult
- Mode Selection (full, training, evaluation, inference, analysis, visualization, reporting)

---

## Public Interfaces

```
PipelineRunner
PipelineOrchestrator
run_cli()
PipelineResult
```

---

## Complete Execution Flow

```
Config
   │
Bootstrap
   │
Earth Engine
   │
Collection
   │
Preprocessing
   │
Feature Engineering
   │
GeoTIFF Export
   │
Patch Generation
   │
Pseudo Labels
   │
Dataset Assembly
   │
PyTorch Dataset
   │
Augmentation
   │
UNet++
   │
Training
   │
Evaluation
   │
Inference
   │
Morphology
   │
Visualization
   │
Reporting
   │
PipelineResult
```

---

## Input Contract

```
Config
```

---

## Output Contract

```
PipelineResult
```

---

## Source Files

```
src/pipeline/

cli.py
contracts.py
factory.py
orchestrator.py
runner.py
validator.py
```

---

## Test Files

```
tests/pipeline/

test_cli.py
test_factory.py
test_orchestrator.py
test_runner.py
test_validator.py
```

---

## Architecture Decisions

- Single entry point for execution.
- Stage-based orchestration.
- Per-AOI state propagation.
- Configuration-driven execution.
- Supports full and partial pipeline modes.
- No module bypasses the orchestrator.
- Public contracts preserved across all stages.

---

## Frozen Public API

```
PipelineRunner
PipelineOrchestrator
run_cli()
PipelineResult
```

---

# End of PROJECT_MODULE_INDEX.md

## Repository Summary

- Total Modules: 20
- Architecture: Modular, Contract-Driven
- Public Contracts: Immutable
- Configuration: YAML-Driven
- Deep Learning Framework: PyTorch
- Remote Sensing Platform: Google Earth Engine
- Segmentation Model: UNet++
- End-to-End Pipeline: Fully Integrated
- Primary Entry Point: `main.py`
- CLI Entry: `python main.py --mode <mode>`
- Production Status: Ready for Deployment (subject to runtime configuration and data availability)

# Part 6 — Operations, Deployment & Maintenance Guide

---

# 20. Complete Runtime Workflow

The following diagram illustrates the complete execution flow of the framework.

```

User

│

▼

Edit config/config.yaml

│

▼

python main.py --mode full

│

▼

Bootstrap Environment

│

▼

Load Configuration

│

▼

Initialize Google Earth Engine

│

▼

Build Landsat Collection

│

▼

Preprocess Images

│

▼

Generate Spectral Features

│

▼

Export GeoTIFF Dataset

│

▼

Generate Image Patches

│

▼

Generate Pseudo Labels

│

▼

Assemble Dataset

│

▼

Build PyTorch Dataset

│

▼

Apply Data Augmentation

│

▼

Construct UNet++

│

▼

Train Model

│

▼

Evaluate Model

│

▼

Inference

│

▼

River Morphology Analytics

│

▼

Visualization

│

▼

Generate Reports

│

▼

PipelineResult

```

---

# 21. Runtime Inputs

The framework requires four categories of inputs.

## A. Configuration

```
config/config.yaml
```

Contains:

- AOIs
- Date ranges
- Landsat settings
- Patch generation
- Model configuration
- Training configuration
- Inference configuration
- Output paths

---

## B. Google Earth Engine

Requires

```
earthengine authenticate
```

before first execution.

---

## C. AOIs

Configured inside

```
config.yaml
```

Example

```yaml
aoi:

  regions:

    - name: River_A

      geometry: ...

    - name: River_B

      geometry: ...

    - name: River_C

      geometry: ...
```

Supports

- one AOI
- multiple AOIs

Module 20 executes them sequentially.

---

## D. Date Range

Configured inside

```
date_range:
```

Example

```yaml
date_range:

  start: "2018-01-01"

  end: "2018-12-31"
```

---

# 22. Runtime Outputs

The framework produces the following outputs.

```
outputs/

│

├── geotiffs/

├── patches/

├── labels/

├── dataset/

├── checkpoints/

├── predictions/

├── morphology/

├── visualization/

├── reports/

└── logs/

```

---

# 23. Model Training

Training begins after:

```
GeoTIFF Export

↓

Patch Generation

↓

Pseudo Labels

↓

Dataset Assembly

↓

PyTorch Dataset

↓

Augmentation

↓

UNet++

↓

Training
```

The Training Engine receives

```
ModelResult

+

TransformPipelineResult
```

and produces

```
TrainingResult
```

Training checkpoints are stored under

```
outputs/checkpoints/
```

or

```
checkpoints/
```

depending on config.

---

# 24. Evaluation

Evaluation consumes

```
TrainingResult
```

and produces

```
EvaluationResult
```

Metrics include

- IoU
- Dice
- Precision
- Recall
- F1
- Pixel Accuracy
- Confusion Matrix

Outputs stored under

```
outputs/evaluation/
```

---

# 25. Inference

Inference requires

```
trained checkpoint
```

and

```
new AOI
```

The workflow is

```
AOI

↓

Earth Engine

↓

Feature Stack

↓

Patch Generation

↓

Model Prediction

↓

Prediction Export
```

Outputs

```
Prediction Mask

Confidence Map

Probability Map

GeoTIFF

PNG

NumPy
```

---

# 26. Morphology

Consumes

```
InferenceResult
```

Produces

```
RiverMorphologyResult
```

Including

- area

- width

- fragmentation

- connected regions

- confidence-weighted statistics

---

# 27. Visualization

Consumes

```
RiverMorphologyResult
```

Produces

- overlays

- segmentation maps

- timelines

- comparison figures

- publication-quality graphics

---

# 28. Reporting

Consumes

```
EvaluationResult

InferenceResult

RiverMorphologyResult

VisualizationResult
```

Produces

```
Markdown

JSON

CSV

PDF
```

---

# 29. Execution Modes

Available CLI modes

```
python main.py --mode full

python main.py --mode training

python main.py --mode evaluation

python main.py --mode inference

python main.py --mode analysis

python main.py --mode visualization

python main.py --mode reporting
```

---

# 30. Configuration Ownership

| Section | Module |
|----------|---------|
| project | M1 |
| paths | M2 |
| gee | M3 |
| aoi | M4 |
| date_range | M4 |
| preprocessing | M5 |
| spectral_bands | M6 |
| export | M7 |
| patch_generation | M8 |
| label_generation | M9 |
| dataset | M10 |
| augmentation | M12 |
| model | M13 |
| loss | M14 |
| optimizer | M14 |
| scheduler | M14 |
| training | M14 |
| inference | M16 |
| morphology | M17 |
| visualization | M18 |
| reporting | M19 |

---

# 31. Public Contracts (Frozen)

The following contracts must never change.

```
Config

CollectionResult

ProcessedCollectionResult

FeatureStackResult

DatasetExportResult

PatchDatasetResult

LabelDatasetResult

TrainingDatasetResult

DataLoaderBundle

TransformPipelineResult

ModelResult

TrainingResult

EvaluationResult

InferenceResult

RiverMorphologyResult

VisualizationResult

ReportResult

PipelineResult
```

---

# 32. Module Freeze Policy

After approval,

modules are frozen.

Allowed

- bug fixes

- coverage improvements

- documentation

Not allowed

- public API changes

- contract changes

- architecture redesign

without full compatibility review.

---

# 33. Repository Statistics

| Item | Value |
|------|-------|
| Total Modules | 20 |
| Language | Python |
| Deep Learning | PyTorch |
| Remote Sensing | Google Earth Engine |
| Segmentation Model | UNet++ |
| Dataset | Landsat 8/9 Collection 2 |
| Architecture | Contract-Driven |
| Configuration | YAML |
| Testing | Pytest |
| Public APIs | Frozen |
| Production Status | Ready |

---

# 34. Future Extensions

The architecture supports future additions without breaking compatibility.

Potential extensions include:

- Additional segmentation models (DeepLabV3+, SegFormer, U-Net 3+)
- Sentinel-2 support
- PlanetScope support
- Multi-temporal sequence models
- Change detection workflows
- Web dashboard
- REST API
- Docker deployment
- Distributed training
- Cloud inference
- Interactive GIS integration

These extensions should be implemented as new modules or plugins while preserving existing public contracts.

---

# 35. Document Ownership

This document is the authoritative technical specification for the repository.

When auditing, extending, maintaining, or reviewing the project:

- Treat this document as the primary architectural reference.
- Validate implementation against the documented module interfaces.
- Report discrepancies rather than making assumptions.
- Update this document whenever an approved architectural change is introduced.

---

**End of PROJECT_MODULE_INDEX.md**

**Document Version:** 1.0.0

**Applies To:** River Morphology Segmentation Framework (Modules 1–20)

**Maintenance Status:** Living Architecture Document