# ARCHITECTURE.md

> River Morphology Segmentation
>
> Software Architecture Documentation
>
> Version: 1.0
>
> Repository Architecture Reference

---

# Table of Contents

1. Architectural Philosophy
2. System Overview
3. High-Level Architecture
4. Repository Organization
5. Package Architecture
6. Processing Pipeline
7. Data Flow
8. Core Design Principles

---

# 1. Architectural Philosophy

The River Morphology Segmentation repository is designed as a layered, modular, production-quality software system.

The primary objective is to ensure that every processing stage is isolated behind a stable public interface.

Modules communicate only through typed contracts.

No module should directly manipulate another module's internal implementation.

The architecture intentionally separates:

- configuration
- infrastructure
- remote sensing
- preprocessing
- feature engineering
- dataset generation
- machine learning
- analytics

into independent packages.

This allows future modules to evolve without breaking completed components.

---

# 2. High-Level System Overview

The repository is organized as a sequential processing pipeline.

```

Configuration

↓

Bootstrap

↓

Earth Engine

↓

Landsat Collection

↓

Preprocessing

↓

Composite Generation

↓

Feature Engineering

↓

GeoTIFF Export

↓

Patch Generation

↓

Label Generation

↓

Dataset Assembly

↓

PyTorch Dataset

↓

Training (Future)

↓

Evaluation (Future)

↓

Inference (Future)

↓

River Morphology Analytics (Future)

```

Every stage has clearly defined:

- inputs
- outputs
- responsibilities
- contracts

No stage bypasses another.

---

# 3. Architectural Layers

The repository is divided into logical layers.

```

Application Layer

↓

Configuration Layer

↓

Infrastructure Layer

↓

Remote Sensing Layer

↓

Feature Engineering Layer

↓

Dataset Engineering Layer

↓

Machine Learning Layer

↓

Analysis Layer

```

---

## 3.1 Application Layer

Responsible for:

- application startup
- command-line interface
- bootstrap sequence

Main components:

```

main.py

Bootstrap

```

---

## 3.2 Configuration Layer

Responsible for:

- loading configuration
- validation
- logging configuration
- environment variables

Everything configurable originates here.

---

## 3.3 Infrastructure Layer

Provides reusable infrastructure:

- exceptions
- logging
- directories
- environment validation

Every other module depends on this layer.

---

## 3.4 Remote Sensing Layer

Responsible for:

- Earth Engine
- Landsat collections
- preprocessing
- composites

No downstream module communicates directly with Google Earth Engine.

---

## 3.5 Feature Engineering Layer

Responsible for:

- harmonized bands
- spectral indices
- feature stacks

Produces machine-learning-ready feature images.

---

## 3.6 Dataset Engineering Layer

Responsible for:

- GeoTIFF export
- patches
- labels
- manifests
- metadata
- dataset assembly

This layer converts imagery into training datasets.

---

## 3.7 Machine Learning Layer

Responsible for:

- dataset loading
- transformations
- training
- inference

Only this layer depends on PyTorch.

---

## 3.8 Analysis Layer

Future modules.

Responsible for:

- river width
- channel extraction
- morphology
- statistics
- reporting

---

# 4. Repository Organization

The repository follows a domain-driven organization.

```

River-Morphology-Segmentation/

config/

docs/

src/

tests/

README.md

main.py

```

Each package contains a complete processing stage.

Examples:

```

src/

core/

gee/

export/

patches/

labels/

dataset/

training/

```

Packages should never become dependent on unrelated domains.

---

# 5. Package Dependencies

Dependencies flow in one direction only.

```

core

↓

gee

↓

export

↓

patches

↓

labels

↓

dataset

↓

training

```

Reverse dependencies are prohibited.

For example:

training

must never call

gee

directly.

---

# 6. Data Flow

The repository processes data through immutable stages.

Configuration

↓

Earth Engine ImageCollection

↓

Preprocessed Collection

↓

Composite

↓

Feature Stack

↓

GeoTIFF

↓

Patches

↓

Labels

↓

Training Dataset

↓

PyTorch Dataset

↓

Model

↓

Predictions

Each stage produces a new artifact.

Existing artifacts are never modified in place.

---

# 7. Module Interaction

Modules communicate only through public interfaces.

Example:

```

LandsatCollectionBuilder

↓

LandsatPreprocessor

↓

FeatureStackAssembler

↓

DatasetExporter

↓

PatchGenerator

↓

LabelGenerator

↓

DatasetAssembler

↓

RiverMorphologyDataset

```

No module accesses internal objects of another module.

Only public contracts may be exchanged.

---

# 8. Core Design Principles

The architecture follows several principles.

## Single Responsibility

Every class performs one task.

---

## Open / Closed

Existing code should rarely change.

New functionality should be added through extension.

---

## Dependency Injection

Dependencies should be injected whenever practical.

---

## Immutable Contracts

Modules exchange immutable dataclasses.

---

## Configuration Driven

Behaviour comes from Config.

Never hardcode.

---

## Testability

Every module should be independently testable.

---

## Extensibility

Future satellite missions should integrate without modifying existing modules.

Examples:

Sentinel-2

PlanetScope

SAR

---

# 9. Package Architecture

The repository is organized around **domain-driven packages** rather than file types.

Each package encapsulates a complete processing stage and exposes only a small, well-defined public API.

Packages should communicate only through public interfaces and typed contracts.

The overall dependency hierarchy is intentionally one-directional.

```
Application
      │
      ▼
Core
      │
      ▼
Google Earth Engine
      │
      ▼
Remote Sensing Processing
      │
      ▼
Feature Engineering
      │
      ▼
Dataset Engineering
      │
      ▼
Machine Learning
      │
      ▼
Analytics
```

Reverse dependencies are prohibited.

---

# 10. Core Package

Location:

```
src/core/
```

Purpose:

Provide reusable infrastructure for the entire repository.

Every module depends on this package.

---

## Responsibilities

Configuration

Logging

Exceptions

Environment validation

Bootstrap

Directory management

Application startup

---

## Public Components

Examples include:

```
Config

Bootstrap

DirectoryManager

EnvironmentValidator

Logger

Project Exceptions
```

---

## Responsibilities that MUST remain here

Configuration loading

Environment validation

Application initialization

Logging configuration

Global exception hierarchy

These responsibilities should never migrate into other packages.

---

# 11. Google Earth Engine Package

Location

```
src/gee/
```

Purpose

Provide a complete abstraction over the Earth Engine API.

Future modules must never communicate directly with Earth Engine.

Instead:

```
EarthEngineClient
```

is the single public gateway.

---

## Internal Components

Authentication

Health checking

Retry logic

Collection building

Preprocessing

Composite generation

Spectral indices

Feature stack generation

---

## Public Components

Examples

```
EarthEngineClient

AuthManager

HealthChecker

LandsatCollectionBuilder

LandsatPreprocessor

FeatureRegistry

FeatureStackAssembler
```

---

## Internal Processing Flow

```
Authenticate

↓

Create Collection

↓

Validate AOI

↓

Cloud Filtering

↓

Preprocessing

↓

Band Harmonization

↓

Scaling

↓

Composite

↓

Feature Engineering

↓

Feature Stack
```

---

# 12. Export Package

Purpose

Convert Earth Engine products into local datasets.

Responsibilities

GeoTIFF export

Metadata

Versioning

Manifest generation

Directory organization

---

## Public Components

Examples

```
DatasetExporter

MetadataWriter

ManifestManager

VersionManager
```

---

## Output

```
GeoTIFF Dataset
```

This package should never generate patches.

Patch generation belongs to Module 8.

---

# 13. Patch Package

Purpose

Generate machine-learning image patches.

Responsibilities

Sliding windows

Overlap

Patch metadata

Patch validation

Patch manifests

---

## Public Components

```
PatchGenerator
```

---

## Output

```
PatchDatasetResult
```

The package is intentionally unaware of labels.

Labels are generated later.

---

# 14. Label Package

Current Status

Transitioning from manual label management toward automated pseudo-label generation.

The public API remains stable.

---

## Responsibilities

Current repository

Discovery

Validation

Statistics

Metadata

Versioning

Future architecture

Spectral classification

Pseudo-label generation

Rule engine

Conflict resolution

Morphology

Confidence

Quality assessment

---

## Planned Internal Flow

```
Feature Stack

↓

Spectral Classification

↓

Rule Evaluation

↓

Conflict Resolution

↓

Morphology

↓

Confidence

↓

Validation

↓

Metadata

↓

Statistics

↓

Versioning

↓

LabelDatasetResult
```

---

## Public Contract

The package exposes only

```
LabelDatasetResult
```

The downstream pipeline should never know how labels were created.

---

# 15. Dataset Package

Purpose

Prepare machine-learning datasets.

Responsibilities

Combine

Image Patch

+

Label

↓

Training Dataset

Additional responsibilities

Dataset validation

Leakage prevention

Dataset statistics

Train/Validation/Test splitting

Manifest generation

---

## Public Components

Examples

```
DatasetAssembler

DatasetSplitter

DatasetValidator

LeakageDetector
```

---

## Output

```
TrainingDatasetResult
```

---

# 16. Training Package

Purpose

Bridge repository datasets into PyTorch.

Current responsibilities

Manifest reading

GeoTIFF loading

Tensor conversion

Dataset creation

DataLoader creation

Future responsibilities

Transforms

Augmentation

Training

Evaluation

Inference

---

## Public Components

Examples

```
RiverMorphologyDataset

GeoTIFFLoader

DatasetReader

DataLoaderFactory

Transform
```

---

## Output

```
TorchDatasetResult
```

---

# 17. Dependency Graph

The package dependency graph should always remain acyclic.

```
core

↓

gee

↓

export

↓

patches

↓

labels

↓

dataset

↓

training

↓

future modules
```

No package should depend on a package above it in the graph.

Examples

Correct

```
training

↓

dataset
```

Incorrect

```
gee

↓

training
```

---

# 18. Public Interface Philosophy

Every package exposes only a minimal public interface.

Internal helper classes remain private.

Example

```
EarthEngineClient

↓

Public
```

Internal retry implementation

↓

Private

Consumers interact only with the public interface.

---

# 19. Data Contract Flow

The repository intentionally avoids passing raw dictionaries.

Instead, processing stages exchange typed result objects.

```
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
```

Each result object is immutable whenever practical.

This prevents accidental mutation across modules.

---

# 20. Architectural Boundaries

Every package owns its own responsibility.

Examples

The Earth Engine package

owns

authentication

collection creation

preprocessing

The Dataset package

owns

splitting

validation

assembly

The Training package

owns

PyTorch integration

No package should implement functionality belonging to another package.

These boundaries are intentionally strict to preserve maintainability.

---

# 21. Architectural Design Patterns

The River Morphology Segmentation project intentionally adopts several proven software engineering design patterns.

These patterns improve maintainability, extensibility, testability, and long-term evolution of the codebase.

Future modules should follow these patterns whenever applicable.

---

# 21.1 Builder Pattern

Purpose

The Builder Pattern is used whenever complex objects require multiple validation steps before construction.

Current examples include:

- LandsatCollectionBuilder

General workflow:

```
Configuration
      │
      ▼
Builder
      │
      ▼
Validation
      │
      ▼
Construct Object
      │
      ▼
Return Immutable Result
```

Advantages:

- readable APIs
- method chaining
- centralized validation
- easier testing

Future builders should follow the same approach.

---

# 21.2 Registry Pattern

Purpose

Allow new functionality to be added without modifying existing code.

Current implementation:

FeatureRegistry

Future registry candidates:

- RuleRegistry
- ModelRegistry
- LossRegistry
- MetricRegistry
- AugmentationRegistry

General workflow:

```
Registry

↓

Registered Component

↓

Selected by Configuration

↓

Executed
```

The Registry Pattern supports the Open/Closed Principle.

---

# 21.3 Factory Pattern

Purpose

Encapsulate creation of configurable objects.

Current example:

DataLoaderFactory

Future examples:

- OptimizerFactory
- SchedulerFactory
- ModelFactory
- LossFactory

Factories should contain object construction logic.

Consumers should never instantiate complex objects directly.

---

# 21.4 Strategy Pattern

Purpose

Allow interchangeable algorithms.

Examples planned:

Dataset splitting

↓

Random

Spatial

Temporal

Pseudo label generation

↓

Spectral Rules

SAM

Hybrid

Manual

Future modules should use strategies instead of conditional logic.

---

# 21.5 Pipeline Pattern

Almost every module follows the pipeline pattern.

Example

```
Collection

↓

Preprocessing

↓

Feature Stack

↓

Export

↓

Patches

↓

Labels

↓

Dataset

↓

Training
```

Each stage consumes one typed result object and produces another.

Pipeline stages remain independent.

---

# 21.6 Dependency Injection

Dependencies should be injected whenever practical.

Avoid

```
class Trainer:

    def __init__(self):

        self.model = Model()
```

Prefer

```
class Trainer:

    def __init__(self, model):

        self.model = model
```

Advantages

- easier testing
- mocking
- flexibility
- loose coupling

---

# 22. Configuration Flow

Configuration originates from a single source.

```
config.yaml

↓

Config

↓

Typed Configuration

↓

Repository Modules
```

No module should create its own independent configuration system.

Every configurable parameter should come from Config.

Examples include:

- thresholds
- paths
- kernels
- learning rate
- batch size
- model parameters
- augmentation settings

---

# 23. Logging Architecture

Logging is centralized.

```
Config

↓

Logging Configuration

↓

Logger

↓

Repository Modules
```

Rules

- Never use print()
- Log meaningful events
- Avoid excessive verbosity
- Include contextual information
- Preserve exception tracebacks

Logging should assist debugging without cluttering output.

---

# 24. Exception Propagation

External library exceptions should never escape repository boundaries.

Example:

```
Rasterio Exception

↓

Dataset Exception

↓

Repository Exception

↓

Handled by Caller
```

Similarly:

```
Earth Engine Exception

↓

GEE Exception

↓

Repository Exception
```

Public APIs expose only project-specific exceptions.

---

# 25. Data Lifecycle

Every dataset progresses through clearly defined stages.

```
Earth Engine

↓

Image Collection

↓

Preprocessed Collection

↓

Composite

↓

Feature Stack

↓

GeoTIFF

↓

Patches

↓

Labels

↓

Training Dataset

↓

PyTorch Dataset

↓

Training

↓

Inference

↓

Analytics
```

Intermediate products are preserved whenever practical.

Generated datasets should remain reproducible.

---

# 26. Metadata Lifecycle

Every generated artifact should carry metadata.

Examples

GeoTIFF

↓

Metadata

Patch

↓

Metadata

Label

↓

Metadata

Dataset

↓

Metadata

Training

↓

Metadata

Future inference products should also preserve metadata.

---

# 27. Versioning Strategy

Every important artifact should be versioned.

Examples

Configuration Version

Dataset Version

Manifest Version

Pseudo Label Version

Model Version

Training Version

Inference Version

Version information should be propagated throughout the pipeline.

---

# 28. Testing Architecture

The repository uses layered testing.

```
Unit Tests

↓

Integration Tests

↓

System Validation
```

Each package owns its own tests.

Future modules should follow the same organization.

Google Earth Engine interactions should always be mocked.

Temporary files should never leak outside test directories.

Tests should remain deterministic.

---

# 29. Configuration Validation

Every configuration value should be validated before use.

Validation occurs as early as possible.

Examples

AOI

Date range

Cloud threshold

Kernel size

Dataset paths

Batch size

Model parameters

Invalid configuration should fail fast.

---

# 30. Error Recovery Philosophy

Where possible, recover gracefully.

Examples

Earth Engine request

↓

Retry

↓

Success

or

↓

Meaningful Exception

Never silently ignore errors.

All failures should be observable.

---

# 31. Performance Architecture

The repository is designed for scalability.

Preferred techniques include:

- lazy loading
- streaming
- batch processing
- configuration-driven parallelism
- reusable caching
- avoiding duplicated computations

Large datasets should not be loaded entirely into memory.

Operations should process data incrementally whenever possible.

---

# 32. Architectural Constraints

The following constraints are intentional.

- One-directional package dependencies.
- Immutable public contracts.
- Configuration-driven behaviour.
- Centralized logging.
- Centralized exceptions.
- High test coverage.
- Minimal public APIs.
- Reusable components.
- Backward compatibility.

Future modules should preserve these constraints.

---

# 33. Architectural Decisions

The architecture intentionally favors:

Long-term maintainability

over

Short-term convenience.

Whenever there is a choice between writing slightly more infrastructure and creating technical debt, the repository prefers the infrastructure.

This philosophy applies to every future module.

---

# 34. End-to-End Execution Flow

The complete execution flow of the repository is illustrated below.

```
                    USER INPUT
                         │
                         ▼
                  Configuration
                         │
                         ▼
                Environment Check
                         │
                         ▼
               Google Earth Engine
                         │
                         ▼
             Landsat Collection Builder
                         │
                         ▼
              Landsat Preprocessing
                         │
                         ▼
              Composite Generation
                         │
                         ▼
            Spectral Feature Generation
                         │
                         ▼
                 GeoTIFF Export
                         │
                         ▼
                Patch Generation
                         │
                         ▼
              Label Generation
                         │
                         ▼
               Dataset Assembly
                         │
                         ▼
             PyTorch Dataset Loader
                         │
                         ▼
               Training Pipeline
                         │
                         ▼
                 Trained Model
                         │
                         ▼
               Evaluation Metrics
                         │
                         ▼
                  Inference Engine
                         │
                         ▼
          River Morphology Analytics
                         │
                         ▼
              Final Reports & Maps
```

Every stage produces an artifact that is consumed by the next stage.

No stage modifies outputs from previous stages.

---

# 35. Module Communication

Modules communicate only through stable public contracts.

Example:

```
Module 6
FeatureStackResult
        │
        ▼
Module 7
DatasetExportResult
        │
        ▼
Module 8
PatchDatasetResult
        │
        ▼
Module 9
LabelDatasetResult
        │
        ▼
Module 10
TrainingDatasetResult
        │
        ▼
Module 11
TorchDatasetResult
```

No module should inspect or modify another module's internal implementation.

---

# 36. Future Module Integration

Future modules must integrate using existing public interfaces.

Current planned expansion:

```
TorchDatasetResult
        │
        ▼
Transform Pipeline
        │
        ▼
Model Factory
        │
        ▼
Training Engine
        │
        ▼
Loss Functions
        │
        ▼
Metrics
        │
        ▼
Inference Engine
        │
        ▼
River Morphology Analytics
```

New functionality should extend the pipeline rather than replacing existing components.

---

# 37. Architectural Extension Strategy

The repository is intentionally designed for long-term growth.

Future extensions should be implemented by:

- adding new modules
- adding new registries
- adding new strategies
- adding new factories
- extending configuration

Avoid modifying completed modules unless correcting a verified bug.

Backward compatibility should be preserved whenever practical.

---

# 38. AI Development Workflow

When an AI assistant contributes to the repository, it should follow this workflow.

Step 1

Read:

- AI_HANDOFF.md
- PROJECT_CONTEXT.md
- ARCHITECTURE.md
- MODULE_FREEZE.md

Step 2

Inspect the repository.

Step 3

Identify reusable classes.

Step 4

Determine whether existing functionality already solves the problem.

Step 5

Design the new module without modifying frozen modules.

Step 6

Implement the module.

Step 7

Add unit tests.

Step 8

Update documentation.

This workflow minimizes architectural drift and prevents duplicate implementations.

---

# 39. Guidelines Before Adding a New Module

Before creating a new module, verify:

✓ The functionality does not already exist.

✓ Existing classes cannot be reused.

✓ Public APIs remain unchanged.

✓ Configuration is reused.

✓ Logging is reused.

✓ Exceptions are reused.

✓ Unit tests are included.

✓ Documentation is updated.

If a feature can be implemented by extending an existing strategy or registry, prefer that approach over introducing a new subsystem.

---

# 40. Architectural Rules

The following rules define the long-term architecture of the repository.

## Stability

Completed modules should remain stable.

Avoid unnecessary refactoring.

---

## Modularity

Each package owns one domain.

Responsibilities must not overlap.

---

## Encapsulation

Internal implementation details remain private.

Expose only the minimum required public API.

---

## Configuration

Everything configurable originates from Config.

Never introduce hidden configuration.

---

## Logging

Use the centralized logging system.

Never use print statements.

---

## Exceptions

Use project-specific exceptions.

Do not expose third-party exceptions.

---

## Data Contracts

Exchange immutable result objects.

Avoid raw dictionaries.

---

## Testing

Every module requires comprehensive unit tests.

Target:

90%+ coverage.

---

## Documentation

Every architectural change should update:

- PROJECT_CONTEXT.md
- ARCHITECTURE.md
- MODULE_FREEZE.md

where applicable.

---

# 41. Repository Lifecycle

The repository follows an incremental development model.

```
Architecture
      │
      ▼
Implementation
      │
      ▼
Testing
      │
      ▼
Documentation
      │
      ▼
Review
      │
      ▼
Freeze
      │
      ▼
Next Module
```

No module is considered complete until:

- implementation is finished
- tests pass
- documentation is updated
- architectural review is completed
- the module is frozen

---

# 42. Quality Assurance Checklist

Before merging any future module:

- [ ] Public API reviewed
- [ ] No breaking changes
- [ ] Configuration-driven
- [ ] Logging integrated
- [ ] Exceptions integrated
- [ ] Type hints complete
- [ ] Documentation complete
- [ ] Unit tests passing
- [ ] Test coverage acceptable
- [ ] Existing architecture preserved

---

# 43. Long-Term Vision

The River Morphology Segmentation repository is intended to become a reusable framework for large-scale river monitoring using multispectral satellite imagery.

The architecture is designed so that future additions—including new sensors, new segmentation models, and new analytical capabilities—can be incorporated with minimal disruption.

The repository prioritizes:

- maintainability
- reproducibility
- scalability
- scientific transparency
- software engineering best practices

These principles should guide all future development.

---

# End of Document
