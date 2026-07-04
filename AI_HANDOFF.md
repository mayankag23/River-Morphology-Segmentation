# AI_HANDOFF.md

> River Morphology Segmentation
>
> AI Assistant Onboarding Guide
>
> Version: 1.0

---

# Purpose

This document is intended for AI coding assistants.

Before generating **any code**, the AI assistant should read this document completely.

Its purpose is to:

- understand the repository
- preserve architecture
- avoid duplicate implementations
- avoid breaking frozen modules
- maintain coding consistency
- continue development from the correct point

This document complements:

- PROJECT_CONTEXT.md
- ARCHITECTURE.md
- MODULE_FREEZE.md

Read those documents before making architectural changes.

---

# Project Summary

River Morphology Segmentation is a production-quality machine learning framework for automatic river morphology analysis using multispectral Landsat imagery.

The final objective is to produce accurate semantic segmentation of river systems and derive river morphology metrics over time.

The primary semantic classes are:

| ID | Class |
|----|-------|
| 0 | Background |
| 1 | Water |
| 2 | Sand |
| 3 | Vegetation |

The architecture is designed to support additional classes in the future.

---

# Current Project Status

Modules completed:

✔ Module 1

Configuration

✔ Module 2

Bootstrap

✔ Module 3

Google Earth Engine

✔ Module 4

Landsat Collection Builder

✔ Module 5

Preprocessing

✔ Module 6

Feature Engineering

✔ Module 7

Dataset Export

✔ Module 8

Patch Generation

✔ Module 10

Dataset Assembly

✔ Module 11

PyTorch Dataset

Module 9

Currently under architectural redesign.

Public API remains unchanged.

Future modules must continue consuming:

LabelDatasetResult

---

# Repository Philosophy

The repository follows production-quality software engineering practices.

The priorities are:

1. Maintainability

2. Reusability

3. Reproducibility

4. Extensibility

5. Testability

Do not generate prototype code.

Do not generate notebook-style implementations.

---

# Read This Before Writing Code

Before implementing anything:

1.

Read

PROJECT_CONTEXT.md

2.

Read

ARCHITECTURE.md

3.

Read

MODULE_FREEZE.md

4.

Inspect the repository.

5.

Reuse existing classes.

6.

Only then implement new functionality.

---

# Coding Rules

Always

✔ Python 3.11+

✔ Absolute imports

✔ PEP8

✔ Type hints

✔ Dataclasses where appropriate

✔ Configuration-driven behaviour

✔ Centralized logging

✔ Project exceptions

✔ Unit tests

✔ Documentation

Never

✘ Hardcode values

✘ Duplicate functionality

✘ Modify frozen modules

✘ Introduce TODOs

✘ Use print()

✘ Use Unicode characters in Python source

---

# Configuration Rules

Everything configurable must originate from Config.

Never hardcode:

- thresholds
- dataset paths
- collection IDs
- class IDs
- learning rates
- kernel sizes
- augmentation settings
- model names
- batch sizes

---

# Logging Rules

Always use the project logging system.

Never print directly.

Log:

- important operations
- warnings
- recoverable failures
- execution summaries

---

# Exception Rules

Never expose:

Earth Engine

Rasterio

PyTorch

or any other third-party exceptions.

Always convert them into project-specific exceptions.

---

# Earth Engine Rules

Never

import ee

outside the Earth Engine package.

Always use:

EarthEngineClient

Authentication must always go through:

AuthManager

Retry logic already exists.

Never duplicate it.

---

# Landsat Rules

Always use:

LandsatCollectionBuilder

BandHarmonizer

LandsatPreprocessor

FeatureStackAssembler

Never:

- hardcode collection IDs
- use raw Landsat band names
- duplicate preprocessing

---

# Feature Engineering Rules

Always obtain spectral indices through:

FeatureRegistry

Never manually calculate indices.

Always consume:

FeatureStackResult

---

# Dataset Rules

GeoTIFF export

↓

Patch generation

↓

Label generation

↓

Dataset assembly

↓

PyTorch dataset

Maintain this pipeline.

Never bypass intermediate stages.

---

# Training Rules

Dataset

↓

Transforms

↓

Model

↓

Training

↓

Evaluation

↓

Inference

Keep these responsibilities separated.

Do not mix:

- dataset logic
- augmentation
- model code
- training loops

---

# Frozen Modules

The following modules are frozen:

Modules 1–8

Module 10

Module 11

Do not modify them unless fixing an approved bug.

Module 9 is under redesign.

---

# Common Mistakes to Avoid

The following issues have occurred previously.

Avoid repeating them.

1.

Using Unicode characters in Python source.

Use ASCII only.

---

2.

Using relative imports.

Always use absolute imports.

---

3.

Hardcoding values instead of Config.

Never hardcode.

---

4.

Duplicating existing functionality.

Inspect the repository first.

---

5.

Calling Earth Engine directly.

Always use EarthEngineClient.

---

6.

Generating metadata manually.

Reuse MetadataWriter.

---

7.

Writing GeoTIFFs directly.

Reuse DatasetExporter.

---

8.

Bypassing FeatureRegistry.

Never manually compute registered indices.

---

9.

Changing public APIs of frozen modules.

Avoid breaking compatibility.

---

10.

Creating functionality already present elsewhere.

Reuse existing implementations.

---

# Before Adding a New Module

Ask:

Does this functionality already exist?

Can an existing class be reused?

Can a strategy or registry be extended instead?

Does this preserve backward compatibility?

If the answer is yes,

extend,

don't rewrite.

---

# Current Development Focus

Current priority:

Finalize Module 9 redesign.

Next planned modules:

Module 12

Data Transformation & Augmentation

Module 13

Segmentation Model Zoo

Module 14

Training Engine

Module 15

Evaluation

Module 16

Inference

Module 17

River Morphology Analytics

---

# AI Workflow

When asked to generate a module:

1.

Inspect repository.

2.

Read existing classes.

3.

Reuse architecture.

4.

Design before coding.

5.

Generate implementation.

6.

Generate tests.

7.

Generate documentation.

8.

Wait for user approval before continuing.

Never generate multiple modules unless explicitly requested.

---

# Repository Goal

The goal is **not** simply to train a segmentation model.

The goal is to build a reusable, production-quality framework for automated river morphology analysis.

Every design decision should support:

- modularity
- extensibility
- reproducibility
- maintainability

Long-term architecture is always preferred over short-term convenience.

---

# End of Document