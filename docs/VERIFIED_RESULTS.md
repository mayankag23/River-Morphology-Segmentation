# River Morphology Segmentation — Verified Results

## Verification Status

The river morphology semantic-segmentation pipeline was verified successfully using the epoch-50 UNet++ checkpoint.

- Pipeline status: **SUCCESS**
- Failed stages: **0**
- Test samples: **66**
- Input bands: **11**
- Patch size: **256 × 256**
- Semantic classes:
  - Background
  - Water
  - Sand
  - Vegetation

## Model

| Property | Value |
|---|---:|
| Architecture | UNet++ |
| Trainable parameters | 9,162,084 |
| Checkpoint epoch | 50 |
| Training loss | 0.2411198 |
| Validation loss | 0.2266843 |
| Inference device | CPU |
| Mean inference confidence | 0.8414 |

## Test Evaluation

| Metric | Score |
|---|---:|
| Pixel accuracy | 0.939571 |
| Mean IoU | 0.659328 |
| Mean Dice | 0.707901 |
| Mean F1 | 0.707901 |
| Cohen's kappa | 0.879711 |
| Balanced accuracy | 0.715763 |

## Per-Class Performance

| Class | Precision | Recall | F1 / Dice | IoU |
|---|---:|---:|---:|---:|
| Background | 0.784132 | 0.019459 | 0.037975 | 0.019355 |
| Water | 0.880445 | 0.903707 | 0.891924 | 0.804931 |
| Sand | 0.970473 | 0.953118 | 0.961717 | 0.926258 |
| Vegetation | 0.897444 | 0.986766 | 0.939988 | 0.886770 |

## Interpretation

The model performs strongly on the three principal river-morphology classes:

- **Sand** achieved the highest IoU at **0.926258**.
- **Vegetation** achieved an IoU of **0.886770** and recall of **0.986766**.
- **Water** achieved an IoU of **0.804931** and F1 score of **0.891924**.

The principal limitation is the **background** class. Its recall is only **0.019459**, indicating that most ground-truth background pixels are classified as sand or vegetation. Consequently, the high overall pixel accuracy should be interpreted together with mean IoU, balanced accuracy, and the per-class metrics.

## Inference Artifacts

Inference generated outputs for all 66 test samples.

For each sample, the pipeline produced:

- Raw class-index mask
- Colorized semantic mask
- RGB source preview
- Segmentation overlay
- Source/prediction comparison image

Artifact validation found:

- Missing outputs: **0**
- Blank outputs: **0**
- Invalid outputs: **0**
- Unknown predicted class IDs: **0**

## Reproducibility

The verified evidence bundle contains:

- Epoch-50 inference log
- Semantic-class evaluation log
- Evaluation JSON
- Per-class metrics CSV
- Verification summary
- Verified Git commit SHA

The local evidence archive is:

`river_morphology_verified_results.tar.gz`

Its verified SHA-256 checksum is:

`52b62d3fb239b31066d9b1f8ecadc8092c542566091124c893ed91dcaf21aef0`
