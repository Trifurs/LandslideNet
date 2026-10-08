# LandslideNet: Large-scale Landslide Susceptibility Mapping via Deep Learning

[![Paper](https://img.shields.io/badge/Paper-Remote%20Sensing%20of%20Environment-1f6feb)](https://doi.org/10.1016/j.rse.2026.115710)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.rse.2026.115710-blue)](https://doi.org/10.1016/j.rse.2026.115710)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of **LandslideNet** for large-scale landslide susceptibility mapping.

> Liu, B., Li, D., Xiao, X., Shao, Z., Li, Y., & Hu, J. (2026). Large-scale landslide susceptibility mapping via deep learning: A case study of Pakistan. *Remote Sensing of Environment*, **347**, 115710. https://doi.org/10.1016/j.rse.2026.115710

---

> [!WARNING]
> This README summarizes the current repository implementation and is intended to support reproducibility. Please verify critical experimental settings against the source code, configuration file, and published paper before use.

## Overview

LandslideNet is a deep-learning framework for **large-scale landslide susceptibility mapping (LSM)** from aligned environmental raster factors and a landslide inventory.

The repository provides a complete workflow for:

- terrain-based physiographic regionalization,
- leakage-safe regional cross-validation,
- fold-specific feature preprocessing,
- DWSS and random negative-sampling experiments,
- LandslideNet training and ablation studies,
- machine-learning and deep-learning comparisons,
- full-domain susceptibility-map generation, and
- experiment reporting and reproducibility auditing.

The framework was developed for large-area susceptibility modelling in Pakistan and is designed for regional-scale geohazard studies where spatial heterogeneity and geographically structured validation are important.

---

## Paper Information

**Full citation**

Liu, B., Li, D., Xiao, X., Shao, Z., Li, Y., & Hu, J. (2026). Large-scale landslide susceptibility mapping via deep learning: A case study of Pakistan. *Remote Sensing of Environment*, 347, 115710. https://doi.org/10.1016/j.rse.2026.115710

**Paper**  
https://doi.org/10.1016/j.rse.2026.115710

---

## Method

The current implementation combines four major components.

### 1. Terrain-based regionalization

Continuous physiographic macro-regions are constructed from terrain factors such as:

- DEM,
- slope,
- relief,
- roughness,
- topographic position index (TPI), and
- topographic wetness index (TWI).

These regions are label-independent and are subsequently used to define spatially separated training, validation, and test partitions.

### 2. Leakage-safe regional experiments

Feature transformations and sampling operations are fitted independently inside each outer fold. Validation and test regions are not used to fit fold-specific mappings.

The implementation supports support-aware regional validation and stores diagnostic information for reproducibility and leakage auditing.

### 3. DWSS negative sampling

The repository implements the manuscript's **DWSS** strategy together with a random-sampling control arm.

DWSS is fitted using training regions only. The current configuration uses:

- joint multivariate Gaussian KDE,
- `theta_min = 0.55`,
- three Jenks natural-break strata,
- a 1:1 positive/negative class ratio, and
- stratum-specific allocation according to divergence statistics.

If the initial candidate pool cannot satisfy a stratum quota, additional background candidates are generated. Sampling capacities, selected counts, and related diagnostics are recorded in experiment metadata.

### 4. LandslideNet

LandslideNet is implemented in `models/landslidenet.py`.

The network contains:

- a convolution-BatchNorm-ReLU stem,
- Spatial Perception Blocks (SPB),
- deformable convolutions for adaptive spatial modelling,
- Dynamic Channel Squeeze-and-Excitation (DCSE),
- residual connections,
- a feature pyramid network (FPN), and
- dense two-class susceptibility prediction.

Controlled ablation variants are also provided:

- `baseline`: neither SPB nor DCSE,
- `only_spb`: SPB without DCSE,
- `only_dcse`: DCSE with ordinary spatial convolution,
- `landslidenet`: complete proposed model.

---

## Repository Structure

```text
LandslideNet/
├── README.md
├── LICENSE
├── environment.yml
├── pyproject.toml
├── Landslide_susceptibility_mapping.xml
├── 1_data_processing.py              # terrain-based macro-region generation/checking
├── 2_model_train.py                  # regional training and comparison experiments
├── 3_model_predict.py                # full-domain susceptibility prediction
├── models/
│   ├── landslidenet.py               # LandslideNet + ablation variants
│   ├── classical.py                  # classical model definitions
│   └── comparisons.py                # deep-learning comparison models
└── utils/
    ├── data.py                       # raster/vector loading and preprocessing
    ├── regions.py                    # physiographic regionalization
    ├── experiment.py                 # experiment orchestration
    ├── training.py                   # deep-model training utilities
    ├── classical_models.py           # machine-learning training utilities
    ├── model_registry.py             # model registry and experiment groups
    ├── msmf.py                       # multi-stride median fusion
    ├── prediction.py                 # susceptibility-map generation
    ├── reporting.py                  # metric/report export
    └── progress.py                   # progress utilities
```

---

## Installation

The repository provides a Conda environment file.

```bash
git clone https://github.com/Trifurs/LandslideNet.git
cd LandslideNet

conda env create -f environment.yml
conda activate landslidenet
```

To update an existing environment:

```bash
conda env update -n landslidenet -f environment.yml
```

The supplied environment uses Python 3.10 and includes the principal dependencies required by the full workflow, including PyTorch, torchvision, rasterio, geopandas, scikit-learn, CatBoost, LightGBM, SHAP, and related scientific Python packages.

---

## Input Data

Before running the pipeline, prepare:

1. a directory containing aligned environmental-factor rasters, and
2. a landslide inventory, preferably as a point vector layer.

All factor rasters should use the same:

- coordinate reference system,
- spatial extent,
- raster dimensions,
- pixel resolution, and
- grid alignment.

The landslide inventory is reprojected to the raster grid when required, and duplicate landslide points falling in the same raster cell are removed.

The current configuration expects 20 raster bands/factors by default, but this can be changed in the XML configuration.

---

## Configuration

The main configuration file is:

```text
Landslide_susceptibility_mapping.xml
```

Update the local paths in this XML file before running the workflow.

Important configuration groups include:

- input factor and landslide-inventory paths,
- macro-region construction,
- fold-specific categorical/frequency-ratio processing,
- regional validation settings,
- DWSS/random negative sampling,
- model selection,
- deep-learning hyperparameters,
- output directories, and
- full-domain prediction settings.

> The XML file currently contains absolute paths from the original development environment. Replace them with paths valid on your own machine.

---

## Quick Start

Run the three stages in sequence.

### Step 1: Build or check physiographic macro-regions

```bash
python 1_data_processing.py Landslide_susceptibility_mapping.xml
```

### Step 2: Train LandslideNet

```bash
python 2_model_train.py   Landslide_susceptibility_mapping.xml   --models landslidenet   --output-dir ./outputs/experiment
```

Both DWSS and random sampling arms are defined by the XML configuration by default.

To continue an interrupted experiment:

```bash
python 2_model_train.py   Landslide_susceptibility_mapping.xml   --models landslidenet   --output-dir ./outputs/experiment   --resume
```

Use `--overwrite` only when existing results should be replaced.

### Step 3: Generate susceptibility maps

```bash
python 3_model_predict.py   ./outputs/experiment   --models landslidenet   --sampling-methods dwss random
```

Deep-learning models use multi-stride median fusion during full-domain prediction. Classical models use direct per-pixel inference.

---

## Model Selection

Models can be selected in the XML configuration or overridden from the command line.

Available experiment groups are:

```text
proposed
machine_learning
deep_learning
ablation
all
```

The current model registry includes:

### Proposed model

- LandslideNet

### Machine-learning comparisons

- Logistic Regression
- CatBoost
- Extra Trees
- LightGBM
- Random Forest

### Deep-learning comparisons

- DBPFNet (task-adapted)
- DA-LSF (task-adapted)
- LGC-Net (task-adapted)

### Ablation models

- Baseline
- Only DCSE
- Only SPB

For example:

```bash
python 2_model_train.py   Landslide_susceptibility_mapping.xml   --models proposed machine_learning   --output-dir ./outputs/comparison
```

> The deep-learning comparison implementations are task-adapted versions for the aligned static-factor susceptibility setting. See `utils/model_registry.py` for implementation notes and references.

---

## Experimental Controls

The current repository includes several controls intended to make regional comparisons consistent and reproducible.

For deep models, the training pipeline includes:

- fold-specific preprocessing,
- source-region balancing,
- a class-conditional source-region V-REx penalty,
- aspect-aware D4 augmentation,
- learning-rate warm-up,
- plateau-based learning-rate reduction,
- EMA-based validation,
- validation-AUC checkpoint selection,
- minimum-epoch protection before early stopping, and
- validation-only threshold selection.

The same applicable training controls are shared across sampling arms and comparison models so that DWSS and random sampling can be compared under a controlled experimental protocol.

---

## Outputs

Training and prediction results are written below the selected output directory.

Typical outputs include:

- fold-specific model checkpoints,
- configuration snapshots,
- sample manifests,
- regional split information,
- training histories,
- fold-level evaluation metrics,
- experiment metadata,
- sampling diagnostics,
- leakage-audit information, and
- raster susceptibility maps.

Two useful summary files generated by the reporting pipeline are:

```text
all_models_5fold_metrics.csv
all_models_training_history.csv
```

The first contains fold-level metrics together with five-fold mean and standard-deviation summaries. The second stores available training or boosting histories across models, folds, and sampling strategies.

---

## Reproducibility Notes

For strict reproduction of the published experiment:

- use the same environmental factors and landslide inventory,
- preserve raster alignment and preprocessing,
- retain the regional split strategy,
- do not fit fold-specific transformations using validation or test regions,
- use the same sampling strategy and random seed,
- verify all XML parameters against the experimental setup reported in the paper, and
- keep the resolved configuration snapshot stored with each experiment.

The experiment system treats the resolved training protocol as part of the resume contract. Results created using incompatible configuration snapshots should be run in a new output directory rather than resumed.

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{liu2026landslidenet,
  title={Large-scale landslide susceptibility mapping via deep learning: A case study of Pakistan},
  author={Liu, Bo and Li, D. and Xiao, X. and Shao, Z. and Li, Y. and Hu, J.},
  journal={Remote Sensing of Environment},
  volume={347},
  pages={115710},
  year={2026},
  doi={10.1016/j.rse.2026.115710}
}
```

---

## Contact

For questions about the code or paper, please contact:

**Bo Liu**  
Email: trifurs@whu.edu.cn

---

## Acknowledgements

We thank the open-source communities supporting PyTorch, geospatial Python, scientific computing, and machine learning. Their tools provide the foundation for the training, raster processing, model evaluation, and susceptibility-mapping workflows used in this project.

---

## License

This repository is released under the [MIT License](LICENSE).
