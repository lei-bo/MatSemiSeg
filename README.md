# Few-Shot Segmentation of Complex Microstructures

This repository contains the paper *Few-Shot Segmentation of Complex Microstructures: Minimizing Training Images by Maximizing Representativeness and Diversity* by Bo Lei, Txai Sibley, and Elizabeth Holm.

The core idea is to reduce annotation cost for semantic segmentation of complex microstructures by selecting a small but informative set of training images. The proposed selection strategy, additively maximizing representativeness and diversity (AMRD), operates on unlabeled candidate images and favors samples that both represent the dataset well and cover uncommon morphologies.

## Method

AMRD follows the workflow described in the manuscript:

1. Extract pixel-level hypercolumn features from the first three VGG16 convolutional blocks.
2. Reduce feature dimensionality with PCA and build a visual vocabulary with KMeans.
3. Encode each image with VLAD.
4. Measure image similarity with cosine similarity between VLAD embeddings.
5. Select images greedily by maximizing representativeness plus a weighted diversity term.

In the paper, smaller values of the diversity weight $\lambda$ work well for more homogeneous datasets such as UHCS, while more heterogeneous datasets such as MetalDAM benefit from larger $\lambda$.

## Datasets Used In The Manuscript

The manuscript experiments focus on two datasets already organized under `data/`:

- `uhcs`: 24 SEM images of ultrahigh carbon steel with four target classes.
- `MetalDAM`: 42 SEM images from additive manufacturing metallurgy with four target classes used in this study.


## Repository Layout

- `selection/`: AMRD, random selection, hypercolumn encoding, and VLAD utilities.
- `segmentation/`: U-Net training, evaluation, configs, and checkpoints.
- `experiments/`: cross-validation runners, analysis scripts, and manuscript figure helpers.
- `data/`: images, labels, and split files.

## Setup

Install the Python dependencies:

```shell
pip install -r requirements.txt
```

## Data And Splits

The repository assumes images, labels, and split files already exist under `data/<dataset>/`.

Common split formats used here are:

- text splits for fixed train/validation/test partitions, such as `train16A.txt`, `validate4A.txt`, and `test4A.txt` in `data/uhcs/splits/`
- CSV splits for cross-validation, such as `full_CV10.csv` for UHCS and `full_CV6.csv` for MetalDAM

Selection outputs are written back into `data/<dataset>/splits/`, and model checkpoints are saved under `segmentation/checkpoints/<dataset>/<experiment>/`.

## Quick Start

### 1. Select training images with AMRD

The main selection entry point is `selection/select_img.py`.

Example: select 4 UHCS training images from the fixed 16-image candidate pool used in the manuscript.

```shell
python -m selection.select_img \
	--dataset uhcs \
	--split_file train16A.txt \
	--n_select 4 \
	--method amrd \
	--amrd_lambda 0.1
```

This writes the selected image list to:

```text
data/uhcs/splits/amrd_lambda0.1_4-shot.txt
```

Useful arguments:

- `--dataset`: dataset name such as `uhcs` or `MetalDAM`
- `--split_file`: text or CSV split file containing candidate images
- `--n_select`: number of images to select
- `--method`: `amrd` or `random`
- `--amrd_lambda`: diversity weight for AMRD
- `--n_samples_per_image`: number of sampled hypercolumns per image
- `--pca_dim`: PCA output dimension for VLAD construction
- `--n_words`: number of visual words
- `--gpu_id`: GPU index

### 2. Train a segmentation model on a fixed split

The main training entry point is `segmentation/train.py`. Config files live under `segmentation/configs/<dataset>/`.

Example: train on the UHCS fixed split with the full 16-image training set.

```shell
python -m segmentation.train --dataset uhcs --config full_sup.yaml
```

Example: train on the 4-shot AMRD-selected UHCS split.

```shell
python -m segmentation.train --dataset uhcs --config amrd_lambda0.1_4-shot.yaml
```

### 3. Run the manuscript-style cross-validation experiments

For full-supervised cross-validation:

```shell
python experiments/run_cv.py --dataset uhcs --config full_CV10.yaml -m train
python experiments/run_cv.py --dataset MetalDAM --config full_CV6.yaml -m train
```

For selected-subset cross-validation using pre-generated selection configs:

```shell
python experiments/run_selected_cv.py --dataset uhcs --config select_CV10/amrd_lambda0.1_4-shot.yaml -m train
python experiments/run_selected_cv.py --dataset MetalDAM --config select_CV6/amrd_lambda0.5_4-shot.yaml -m train
```

## Citation

If you use this repository, please cite the manuscript and reference this codebase in your experimental setup description.
