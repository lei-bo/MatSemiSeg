# Few-Shot Segmentation of Complex Microstructures

## Setup
```shell
pip install -r requirements.txt
```

## Training image selection
The training image selection is implemented in `selection/select_img.py`. The script takes the following arguments:
- `--dataset`: the dataset name, e.g. `uhcs`
- `--split_file`: the split file name containing all the candidate training images, e.g. `train16A.txt`
- `--n_select`: the number of images to select
- `--method`: the selection method, supported methods are `random`, `amrd`
- `--amrd_lam`: the lambda parameter for AMRD, only used when `--method` is `amrd`
- `--gpu_id`: the GPU id to use, default is 0

The following command selects 4 images from a UHCS training set using AMRD with $\lambda = 0.1$:
```shell
python -m selection.select_img --dataset uhcs --split_file train16A.txt --n_select 4 --method amrd --amrd_lam 0.1
```

## Training segmentation models
The details of data preparation and training configuration can be found at https://github.com/leibo-cmu/MatSeg. 
The segmentation model training is implemented in `segmentation/train.py`. The script takes the following arguments:
- `--config`: the config file name, e.g. `full_sup.yaml`
- `--gpu_id`: the GPU id to use, default is 0

The following command trains a segmentation model using the fully-supervised setting(16 training images) / AMRD(4 training images):
```shell
python -m segmentation.train --config full_sup.yaml
python -m segmentation.train --config amrd:lam0.1_4-shot.yaml
```