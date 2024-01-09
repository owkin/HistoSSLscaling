<div align="center">

<h1>Scaling Self-Supervised Learning for Histopathology with Masked Image Modeling</h1>

</div>


<details>
<summary>
  <b>Scaling Self-Supervised Learning for Histopathology with Masked Image Modeling</b>, MedRxiv, July 2023.   
  
  [[`MedRxiv`]](https://www.medrxiv.org/content/10.1101/2023.07.21.23292757v2) [[`Project page`]](https://www.owkin.com/publications/scaling-self-supervised-learning-for-histopathology-with-masked-image-modeling)  [[`Paper`]](https://www.medrxiv.org/content/10.1101/2023.07.21.23292757v2.full.pdf)

</summary>

> Filiot, A., Ghermi, R., Olivier, A., Jacob, P., Fidon, L., Kain, A. M., Saillard, C., & Schiratti, J.-B. (2023). Scaling Self-Supervised Learning for Histopathology with Masked Image Modeling. MedRxiv.

```bash
@article{Filiot2023scalingwithMIM,
	author       = {Alexandre Filiot and Ridouane Ghermi and Antoine Olivier and Paul Jacob and Lucas Fidon and Alice Mac Kain and Charlie Saillard and Jean-Baptiste Schiratti},
	title        = {Scaling Self-Supervised Learning for Histopathology with Masked Image Modeling},
	elocation-id = {2023.07.21.23292757},
	year         = {2023},
	doi          = {10.1101/2023.07.21.23292757},
	publisher    = {Cold Spring Harbor Laboratory Press},
	url          = {https://www.medrxiv.org/content/early/2023/07/26/2023.07.21.23292757v2},
	eprint       = {https://www.medrxiv.org/content/early/2023/07/26/2023.07.21.23292757v2.full.pdf},
	journal = {medRxiv}
}
```
</details>

### Update :tada: Phikon release on Hugging Face :tada:
We released our Phikon model on [Hugging Face](https://huggingface.co/owkin/phikon). Check out our community [blog post](https://huggingface.co/blog/EazyAl/phikon) !
We also provide a [Colab notebook](https://colab.research.google.com/drive/1zjxscEBgpizHBCwMy-aNz2916AVdB642) to perform weakly-supervised learning on Camelyon16 and fine-tuning with LoRA on NCT-CRC-HE using Phikon.

Here is a code snippet to perform feature extraction using Phikon.
```python
from PIL import Image
import torch
from transformers import AutoImageProcessor, ViTModel

# load an image
image = Image.open("assets/example.tif")

# load phikon
image_processor = AutoImageProcessor.from_pretrained("owkin/phikon")
model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)

# process the image
inputs = image_processor(image, return_tensors="pt")

# get the features
with torch.no_grad():
    outputs = model(**inputs)
    features = outputs.last_hidden_state[:, 0, :]  # (1, 768) shape
```
___



**Official PyTorch Implementation** and pre-trained models for `Scaling Self-Supervised Learning for Histopathology with Masked Image Modeling`. This minimalist repository aims to:
- **Publicly release the weights of our Vision Transformer Base (ViT-B) model **Phikon** pre-trained with iBOT on 40M pan-cancer histology tiles from TCGA.** **Phikon** achieves state-of-the-art performance on a large variety of downstream tasks compared to other SSL frameworks available in the literature.

⚠️ **Addendum** :warning:

**From 09.01.2023 to 10.30.2023, this repository stated using the `student`, please use the `teacher` backbone instead**.

```python
# feature extraction snippet with `rl_benchmarks` repository
from PIL import Image
from rl_benchmarks.models import iBOTViT

# instantiate iBOT ViT-B Pancancer model, aka Phikon
# /!\ please use the "teacher" encoder which produces better results !
weights_path = "/<your_root_dir>/weights/ibot_vit_base_pancan.pth">
ibot_base_pancancer = iBOTViT(architecture="vit_base_pancan", encoder="teacher", weights_path=weights_path)

# load an image and transform it into a normalized tensor
image = Image.open("assets/example.tif")  # (224, 224, 3), uint8
tensor = ibot_base_pancancer.transform(image) # (3, 224, 224), torch.float32
batch = tensor.unsqueeze(0)  # (1, 3, 224, 224), torch.float32

# compute the 768-d features
features = ibot_base_pancancer(batch).detach().cpu().numpy()
assert features.shape == (1, 768)
```


- **Publicly release the histology features of our ViT-based iBOT models** (`iBOT[ViT-S]COAD`, `iBOT[ViT-B]COAD`, `iBOT[ViT-B]PanCancer`, `iBOT[ViT-L]COAD`) for i) 11 TCGA cohorts and Camelyon16 slides datasets; and ii) NCT-CRC and Camelyon17-Wilds patches datasets.
- Reproduce the results from our publication, including: features extraction and clinical data processing, cross-validation experiments, results generation.

# Abstract
<details>
<summary> Read full abstract from MedRxiv.

![main_figure](assets/main_figure.png)

</summary>
Computational pathology is revolutionizing the field of pathology by integrating advanced computer vision and machine learning technologies into diagnostic workflows. Recently, Self-Supervised Learning (SSL) has emerged as a promising solution to learn representations from histology patches, leveraging large volumes of unannotated whole slide images whole slide images (WSI). In particular, Masked Image Modeling (MIM) showed remarkable results and robustness over purely contrastive learning methods. In this work, we explore the application of MIM to histology using iBOT, a self-supervised transformer-based framework. Through a wide range of downstream tasks over seven cancer indications, we provide recommendations on the pre-training of large models for histology data using MIM. First, we demonstrate that in-domain pre-training with iBOT outperforms both ImageNet pre-training and a model pre-trained with a purely contrastive learning objective, MoCo V2. Second, we show that Vision Transformers (ViT), when scaled appropriately, have the capability to learn pan-cancer representations that benefit a large variety of downstream tasks. Finally, our iBOT ViT-Base model, pre-trained on more than 40 million histology images from 16 different cancer types, achieves state-of-the-art performance in most weakly-supervised WSI classification tasks compared to other SSL frameworks. Our code, models and features are publicly available at https://github.com/owkin/HistoSSLscaling.
</details>

# Data structure

## Download

You can download the data necessary to use the present code and reproduce our results here:
- raw data: [Google Drive](https://drive.google.com/drive/folders/1_fsnJqyNS00WWWP38NTAWk1n9sD2iCEq?usp=drive_link)
- preprocessed data: [Google Drive](https://drive.google.com/drive/folders/1pZkayJjhvgRZUU6Q3ADIjbXJ2IemXvlR?usp=drive_link)
- weights: [Google Drive](https://drive.google.com/drive/folders/1wIrLw4KZa8oI3hZVykH1dyvXu08_WwmL?usp=drive_link)

Please create `weights`, `raw` and `preprocessed` folders containing the content of the different downloads. This step may take time depending on your wifi bandwidth (folder takes **1.2 To**). You can use [rclone](https://rclone.org/) to download the folder from a remote machine (preferred in a `tmux` session).

## Description

The bucket contains three main folders: a `weights`, `raw` and `preprocessed` folders. The `weights` folder contains weights for `iBOT[ViT-B]PanCancer` (our best ViT-B iBOT model). Other models from the literature can be retrieved from the corresponding Github repositories:
- CTransPath: https://github.com/Xiyue-Wang/TransPath
- HIPT: https://github.com/mahmoodlab/HIPT
- Dino[ViT-S]BRCA: https://github.com/Richarizardd/Self-Supervised-ViT-Path

````
weights/
└── ibot_vit_base_pancan.pth          # Ours
````

The `raw` folder contains two subfolders for slide-level and tile-level downstream task.

- Slide-level: each cohort contains 2 folders, `clinical` and `slides`. We provide clinical data but not raw slides. No modification was performed on the folders architectures and files names of raw slides and patches compared to the original source (i.e. TCGA, Camelyon16, NCT-CRC and Camelyon17-WILDS).
- Tile-level: each cohort contains 2 folders, `clinical` and `patches`. We only provide clinical data (i.e. labels), not patches datasets.

> [!WARNING]
> **We don't provide raw slides or patches (`slides`, `patches` folders are empty).**
> You can download raw slides or patches here:
> - PAIP: http://www.wisepaip.org/paip/guide/dataset
> - TCGA: https://portal.gdc.cancer.gov/
> - Camelyon16: http://gigadb.org/dataset/100439
> - NCT-CRC: https://zenodo.org/record/1214456
> - Camelyon17-WILDS: https://github.com/p-lambda/wilds/blob/main/wilds/download_datasets.py
>
> Once you downloaded the data, please follow the same folders architecture as indicated below (without applying modifications on folders and files names compared to original download).


````
raw/
├── slides_classification               # slides classification tasks
===============================================================================
│   ├── CAMELYON16_FULL                 # cohort
│   │   ├── clinical                    # clinical data (for labels)
│   │   │   ├── test_clinical_data.csv
│   │   │   └── train_clinical_data.csv
│   │   └── slides                      # raw slides (not provided)
│   │        ├── Normal_001.tif
│   │        ├── Normal_002.tif...
│   └── TCGA
│       ├── tcga_statistics.pk          # For each cohort and label, list (n_patients, n_slides, labels_distribution)
│       ├── clinical                    # for TCGA, clinical data is divided into subfolders
│       │   ├── hrd
│       │   │   ├── hrd_labels_tcga_brca.csv
│       │   │   └── hrd_labels_tcga_ov.csv
│       │   ├── msi
│       │   │   ├── msi_labels_tcga_coad.csv
│       │   │   ├── msi_labels_tcga_read.csv...
│       │   ├── subtypes
│       │   │   ├── brca_tcga_pan_can_atlas_2018_clinical_data.tsv.gz
│       │   │   ├── coad_tcga_pan_can_atlas_2018_clinical_data.tsv.gz...
│       │   └── survival
│       │       ├── survival_labels_tcga_brca.csv
│       │       ├── survival_labels_tcga_coad.csv...
│       └── slides
│           └── parafine
│               ├── TCGA_BRCA
│               │   ├── 03627311-e413-4218-b836-177abdfc3911
│               │   │   └── TCGA-XF-AAN7-01Z-00-DX1.B8EDF045-604C-48CB-8E54-A60564CAE2AD.svs
...

└── tiles_classification                # tiles classification tasks
===============================================================================
    ├── CAMELYON17-WILDS_FULL           # cohort
    │   ├── clinical                    # clinical data (for labels)
    │   │    └── metadata.csv
    │   └── patches                     # patches (not provided)
    │        ├── patient_004_node_4...
    │        │   ├── patch_patient_004_node_4_x_10016_y_16704.png...
    └── NCT-CRC_FULL
        ├── labels                      # here the labels are set using the folders architecture
        │   └── dict_labels.pkl
        └── patches
            ├── NCT-CRC-VAL-HE-7K
            │    ├── ADI...
            │    │    ├── ADI-TCGA-AAICEQFN.tif...
            └── NCT-CRC-HE-100K-NONORM
                 ├── ADI...
                 │    ├── ADI-AAAFLCLY.tif...
````

The `preprocessed` folder contains two subfolders for slide-level and tile-level downstream tasks.
- Slide-level: for each feature extractor and dataset, we provide coordinates and features. Coordinates are provided as (N_tiles_slide, 3) numpy arrays where the 3 first columns rows correspond to `(tile_level, x_coordinate, y_coordinate)`. Features are provided as (N_tiles_slide, 3+d) numpy arrays, the d last columns being the model's features (3 first are the previous coordinates). **Coordinates are meant to extract the same tiles as done in our publication but are not needed for downstream experiments (only features are needed)**. Note that coordinates are divided into `coords_224`, `coords_256` and `coords_4096`, corresponding to 224 x 224 tiles (iBOT, CTransPath and ResNet models), 256 x 256 (Dino models) and 4096 x 4096 (HIPT) tiles, respectively. 

> [!NOTE]
> We provide all matter tiles for each slide. All tiles were extracted at 0.5 micrometers / pixel (20x magnification) except for CTransPath (mpp = 1.0 following the authors recommendation).

> [!WARNING]
> The `tile_level` is computed with `openslide.deepzoom.DeepZoomGenerator` through the following schematic syntax:
>
> ```python
> from openslide import open_slide
> from openslide.deepzoom import DeepZoomGenerator
>
> slide = open_slide("<slide_path>")
> dzg = DeepZoomGenerator(slide, tile_size=224, overlap=0)
> tile = dzg.get_tile(level=17, address=(8, 10))
> # this corresponds to coordinates (17, 8, 10) in the coordinates we provide for the given slide
>```
- Tile-level: for each feature extractor and dataset, we provide patches ids and features. Features are (N_patches_dataset, d) numpy arrays and ids take the form of (N_patches_dataset, 1) string numpy array.


Here is a description of the different features and coordinates we provide in the `preprocessed` folder.

```
preprocessed/                         # preprocessed data (coords, features)
===============================================================================
├── slides_classification             # slides classification tasks
│   ├── coords
│   │   ├── coords_224                # coordinates for 224 x 224 tiles
│   │   │   ├── CAMELYON16_FULL       # cohort 
│   │   │   │   ├── Normal_001.tif    # slide_id
│   │   │   │       └── coords.npy    # coordinates array (N_tiles_slide, 3)
...
│   │   │   ├── TCGA
│   │   │   │   ├── TCGA_BRCA
│   │   │   │   │   ├── TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF-D8FB-45CF-B4BD-C6B76294C291.svs
│   │   │   │   │       └── coords.npy
...
│   │   ├── coords_256                # coordinates for 256 x 256 tiles
│   │   └── coords_4096               # coordinates for 4096 x 4096 tiles

...
│   └── features                      # features
│       ├── iBOTViTBasePANCAN         # feature extractor
│       │   ├── CAMELYON16_FULL       # cohort
│       │   │   ├── Normal_001.tif    # slide_id
│       │   │       └── features.npy  # features array (N_tiles_slide, 3+d)
...
│       │   ├── TCGA
│       │   │   ├── TCGA_BRCA
│       │   │   │   ├── TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF-D8FB-45CF-B4BD-C6B76294C291.svs
│       │   │   │       └── features.npy
...
│       ├── MoCoWideResNetCOAD        # same structure applies for all extractors
│       ├── ResNet50
│       ├── iBOTViTBaseCOAD
│       ├── iBOTViTBasePANCAN
│       ├── iBOTViTLargeCOAD
│       ├── iBOTViTSmallCOAD
...
/!\ If you wish to extract features for Dino[ViT-S]BRCA, Dino[ViT-S]PanCancer, HIPT and CTransPath, those features should be placed here.

│       ├── DinoChenBRCA              
│       ├── DinoChenPancancer
│       ├── HIPT
│       └── CTransPath
===============================================================================
└── tiles_classification              # tiles classification tasks
    └── features                      # features
        ├── iBOTViTBasePANCAN         # feature extractor
        │   ├── CAMELYON17-WILDS_FULL # cohort
        │   │   ├── tile_features.npy # tiles features array (N_tiles_cohort, d)
        │   │   └── tile_ids.npy      # tiles ids array (N_tiles_cohort,)
        │   └── NCT-CRC_FULL
        │       ├── tile_features.npy
        │       └── tile_ids.npy
        ├── MoCoWideResNetCOAD
        ├── ResNet50
        ├── iBOTViTBaseCOAD
        ├── iBOTViTBasePANCAN
        ├── iBOTViTLargeCOAD
        └── iBOTViTSmallCOAD
````

# `rl_benchmarks` repository

You can find a detailed description of the repository in ``rl_benchmarks/README.md`` file.

0. [Hardware](#hardware)
1. [Installation](#installation)
2. [Feature extraction](#feature-extraction)
3. [Slide-level downstream tasks](#slide-level-downstream-tasks)
4. [Tile-level downstream tasks](#tile-level-downstream-tasks)
5. [Notes](#notes)

## Hardware

As a pre-requirement, we suggest to work on a machine with at least 8 CPUs, 32 Gb RAM and 1 GPU with at least 15Gb RAM. For instance, our experiments run on a Tesla T4 (15 Gb RAM), 32 Intel(R) Xeon(R) CPUs (@ 2.00GHz) and 64 Gb RAM.

## Installation

### Installing OpenSlide

`rl_benchmarks` relies on the `OpenSlide` library to read WSI. The python bindings are automatically installed with `rl_benchmarks` library, but you will also need the C library:

- On Linux:
  ```shell
  apt update && apt install openslide-tools
  ```

### Installing the correct pixman version
Pixman is a dependency of `libopenslide` (the C library installed through `apt`). Note that versions `0.3*` and `0.4*` gives different results one versus the other. Experiments were conducted with version `0.36.0`. You can change the system wide version of Pixman using `apt`. The following command should show the version installed:
```bash
apt list --installed | grep pixman
```
If the returned version is not `0.36.0` you can try to install it with your package manager:
```bash
sudo apt update
sudo apt install libpixman-1-0
```

All the accessible versions are stored in [this website](https://pkgs.org/download/libpixman-1-0).
You can eventually run `apt-get check` to check for broken dependencies.


### Installing `rl_benchmarks` package within this repo

Create a dedicated conda environment (optional):

```bash
conda create -n rl_benchmarks python=3.8
conda activate rl_benchmarks
```

Install `rl_benchmarks` package and its dependencies using the `install.sh` file.

```bash
git clone https://github.com/owkin/HistoSSLscaling.git
cd ./HistoSSLscaling
# Install the RL_benchmarks repository (in editable mode) together with other requirements
python -m pip install -e .  -r requirements.txt 
```

Once the installation and data download steps are completed, you finally need to edit the `conf.yaml` file so that to specify:
- `logs_save_dir`: directory for cross-validation experiments logs
- `data_dir`: root directory that contains the downloaded data. If you downloaded the data in `/home/user/downloaded_data/` folder, then this should be the `data_dir`.

### Run tests (10 minutes)

Once data has been downloaded and the previous installation steps done, you can
run the full test suite to make sure features are loaded correctly. You first
need to add specific requirements via:
```bash
python -m pip install -r requirements-tests.txt
```
Then, you can run the whole stack of tests by running the following command (within a `tmux` session is strongly recommended):

```bash
bash dev_tools/run_tests.sh
```

You can also perform linting checks via:
```bash
bash dev_tools/linting.sh
```
> [NOTE!]
> If you also wish to only test that your raw data (WSIs datasets and tiles
> datasets) follow the good structure, please run
> ```bash
> pytest -v tests/ -m test_raw_data_loading
> ```
>

## Feature extraction

This repository enables you to extract and store the features associated with our `iBOT[ViT-B]PanCancer`. Beforehand, you will need to download raw slides and *strictly* stick to the architecture described in the `Data structure` section (`raw` folder).

> [!NOTE]
> If you are only interested in reproducing the results by running cross-validations, you can directly download and use coordinates and features (provided as numpy arra`ys) for all representation learning models and cohorts used in our publication.

### Slide features extraction

To extract features for each slide of a slide-level dataset, use the following tool: `./tools/extract_features/extract_slide_features.py`.

```bash
python ./tools/extract_features/extract_slide_features.py \
  feature_extractor=$feature_extractor \
  slide_dataset=$dataset \
  n_tiles=$n_tiles \
  batch_size=$batch_size \
  random_sampling=$random_sampling \
  seed=$seed \
  num_workers=$num_workers \
  device=$device \
  features_output_dir=$output_dir
```

Example:

```bash
python ./tools/extract_featuresextract_slide_features.py \
  feature_extractor="iBOTViTBasePANCAN" \
  slide_dataset="tcga_coad" \
  n_tiles=1_000 \
  batch_size=64 \
  random_sampling=True \
  seed=0 \
  num_workers=8 \
  device="[0,1]" \
  features_output_dir=null 
```

The following command extracts features from `TCGA-COAD` cohort using our ViT-based iBOT model `iBOT[ViT-B]PanCancer`. 1,000 slides per slide are extracted in a random order (with seed set to 0). Process uses 2 GPUs (id 0 and 1) and 8 workers. `features_output_dir=null` will assign `None` value to `features_output_dir`. In that case, the path to the features output directory will automatically be picked up in `conf.yaml` file.

> [!NOTE]
> Slide features are saved as follows:
> `{features_path}/{feature_extractor}/{slide_dataset}/{slide_id}.{slide_format}/features.npy`
>
> For each slide, a (N_tiles, 3+d) numpy arrays is saved, with `d` being the model's last layer. The 3 first columns rows correspond to `(tile_level, x_coordinate, y_coordinate)` where `tile_level` is computed with `openslide.deepzoom.DeepZoomGenerator` (see "Data structure" section).
>
> For example:
>
> `/workspace/data/preprocessed/slides_classification/features/ResNet50/TCGA/TCGA_COAD/TCGA-AA-3864-01Z-00-DX1.f6992bc7-ba05-4c30-9500-8f7b07b30f9a.svs/features.npy`
>
> To import them, you can use `np.load`:
>
> ```python
> import numpy as np
>
> features = np.load(”features.npy”)
> assert features.shape == (n_tiles, feature_dim+3)
> ```

> [!WARNING]
> Once you have downloaded the data, tile levels and coordinates are automatically retrieved for each cohort and feature extractor. Our repository allows to generate the features used in our experiments. If you wish to change the tiles coordinates and level, you can create new `coords.npy` files and change the path to coordinates folder in the `constants.py` file.

### Tile features extraction

To extract features for tile-level datasets (i.e. NCT-CRC and Camelyon17-WILDS), use the following tool: `./tools/extract_features/extract_tile_features.py`.

```bash
python ./tools/extract_features/extract_tile_features.py \
    tile_dataset=$dataset \
    feature_extractor=$feature_extractor \
    batch_size=$batch_size \
    seed=$seed \
    num_workers=$num_workers \
    device=$device \
    output_dir=$output_dir
```

Example:

The following command extracts features from `TCGA-COAD` cohort using our ViT-based iBOT model `iBOT[ViT-B]PanCancer`. 1,000 slides per slide are extracted in a random order (with seed set to 0). Process uses 2 GPUs (id 0 and 1) and 8 workers. `features_output_dir=null` will assign `None` value to `features_output_dir`. In that case, the path to the features output directory will automatically be picked up in `conf.yaml` file.

```bash
python ./tools/extract_features/extract_tile_features.py \
    tile_dataset="nct_crc" \
    feature_extractor="iBOTViTBasePANCAN" \
    batch_size=64 \
    seed=0 \
    num_workers=8 \
    device="[0,1]" \
    output_dir=null
```

> [!NOTE]
> Tile features are saved as two numpy arrays, one containing the tile features (`tile_features.npy`) and the other containing the corresponding tile ids (`tile_ids.npy`) in `{features_path}/{feature_extractor}/{tile_dataset}/` folder.
>
> For example:
>
> `/workspace/data/preprocessed/tiles_classification/features/ResNet50/NCT-CRC_FULL/tile_features.npy` and `/workspace/data/preprocessed/tiles_classification/features/ResNet50/NCT-CRC_FULL/tile_ids.npy`
>
> ```python
> import numpy as np
> 
> features = np.load("tile_features.npy")
> ids = np.load("tile_ids.npy")
> assert features.shape == (n_samples, feature_dim)
> assert ids.shape == (n_samples,)
> ```

### Bash script

> [!WARNING]
> If you wish to run all feature extractions sequentially, you can directly run
> ```bash
> bash scripts/extract_slide_features.sh
> bash scripts/extract_tile_features.sh
> ```
> In those files, datasets and feature extractor are referenced as follows:
> ```bash
> datasets="camelyon16_full tcga_coad tcga_kich tcga_kirc tcga_kirp tcga_luad tcga_lusc tcga_ov tcga_paad tcga_read tcga_stad "
> ```
> ```bash
>feature_extractors="iBOTViTBasePANCAN"
> ```


### Other feature extractors

If you wish to extract features (both at the slide and tile-level) for `CTransPath` [1], `HIPT` [2], `DinoChenBRCA` [3] and `DinoChenPancancer` [2], please directly use the corresponding repositories. Those models correspond to `CTransPath`, `HIPT`, `Dino[ViT-S]BRCA` and `HIPT[ViT_256]`:

- `CTransPath` ([1], named `CTransPath` in our repository): [extraction script](https://github.com/Xiyue-Wang/TransPath/blob/main/get_features_CTransPath.py). Wang, Xiyue, et al. "Transformer-based unsupervised contrastive learning for histopathological image classification." Medical image analysis 81 (2022): 102559.
- `HIPT` ([2], named `HIPT` in our repository): [extraction script](https://github.com/mahmoodlab/HIPT#how-hipt-works). Richard J. Chen, Chengkuan Chen, Yicong Li, Tiffany Y. Chen, Andrew D. Trister, Rahul G. Krishnan, and Faisal Mahmood. "Scaling vision transformers to gigapixel images via hierarchical self-supervised learning". In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16144–16155, June 2022.
- `Dino[ViT-S]BRCA` ([3], named `DinoChenBRCA` in our repository): [extraction cript](https://github.com/Richarizardd/Self-Supervised-ViT-Path/blob/master/patch_extraction.py). Richard J Chen and Rahul G Krishnan. "Self-supervised vision transformers learn visual concepts in histopathology". Learning Meaningful Representations of Life, NeurIPS 2021, 2021.
- `HIPT[ViT_256]` ([23] Dino[ViT-S]PanCancer in our repository): [weights](https://github.com/mahmoodlab/HIPT/blob/master/HIPT_4K/Checkpoints/vit256_small_dino.pth). 

After preprocessing of WSIs, we suggest using the above SSL models on tiles with same coordinates as provided in `coords_256` (`Dino[ViT-S]BRCA`, `HIPT[ViT_256]`), `coords_4096` (HIPT) and `coords_224` (CTransPath). Generated features should follow, for each dataset, the same structure as described in the previous sections (1 features matrix for slides with (deepzoom_level, x, y) coordinates as first 3 columns, 1 features matrix for tiles-datasets).

## Running experiments

This section describes how to run cross-validation experiments.

### Slide-level downstream tasks

The `scripts/slides_classification.sh` script allows you to run 5x5 nested cross-validations on slide classification tasks (TCGA cohorts and Camelyon16 dataset). No parameter tweaking should be performed. `scripts/slides_classification.sh` iterates on:
- Slides classification tasks:
```bash
"
camelyon16_train_tumor_prediction
tcga_crc_msi_prediction
tcga_stad_msi_prediction
tcga_ov_hrd_prediction
tcga_brca_hrd_prediction
tcga_brca_histological_subtype_prediction
tcga_brca_molecular_subtype_prediction
tcga_nsclc_cancer_subtype_prediction
tcga_rcc_cancer_subtype_prediction
tcga_brca_os_prediction
tcga_coad_os_prediction
tcga_luad_os_prediction
tcga_lusc_os_prediction
tcga_paad_os_prediction
"
```
- Feature extractors (a thorough description is provided in our publication):
```bash
"
ResNet50
MoCoWideResNetCOAD
iBOTViTSmallCOAD
iBOTViTBaseCOAD
iBOTViTBasePANCAN
iBOTViTLargeCOAD
"
```
- Multiple Instance Learning models (description below):
```bash
"
mean_pool
chowder
abmil
dsmil
hipt_mil
trans_mil
"
```

- `HIPTMIL` is a lightweight transformer MIL aggregator used in [Chen et al.](https://arxiv.org/pdf/2206.02647.pdf), page 6., to aggregate 4096x4096 features into a WSI-wise representation, further used for fine-tuning. The authors also call this MIL model _3-stages HIPT model_ with local and global pretraining. In this repo, we only consider frozen local and global transformers. Fine-tuning is only performed for the last transformer. See [details here](https://github.com/mahmoodlab/HIPT/blob/master/2-Weakly-Supervised-Subtyping/models/model_hierarchical_mil.py#L69).
- `DSMIL` is a MIL aggregator proposed in [Li et al.](https://arxiv.org/pdf/2011.08939.pdf). It takes tile-level features as input and produces a classification score based on a dual-stream mechanism (two branches). The first branch computes a "critical" instance (i.e. an important tile for the classification) and its score. The second branch computes a representation and an attention score for every tile (with respect to the critical instance), averages these representations to get a single slide-wise representation, and then computes a score. The final score is the average of the scores from the two branches.
- `ABMIL` is a MIL aggregator proposed in [Ilse et al.](https://arxiv.org/pdf/1802.04712.pdf). It computes a representation and an attention score for every tile, and computes a slide-wise representation by averaging the representations with respect to the attention scores. This slide-wise representation is then passed to an MLP for the final task.
- `Chowder` has been proposed in [Courtiol et al.](https://arxiv.org/pdf/1802.02212.pdf). It computes a score for each tile and selects only the top and bottom `N` scores. These `2N` scores are then passed to an MLP for the downstream task.
- `MeanPool` computes the slide-average representation from all tiles and applies an MLP on top of it.
- `TransMIL` implements the model proposed by [Shao et al.](https://arxiv.org/abs/2106.00908.pdf). The TransMIL model is composed of the following steps: 1) sequence squaring, 2) Correlation modelling, 3) Position encoding (Pyramid Position Encoding Generator), 4) Deep Feature Aggregation and 5) Classification (see Figure 3 and Algorithm 2 in [Shao et al.](https://arxiv.org/abs/2106.00908.pdf)).

---

During nested-cross validations, gridsearching is performed on 2 hyperparameters: learning rate ($\{10^{-3}, 10^{-4}\}$) and weight decay ($\{0, 10^{-4}\}$) as defined by the following instructions:
```bash
learning_rate_gs="[1.0e-3,1.0e-4]"
weight_decay_gs="[0.,1.0e-4]"
```

Also, stratification is performed at the patient level:
```bash
stratified=True
split_mode="patient_split"
```

The script `tools/slide_level_tasks/get_results.py` allows you to retrieve slide-classification results from each experiments. Output results take the form of a `pd.DataFrame` with all experiments' parameters and corresponding results. To get the results of nested cross-validations, simply do:
```bash
python tools/slide_level_tasks/get_results.py
```


> [!WARNING]
> `hipt_mil` algorithm needs to set 1 slide's feature matrix per batch (batch size = 1). You can find the original implementation by HIPT authors [here](https://github.com/mahmoodlab/HIPT/blob/master/2-Weakly-Supervised-Subtyping/models/model_hierarchical_mil.py#L156).

### Tile-level downstream tasks

The `scripts/tiles_classification.sh` script allows you to run cross-validations and test evaluation on tile classification tasks on NCT-CRC and Camelyon17-WILDS datasets using a standard SGD classifier _on top of frozen features_. This script makes use of `tools/tile_level_tasks/linear_evaluation.py` which performs linear evaluation and stores metrics accordingly.

`scripts/slides_classification.sh` iterates on:
- tiles classification tasks:
```bash
"camelyon17_wilds nct_crc"
```
- feature extractors:
```bash
"
ResNet50
MoCoWideResNetCOAD
iBOTViTSmallCOAD
iBOTViTBaseCOAD
iBOTViTBasePANCAN
iBOTViTLargeCOAD
"
```

> [!NOTE]
> Once corresponding features have been extracted and stored appropriately according to our data structure (see first section), you can run the above experiments on `CTransPath`, `HIPT`, `DinoChenBRCA` and `DinoChenPancancer` by simply adding in the `feature_extractors` parameter (bash scripts): `CTransPath HIPT DinoChenBRCA DinoChenPancancer`. Note that should not use HIPT for tiles classification but rather the first ViT-S/256 extractors (which is denoted by `DinoChenPancancer`).

## Notes
When loading `iBOTViTBasePANCAN`, you may encounter the following message:

```bash
Pretrained weights found at <your_data_dir>/weights/ibot_vit_base_pancan.pth and loaded with msg: _IncompatibleKeys(missing_keys=[], unexpected_keys=['head.mlp.0.weight', 'head.mlp.0.bias', 'head.mlp.2.weight', 'head.mlp.2.bias', 'head.mlp.4.weight', 'head.mlp.4.bias', 'head.last_layer.weight_g', 'head.last_layer.weight_v', 'head.last_layer2.weight_g', 'head.last_layer2.weight_v'])
```

If so, this message is normal as our weights also contain the final MLP head,
which are is needed for features extraction.

# Todo
- [ ] Add CI configuration in `.github/workflows/`.
- [ ] Add Sphinx documentation.


# License

# Issues

Please open new issues directly on the repository, we'll do our best to address those quickly.

# Acknowledgements

Vision Transformers architectures were derived from [facebookresearch/dino](https://github.com/facebookresearch/dino) (Apache License 2.0), [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models/tree/main) (Apache License 2.0) and [lmlpen/Nystromformer](https://github.com/mlpen/Nystromformer/tree/main) (MIT License) repositories.

`hipt_mil` multiple-instance learning algorithm was directly inspired from the [HIPT repository](https://github.com/mahmoodlab/HIPT/blob/master/2-Weakly-Supervised-Subtyping/models/model_hierarchical_mil.py) (Apache License 2.0 with Commons Clause).

**The following table describe the different libraries used in this work.**

| Name of the code library | Version | License                                 | Licensor                                  | Github repository                                                                                                             |
| ------------------------ | ------- | --------------------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| HIPT                     |   \-   | Apache License 2.0 with Commons Clause | Mahmood Lab                               | [https://github.com/mahmoodlab/HIPT/](https://github.com/mahmoodlab/HIPT/blob/master/LICENSE)                                 |
| dino                     | \-      | Apache License 2.0                      | Not specified                             | [https://github.com/facebookresearch/dino/](https://github.com/facebookresearch/dino/blob/main/LICENSE)                       |
| pytorch-image-models   | 0.9.0   | Apache License 2.0                      | Ross Wightman                             | [https://github.com/huggingface/pytorch-image-models/](https://github.com/huggingface/pytorch-image-models/blob/main/LICENSE) |
| nystrom-attention        | 0.0.11  | MIT License                             | Phil Wang                                 | [https://github.com/lucidrains/nystrom-attention/](https://github.com/lucidrains/nystrom-attention/blob/main/LICENSE)         |
| einops                   | 0.6.1   | MIT License                             | Alex Rogozhnikov                          | [https://github.com/arogozhnikov/einops/](https://github.com/arogozhnikov/einops/blob/master/LICENSE)                         |
| hydra-core               | 1.3.2   | MIT License                             | Facebook, Inc. and its affiliates         | [https://github.com/facebookresearch/hydra/](https://github.com/facebookresearch/hydra/blob/main/LICENSE)                     |
| imageio                  | 2.31.1  | BSD-2 Clause                            | Imageio developers                        | [https://github.com/imageio/imageio/](https://github.com/imageio/imageio/blob/master/LICENSE)                                 |
| lifelines                | 0.27.7  | MIT License                             | Cameron Davidson-Pilon                    | [https://github.com/CamDavidsonPilon/lifelines/](https://github.com/CamDavidsonPilon/lifelines/blob/master/LICENSE)           |
| loguru                   | 0.7.0   | MIT License                             | Not specified                             | [https://github.com/Delgan/loguru/](https://github.com/Delgan/loguru/blob/master/LICENSE)                                     |
| openslide-python       | 1.3.0   | GNU LGPL v2.1                           | Free Software Foundation                  | [https://github.com/openslide/openslide-python/](https://github.com/openslide/openslide-python/blob/main/COPYING.LESSER)      |
| PyYAML                   | 6.0.1   | MIT License                             | Ingy döt Net and Kirill Simonov           | [https://github.com/yaml/pyyaml/](https://github.com/yaml/pyyaml/blob/master/LICENSE)                                         |
| scikit-learn             | 1.3.0   | BSD-3 Clause                            | Scikit-learn developers                   | [https://github.com/scikit-learn/scikit-learn/](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING)               |
| torch                    | 1.13.1  | Modified BSD Clause                     | See LICENSE                               | [https://github.com/pytorch/pytorch/](https://github.com/pytorch/pytorch/blob/main/LICENSE)                                   |
| torchvision              | 0.14.1  | BSD-3 Clause                            | Soumith Chintala                          | [https://github.com/pytorch/vision](https://github.com/pytorch/vision)                                                        |
| tqdm                     | 4.66.1  | MIT and Mozilla Public License          | See LICENSE                               | [https://github.com/tqdm/tqdm/](https://github.com/tqdm/tqdm/blob/master/LICENCE)                                             |
| dill                     | 0.37.1  | BSD-3 Clause                            | The Uncertainty Quantification Foundation | [https://github.com/uqfoundation/dill/](https://github.com/uqfoundation/dill/blob/master/LICENSE)                             |

**The following table describe the different datasets from which either features or labels were extracted.**

| Name of the dataset | License           | Dataset home page                                                                                  |
| ------------------- | ----------------- | -------------------------------------------------------------------------------------------------- |
| NCT-CRC-HE-100K     | CC-BY 4.0 License | [https://zenodo.org/record/1214456](https://zenodo.org/record/1214456)                             |
| Camelyon16          | CC0 1.0 License   | [https://camelyon17.grand-challenge.org/Data/](https://camelyon17.grand-challenge.org/Data/)       |
| Camelyon17-WILDS    | CC0 1.0 License   | [https://wilds.stanford.edu/datasets/#camelyon17](https://wilds.stanford.edu/datasets/#camelyon17) |

The results published here are also partly based upon data generated by the TCGA Research Network: https://www.cancer.gov/tcga.


# Citation

If you found our work useful in your research, please consider citing it at:

```
@article{Filiot2023ScalingSSLforHistoWithMIM,
	author       = {Alexandre Filiot and Ridouane Ghermi and Antoine Olivier and Paul Jacob and Lucas Fidon and Alice Mac Kain and Charlie Saillard and Jean-Baptiste Schiratti},
	title        = {Scaling Self-Supervised Learning for Histopathology with Masked Image Modeling},
	elocation-id = {2023.07.21.23292757},
	year         = {2023},
	doi          = {10.1101/2023.07.21.23292757},
	publisher    = {Cold Spring Harbor Laboratory Press},
	url          = {https://www.medrxiv.org/content/early/2023/07/26/2023.07.21.23292757},
	eprint       = {https://www.medrxiv.org/content/early/2023/07/26/2023.07.21.23292757.full.pdf},
	journal      = {medRxiv}
}
```
