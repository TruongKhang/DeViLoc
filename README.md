# DeViLoc: Learning to Produce Semi-dense Correspondences for Visual Localization
Our paper is accepted for an **oral presentation (top 3.3%)** at CVPR 2024. PDF is available at [arxiv](https://arxiv.org/abs/2402.08359)

![Alt Text](demo_deviloc_short.gif)

## Requirements
All experiments were implemented under Ubuntu 16.04 and NVIDIA TESLA V100/NVIDIA GeForce RTX 3090 with the cuda version of 11.3/11.6.

To setup working environment, you need to create a virtual Python environment using Conda and then install the required packages using pip

    conda create -n dvl_env python=3.8 -c anaconda
    conda activate dvl_env
    conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
    pip install -r requirements.txt

Next, clone the feature matching model [TopicFM](https://github.com/TruongKhang/TopicFM.git) and put it in the `third_party` folder

    mkdir third_party/feat_matcher && cd third_party/feat_matcher
    git clone https://github.com/TruongKhang/TopicFM.git
    cd TopicFM && git checkout dev_2

## Training
We train our network on the [MegaDepth](https://www.cs.cornell.edu/projects/megadepth/) dataset. 
Note that you just need to download the file `MegaDepth v1 Dataset (tar.gz, 199 GB)` from the official website.

We recommend storing your training data in `data/megadepth` folder. The structure of this folder looks like this:
```
data
├── megadepth
    ├── phoenix # this folder is uncompressed from the file .tar.gz
    ├── preprocessed
        ├── scene_points3d
        └── megadepth_2d3d_q500ov0.2-1.0covis3-15.npy
```

The data in the `preprocessed` folder is uploaded [here](https://drive.google.com/drive/folders/1CxLrXnt5JpWe9WdweCs3dE22Eg6mFvZA?usp=sharing)

After downloading all data, you can change some training parameters in `scripts/train_megadepth.sh` and then run this script to train models

    bash scripts/train_megadepth.sh configs/megadepth.yml


## Evaluation

### 7scenes
Download the dataset and save it into the folder `data`. This script are provided by [HLoc](https://github.com/cvg/Hierarchical-Localization).

    bash scripts/download_7scenes.sh

Run the evaluation code as follows:

    python evaluate.py configs/se7scenes.yml --ckpt_path pretrained/deviloc_weights.ckpt

### Cambridge Landmarks

    bash scripts/download_cambridge.sh
    python evaluate.py configs/cambridge.yml --ckpt_path pretrained/deviloc_weights.ckpt

### Long-term Visual Localization Benchmarks
The estimated camera poses of these datasets are evaluated on this [benchmark website](https://www.visuallocalization.net/benchmark/).

**Downloading files contains pairs of query-reference images**
For the Aachen, Robotcar, and CMU datasets, it is required to select *K* reference images per a query image for localization.
First, you need to download each dataset using the provided script in `scripts/download_aachen/robotcar/cmu.sh`

Next, please download our preprocessed pair files [here](https://drive.google.com/file/d/1waAnXuPnoa4Nzjat0xE3S5GYMTIMM-_p/view?usp=sharing) and put each of them into the dataset folder like this:
```commandline
data
├── aachen
    ├── pairs
        ├── pairs-query-netvlad50.txt

├── RobotCarSeasons
    ├── pairs-query-cosplace20.txt
├── Extended-CMU-Seasons
    ├── slice2
        ├── pairs-query-cosplace10.txt
    .
    .
    .
    ├── slice21
        ├── pairs-query-cosplace10.txt
```

#### Aachen Day-Night

    bash scripts/download_aachen.sh
    python evaluate.py configs/aachen.yml --ckpt_path pretrained/deviloc_weights.ckpt --out_file aachen_eval_deviloc.txt --covis_clustering

#### Robotcar-Seasons

    bash scripts/download_robotcar.sh
    python evaluate.py configs/robotcar.yml --ckpt_path pretrained/deviloc_weights.ckpt --out_file robotcar_eval_deviloc.txt

#### Extended CMU-Seasons

    bash scripts/download_cmu.sh
    python evaluate.py configs/cmu.yml --ckpt_path pretrained/deviloc_weights.ckpt --out_file cmu_eval_deviloc.txt

## Citation

```commandline
@inproceedings{giang2024learning,
  title={Learning to Produce Semi-dense Correspondences for Visual Localization},
  author={Giang, Khang Truong and Song, Soohwan and Jo, Sungho},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  year={2024}
}
```
## Copyright
This work is affiliated with NMAIL-KAIST, and its intellectual property belongs to NMAIL-KAIST.

```commandline
Copyright NMAIL-KAIST. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
