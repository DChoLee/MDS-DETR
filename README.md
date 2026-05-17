# MDS-DETR: DETR with Masked Duplicate Suppressor

This is the official implementation of the paper **"MDS-DETR: DETR with Masked Duplicate Suppressor"**.

---

## 🛠️ Environment & Installation

### 1. Prerequisites
Our experiments were built on top of the Docker image: `pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel`.

We tested our code on a server with 8x NVIDIA RTX 4090 GPUs.

### 2. Basic Installation
Clone the repository and install the required packages:
```bash
git clone https://github.com/DChoLee/MDS-DETR.git
cd MDS-DETR
pip install -r requirements.txt
```
### 3. Multi-Scale Deformable Attention CUDA Compilation
Please follow the installation guide from (https://github.com/fundamentalvision/Deformable-DETR)

### 4. Swin-Transformer Backbone Setup
If you want to use the Swin-Transformer backbone, please install mmcv and mmdet as below.
```bash
pip install openmim
mim install mmcv-full==1.7.2
pip install mmdet==2.28.1
```

### 💾 [Checkpoints & Results]

We provide the pre-trained checkpoints trained on the MS-COCO dataset.
You can download them from the Google Drive links below:

* MDS-DETR (ResNet-50)   | 12e | 300 Q | https://drive.google.com/file/d/1wRq-NmPTnAT0IWKjyI2TXf_BgKDjRznU/view?usp=drive_link
* MDS-DETR (ResNet-50)   | 12e | 900 Q | https://drive.google.com/file/d/1fkXL3ZEW9Aa0RJvXzg4_TqWMtDk50SZN/view?usp=drive_link
* MDS-DETR (ResNet-50)   | 24e | 900 Q | https://drive.google.com/file/d/1L0byhxsoHmOgWCqh-_qmeapNZs4C7UFk/view?usp=drive_link
* MDS-DETR (Swin-Large)  | 12e | 900 Q | https://drive.google.com/file/d/1vdCwnjSIR1k8g1aSHF6cQ2MCJgUAofQM/view?usp=drive_link


# 🚀 [Training & Evaluation]

### 1. Dataset Preparation
Specify your MSCOCO dataset path inside your shell scripts (.sh).
Modify the coco_path variable to point to your local COCO dataset directory:
```bash
coco_path="../../data/coco"  # <-- Change this to your actual COCO path
```

### 2. Run Scripts
To train/test a model (MDS-DETR-ResNet50-12e-300Q):

```bash
# Train a model from scratch
sh ./train_mds_r50_300_12e.sh

# Evaluate a model with checkpoint.
sh ./eval_mds_r50_300_12e.sh 
```

### 🙏 [Acknowledgements]
Our code is heavily based on the official implementations of the following works:

* Deformable-DETR (https://github.com/fundamentalvision/Deformable-DETR)
* MS-DETR (https://github.com/Atten4Vis/MS-DETR)
