# MDS-DETR: DETR with Masked Duplicate Suppressor

This is the official implementation of the paper **"MDS-DETR: DETR with Masked Duplicate Suppressor"**.

---

## 🛠️ Environment & Installation

### Prerequisites
Our experiments were built on top of the Docker image: `pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel`.
We tested our code on a server with **8x NVIDIA RTX 4090 GPUs** (Total batch size = 16).

### 1. Basic Installation
Clone the repository and install the required packages:
```bash
git clone [https://github.com/your_username/MDS-DETR.git](https://github.com/your_username/MDS-DETR.git)
cd MDS-DETR
pip install -r requirements.txt

