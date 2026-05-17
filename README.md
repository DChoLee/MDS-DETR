Implementation of 'MDS-DETR: DETR with Masked Duplicate Suppressor'

Our codes are based on official implementation github of Deformable-DETR,H-DETR and MS-DETR
Our experiments are built on docker image      pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel      

Install requirements.txt
For MultiScaleDeformableAttention, follow the installation guide from (https://github.com/fundamentalvision/Deformable-DETR)


For Swin-Transformer backbone, install below
pip install openmim
mim install mmcv-full ( 1.7.2)
pip install mmdet==2.28.1

Our code was tested on 8x NVIDIA RTX 4090 server, with 16 batches.

Checkpoints
r50_300_12e : https://drive.google.com/file/d/1wRq-NmPTnAT0IWKjyI2TXf_BgKDjRznU/view?usp=drive_link
r50_900_12e : https://drive.google.com/file/d/1fkXL3ZEW9Aa0RJvXzg4_TqWMtDk50SZN/view?usp=drive_link
r50_900_24e : https://drive.google.com/file/d/1L0byhxsoHmOgWCqh-_qmeapNZs4C7UFk/view?usp=drive_link
swinl_900_12e : https://drive.google.com/file/d/1vdCwnjSIR1k8g1aSHF6cQ2MCJgUAofQM/view?usp=drive_link

For train and test
1. Specify your MSCOCO dataset path (e.g. in shell script, coco_path=../../data/coco  --> Your coco path)
2. Run sh files. For test, checkpoints should be in same location.