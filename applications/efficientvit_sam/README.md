# Run Benchmarks on BDD100K Dataset

## Setup

The BDD100K dataset is a large-scale dataset for driving video analysis. Follow these steps to evaluate EfficientViT-SAM for segmentation tasks on BDD100K.

### Prerequisites

Install the necessary libraries with the following commands:

```bash
pip install bdd100k
pip install pydantic==1.10.12
```

### Steps to Prepare BDD100K Dataset
1. [Download all the images](https://dl.cv.ethz.ch/bdd100k/data/) from the **train**, **validation**, and **test** splits of the BDD100K dataset.
2. Combine all images into a single folder located at:
 ```bash
 ${efficientvit_repo}/assets/datasets/bdd100k/images/10k/all
```
3. Download and Format Annotations
* Download the file `bdd100k_ins_seg_labels_trainval.zip` from the BDD100K website and extract it.
* Run the following command to convert the annotation file into COCO-style format:

```bash
python3 -m bdd100k.label.to_coco -m ins_seg \
    ${in_path} -o ${efficientvit_repo}/assets/datasets/bdd100k/annotations/bdd100k_coco_all.json
```
Place the generated COCO-style annotation file (`bdd100k_coco_all.json`) in the appropriate directory for evaluation.

Expected directory structure:
```bash
bdd100k
├── images
│   ├── 10k
│       ├── all  # Combined train, validation, and test images
├── annotations
│   ├── bdd100k_coco_all.json
```

### Pretrained EfficientViT-SAM Models

Latency/Throughput is measured on NVIDIA Jetson AGX Orin, and NVIDIA A100 GPU with TensorRT, fp16. Data transfer time is included. Please put the downloaded checkpoints under *${efficientvit_repo}/assets/checkpoints/efficientvit_sam/*

| Model         |  Resolution | COCO mAP | LVIS mAP | Params |  MACs | Jetson Orin Latency (bs1) | A100 Throughput (bs16) | Checkpoint |
|----------------------|:----------:|:----------:|:---------:|:------------:|:---------:|:---------:|:------------:|:------------:|
| EfficientViT-SAM-L0 | 512x512 | 45.7 | 41.8 | 34.8M  | 35G | 8.2ms  | 762 images/s | [link](https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_l0.pt) |
| EfficientViT-SAM-L1 | 512x512 | 46.2 | 42.1 | 47.7M | 49G |  10.2ms | 638 images/s | [link](https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_l1.pt) |
| EfficientViT-SAM-L2 | 512x512 | 46.6 | 42.7 | 61.3M | 69G |  12.9ms | 538 images/s  | [link](https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_l2.pt) |
| EfficientViT-SAM-XL0 | 1024x1024 | 47.5 | 43.9 | 117.0M | 185G | 22.5ms  | 278 images/s | [link](https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_xl0.pt) |
| EfficientViT-SAM-XL1 | 1024x1024 | 47.8 | 44.4 | 203.3M | 322G | 37.2ms  | 182 images/s | [link](https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_xl1.pt) |

<p align="center">
<b> Table1: Summary of All EfficientViT-SAM Variants.</b> COCO mAP and LVIS mAP are measured using ViTDet's predicted bounding boxes as the prompt. End-to-end Jetson Orin latency and A100 throughput are measured with TensorRT and fp16.
</p>

### Pretrained SAM Models

[Follow these instructions](<https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints>) to download the SAM model checkpoints, and place them under *${efficientvit_repo}/assets/checkpoints/sam/*. Our results were obtained with the SAM-ViT-H model.

## Benchmarking EfficientViT-SAM on BDD100K
To measure the latency of EfficientViT-SAM on the BDD100K dataset, use the commands below. Ensure the `--measure_latency` flag is specified to get latency results for each model variant.
* EfficientViT-SAM
```bash
python /root/efficientvit/applications/efficientvit_sam/eval_efficientvit_sam_model.py \
    --model ${model} \
    --prompt_type box \
    --dataset bdd100k \
    --image_root ${efficientvit_repo}/assets/datasets/bdd100k/images/10k/all \
    --annotation_json_file ${efficientvit_repo}/assets/datasets/bdd100k/annotations/bdd100k_coco_all.json \
    --weight_url assets/checkpoints/efficientvit_sam/{model_weights} \
    --measure_latency
```

* SAM
```bash
python /root/efficientvit/applications/efficientvit_sam/eval_sam_model.py \
    --model ${model} \
    --prompt_type box \
    --dataset bdd100k \
    --image_root ${efficientvit_repo}/assets/datasets/bdd100k/images/10k/all \
    --annotation_json_file ${efficientvit_repo}/assets/datasets/bdd100k/annotations/bdd100k_coco_all.json \
    --weight_url assets/checkpoints/sam/{model_weights} \
    --measure_latency
```

## Expected Results: 

### Table: Performance Comparison of EfficientViT-SAM Variants and the Original SAM Model on the BDD100K Dataset

| Model                  | mIoU (All) | mIoU (Small) | mIoU (Medium) | mIoU (Large) | Latency (ms) | Throughput (images/s) |
|------------------------|------------|--------------|---------------|--------------|--------------|-----------------------|
| SAM-ViT-H             | 41.233     | 42.985       | 40.668        | 36.026       | 4445.94      | 2.26                  |
| EfficientViT-SAM-L0   | 52.696     | 57.938       | 49.920        | 39.384       | 527.97       | 19.04                 |
| EfficientViT-SAM-L1   | 52.755     | 57.903       | 50.127        | 39.480       | 540.31       | 18.60                 |
| EfficientViT-SAM-L2   | 52.916     | 57.814       | 50.142        | 40.860       | 586.49       | 17.14                 |
| EfficientViT-SAM-XL0  | 49.148     | 53.594       | 45.746        | 40.080       | 622.94       | 16.13                 |
| EfficientViT-SAM-XL1  | 48.927     | 52.871       | 45.830        | 41.050       | 735.36       | 13.67                 |


## Visualization
The scripts we used to generate demo videos for EfficientViT-SAM and SAM on the BDD100K dataset are available in this [![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1F5d_fRJu8v-aIQMJYC5v7R616woqvgn5?usp=sharing) notebook. The generated demo videos showcasing EfficientViT-SAM and SAM can be viewed [here](https://drive.google.com/drive/folders/1_LGqplLApgzD2mTX7z3FHZLy96hmy4-B?usp=sharing).

## Reference

```bibtex
@inproceedings{cai2023efficientvit,
  title={Efficientvit: Lightweight multi-scale attention for high-resolution dense prediction},
  author={Cai, Han and Li, Junyan and Hu, Muyan and Gan, Chuang and Han, Song},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={17302--17313},
  year={2023}
}

@article{zhang2024efficientvit,
  title={EfficientViT-SAM: Accelerated Segment Anything Model Without Performance Loss},
  author={Zhang, Zhuoyang and Cai, Han and Han, Song},
  journal={arXiv preprint arXiv:2402.05008},
  year={2024}
}
```
