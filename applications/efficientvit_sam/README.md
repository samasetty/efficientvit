# Run Benchmarks on BDD100K Dataset

## Setup
The BDD100K dataset is a large-scale dataset for driving video analysis. Follow these steps to evaluate EfficientViT-SAM for segmentation tasks on BDD100K, or run our code on this <colab notebook>.

### Steps to Prepare BDD100K Dataset
1. Download all the images from the **train**, **validation**, and **test** splits of the BDD100K dataset <hyperlink>.
2. Combine all images into a single folder located at:
 ```bash
 ~/dataset/bdd100k/images/10k/all
```
This folder should contain all the 10K images from the dataset.
3. Download and Format Annotations
* Download the file bdd100k_ins_seg_labels_trainval.zip from the BDD100K website and extract it.
* Run the following command to convert the annotation file into COCO-style format:

```bash
python3 -m bdd100k.label.to_coco -m ins_seg \
    ${in_path} -o ${~/bdd100k_coco_all}
```
Download the file bdd100k_ins_seg_labels_trainval.zip from the BDD100K website and extract it.
4. Place the generated COCO-style annotation file (bdd100k_coco_all.json) in the appropriate directory for evaluation.

Expected directory structure:
```bash
bdd100k
├── images
│   ├── 10k
│       ├── all  # Combined train, validation, and test images
├── annotations
│   ├── bdd100k_coco_all.json
```

## Pretrained EfficientViT-SAM Models

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

## Benchmarking EfficientViT-SAM on BDD100K
To measure the latency of EfficientViT-SAM on the BDD100K dataset, use the commands below. Ensure the `--measure_latency` flag is specified to get latency results for each model variant.
* EfficientViT-SAM
```bash
python /root/efficientvit/applications/efficientvit_sam/eval_efficientvit_sam_model.py \
    --model ${model} \
    --prompt_type box \
    --dataset bdd100k \
    --image_root /content/drive/MyDrive/bdd100k_train/images/10k/all \
    --annotation_json_file ~/ins_seg_val_cocofmt.json \
    --weight_url assets/checkpoints/efficientvit_sam/{model_weights} \
    --measure_latency
```

# expected results: 
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.478
```

```bash
# LVIS
torchrun --nproc_per_node=8 applications/efficientvit_sam/eval_efficientvit_sam_model.py --dataset lvis --image_root ~/dataset/coco --annotation_json_file ~/dataset/coco/annotations/lvis_v1_val.json --model efficientvit-sam-xl1 --prompt_type box_from_detector --source_json_file ~/dataset/coco/source_json_file/lvis_vitdet.json

# expected results: 
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=300 catIds=all] = 0.444
```

## Visualization

Please run [demo_efficientvit_sam_model.py](demo_efficientvit_sam_model.py) to visualize our segment anything models.

Example:

```bash
# segment everything
python applications/efficientvit_sam/demo_efficientvit_sam_model.py --model efficientvit-sam-xl1 --mode all

# prompt with points
python applications/efficientvit_sam/demo_efficientvit_sam_model.py --model efficientvit-sam-xl1 --mode point

# prompt with box
python applications/efficientvit_sam/demo_efficientvit_sam_model.py --model efficientvit-sam-xl1 --mode box --box "[150,70,640,400]"

```

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
