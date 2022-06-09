# AI-CUP Competition: STAS Detection

## Environment Setup
Device: single 2080ti with CUDA 10.2 
```bash
pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
python setup.py install
# python setup.py develop
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html
# may be used?
# pip install -U albumentations
# pip install shapely
# pip install ensemble-boxes
```

### Apex Installation:
Following [Swin Transformer for Object Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection), we use apex for mixed precision training by default. To install apex, run:
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```


## STAS Detection Model and Config
put pretrained model in ckpt folder (AI-CUP/ckpt)   
[origin pretrain model](https://github.com/CBNetwork/storage/releases/download/v1.0.0/htc_cbv2_swin_base22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.pth.zip)   
[origin config](https://github.com/VDIGPKU/CBNetV2/blob/main/configs/cbnet/htc_cbv2_swin_base_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.py)     
   
unzip competition model and config in base directory (AI-CUP)  
[competition model and config](https://www.dropbox.com/s/xb5g1pyq6fp1vvj/work_dirs.zip?dl=0)

## STAS Detection Train
please only use single gpu for train and inference 
```bash
python -m torch.distributed.launch tools/train.py 
    configs/cbnet/swin_coco.py 
    --gpus 1 --deterministic --seed 123  
    --work-dir work_dirs/swin_coco
python -m torch.distributed.launch tools/train.py 
    configs/cbnet/swin_custom_fine.py 
    --gpus 1 --deterministic --seed 123  
    --work-dir work_dirs/swin_custom_fine
```

## STAS Detection Evaluate
```bash
python tools/test.py 
    work_dirs/swin_custom_fine/swin_custom_fine.py 
    work_dirs/swin_custom_fine/latest.pth 
    --out result.json 
    --show --show-dir ckpt
```

## Other Links
> **Original CBNet**: See [CBNet: A Novel Composite Backbone Network Architecture for Object Detection](https://github.com/VDIGPKU/CBNet).

## Citation
If you use our code/model, please consider to cite our paper [CBNetV2: A Novel Composite Backbone Network Architecture for Object Detection](http://arxiv.org/abs/2107.00420).
```
@article{liang2021cbnetv2,
  title={CBNetV2: A Composite Backbone Network Architecture for Object Detection}, 
  author={Tingting Liang and Xiaojie Chu and Yudong Liu and Yongtao Wang and Zhi Tang and Wei Chu and Jingdong Chen and Haibing Ling},
  journal={arXiv preprint arXiv:2107.00420},
  year={2021}
}
```
