# 3D **R**econstruction with 3D **I**nstance segmentation and 3D **S**cene graph generation


## Requirements
* Ubuntu 16.04+
* Python 3.6
* CUDA 10.0~10.2
* Typing below commands to setting environments
```
    conda create -n recon3d python=3.6
    conda activate recon3d
    bash requirements.sh
```

## Dataset
__1. ScanNetV2 Dataset__
* RGB-D video dataset containing 2.5 million views in more than 1500 scans, annotated with 3D camera poses, provided by the [ScanNet](https://github.com/ScanNet/ScanNet)

__2. Your Own Dataset__
* You can get RGB-D image using realsense camera, and extract camera pose through [ORB SLAM2](https://github.com/raulmur/ORB_SLAM2).

__3. Data composition__
* Sample Datasets are available to download here [lab_kitchen17](https://drive.google.com/file/d/1pWWgPiP2Cvt7CoiGwxehriP0X2PYCFs1/view?usp=sharing), [scene0000]().
```
.
+-- data 
|   +-- scene0000 (ScanNetv2)
|   |   +-- color
|   |   +-- depth
|   |   +-- pose
|   |   +-- intrinsic
|   +-- labkitchen17 (Your Own Dataset)
|   |   +-- color
|   |   +-- depth
|   |   +-- pose
|   |   +-- intrinsic
```


## Example of usage
* Test performance on scannet data (scene0000)
```
    python main.py --data_root [DATA_ROOT_DIR] --scene scene0000 --is_scannet --trained_model=weights/yolact_plus_resnet50_54_800000.pth --score_threshold=0.7 --top_k=15
```
* Test performance on your own data (lab_kitchen17)
```
    python main.py --data_root [DATA_ROOT_DIR] --scene lab_kitchen17 --trained_model=weights/yolact_plus_resnet50_54_800000.pth --score_threshold=0.7 --top_k=15
```




