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
* Data composition
```
.
+-- data 
|   +-- scene0000
|   |   +-- color
|   |   +-- depth
|   |   +-- pose
|   |   +-- intrinsic
```
