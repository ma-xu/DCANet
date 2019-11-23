# Usage
This file will showcase how to reproduce our DCANet work.

## Requirement
python 3.6+ (tested in 3.7)<br>
PyTorch 1.1 or higher (tested in 1.2 & 1.3)<br>
CUDA 9.0 or higher (tested in 10.0)<br>
Linux (tested in Ubuntu 18.04)<br>
apex (see [NVIDIA/apex](https://github.com/NVIDIA/apex))<br>
DALI (see [NVIDIA/DALI](https://github.com/NVIDIA/DALI))<br>



## Training
### Classification

You can use the following commands to train a classifcation network.<br> 
We are training on 8 Tesla V100 GPUs. If you have 4, change "--nproc_per_node=8" to "--nproc_per_node=4".<br>
For more detail parameters, please see main.py and main_mobile.py files.
```shell
# For normal networks, like ResNet. "--fp16" is for fast training.
python3 -m torch.distributed.launch --nproc_per_node=8 main.py -a dca_se_resnet50 --fp16 --b 32

# For lightweight networks, like MnasNet1_0 and MobileNetV2
python3 -m torch.distributed.launch --nproc_per_node=8 main_mobile.py -a dca_se_mobilenet_v2 --b 64 --opt-level O0
```
### Detection

Please refer to [INSTALL.md](detection/INSTALL.md) for installation and dataset preparation.

Our detection codes are based on mmdetection framework. Thanks to mmdetection. For more details, please see [mmdetection github](https://github.com/open-mmlab/mmdetection).



```shell
# For training.
./tools/dist_train.sh local_configs/{config_file_name}.py 8

# For testing
python3 tools/test.py local_configs/{config_file_name}.py 
work_dirs/{model_path}/epoch_24.pth --gpus 8 --out work_dirs/{save_path}/{results_name}.pkl --eval bbox
```

### Other tools

We also provide a series of related tools, such as visualization, analysis, and parameters/flops counter.

  * For visualization, please see [GAMCAM/main.py](ablation/GAMCAM/main.py).

  * For analysis, please see [Analysis/weights.py](ablation/Analysis/weights.py).

  * For parameters/flops counter, please see [flops_counter.py](flops_counter.py).

