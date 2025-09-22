# EFIFNet

## Installation

```shell
conda create -n EFIFNet python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate EFIFNet
pip install openmim
mim install "mmengine>=0.3.1"
mim install "mmcv>=2.0.0rc1,<2.1.0"
mim install "mmdet>=3.0.0rc3,<3.1.0"
# Install albumentations
pip install -r requirements/albu.txt
# Install MMYOLO , don't forget it!
mim install -v -e .
```
### Datasets
 - **KAIST**  
Link：https://pan.baidu.com/s/1UdwQJH-cHVL91pkMW-ij6g 
Code：ig3y

 - **FLIR-aligned**  
Link：https://pan.baidu.com/s/1ljr8qJYdz-60Lj-iVEHBvg 
Code：uqzs

 - **M3FD**   
Link: https://drive.google.com/drive/folders/1H-oO7bgRuVFYDcMGvxstT1nmy0WF_Y_6

### Weights
 - **M3FD** 
Link: https://pan.baidu.com/s/1I6vxv7B66Ow5nuMqXRPnlw Code: nu2p

### Result
 - **M3FD** 
https://drive.google.com/drive/folders/1dRtojKel1sp0BQv2xmpnnW57cseaANNa?usp=drive_link
