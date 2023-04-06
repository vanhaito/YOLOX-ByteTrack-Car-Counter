# YOLOX-ByteTrack-Car-Counter

Car tracking and car counter implemented with YOLOX, ByteTrack and Pytorch. We can take the ouput of YOLOX feed these object detections into ByteTrack in order to create a highly accurate object tracker. I have created a custom function within the file [detector.py](https://github.com/vanhaito/YOLOX-ByteTrack-Car-Counter/blob/master/detect.py) that can be used to count and keep track of the number of cars detected at a given moment within each video. In can be used to count total cars found or can count number of cars detected.

## Demo of Car Counter in Action
![](https://github.com/vanhaito/YOLOX-ByteTrack-Car-Counter/blob/master/YOLOX/assets/demo.gif)

## Getting started
### Environments
```bash
conda create -n <env name> python=3.7
conda activate <env name>
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install -v -e .
```
### torch2trt
```bash
git clone https://github.com/NVIDIA-AI-IOT/torch2trt 
cd torch2trt
export CUDA_HOME=$CONDA_PREFIX
python setup.py install
```
## Download a pretrained model
Download pretrained yolox_s.pth file: https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth

Copy and paste yolox_s.pth from your downloads folder into the working directory

## Convert model to TensorRT
```bash
python tools/trt.py -n yolox-s -c yolox_s.pth
```

## Runing Car Counter with YOLOX-s
```bash
python detector.py --video <path to video> --output <path to output video> --trt <path to trt model> --device 0
```

## References
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)