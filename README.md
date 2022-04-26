# YOLOX-ByteTrack-Car-Counter

Object tracking and car counter implemented with YOLOX, ByteTrack and Pytorch. We can take the ouput of YOLOX feed these object detections into ByteTrack in order to create a highly accurate object tracker. I have created a custom function within the file [detector.py](https://github.com/vanhaito/YOLOX-ByteTrack-Car-Counter/blob/master/detector.py) that can be used to count and keep track of the number of cars detected at a given moment within each video. In can be used to count total cars found or can count number of cars detected.

## Demo of Car Counter in Action
![](https://github.com/vanhaito/YOLOX-ByteTrack-Car-Counter/blob/master/YOLOX/assets/demo.gif)

## Getting started
### Pip
```bash
pip install -r requirements.txt
cd YOLOX
pip install -v -e .
```
### Nvidia Driver
Make sure to use CUDA Toolkit version 11.2 as it is the proper version for the Torch version used in this repository: https://developer.nvidia.com/cuda-11.2.0-download-archive

### torch2trt
Clone this repository and install: https://github.com/NVIDIA-AI-IOT/torch2trt 

## Download a pretrained model
Download pretrained yolox_s.pth file: https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth

Copy and paste yolox_s.pth from your downloads folder into the 'YOLOX' folder of this repository.

## Conver model to TensorRT
```bash
python tools/trt.py -n yolox-s -c yolox_s.pth
```

## Runing Car Counter with YOLOX-s
In file [detector.py](https://github.com/vanhaito/YOLOX-ByteTrack-Car-Counter/blob/master/detector.py) you need to replace the file video name in line 131:
![](https://github.com/vanhaito/YOLOX-ByteTrack-Car-Counter/blob/master/YOLOX/assets/filename.png). 

Then run this command:
```bash
python detector.py
```

## References
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)

