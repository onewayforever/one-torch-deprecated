# Object Detection

## Dataset
use COCO2014

the dataset directory:
\#ls
coco.names  images  labels  train2014_tiny.txt  train2014.txt  val2014_tiny.txt  val2014.txt  yolo.data

## Yolov3 Code
Porting from https://github.com/eriklindernoren/PyTorch-YOLOv3

## Run

* with tiny data to check code
python3 ../../one_torch_main.py -a train  --src yolov3_experiment.py -c --hparam tiny_data.yml

* with full data to run
python3 ../../one_torch_main.py -a train  --src yolov3_experiment.py -c --hparam full_data.yml
