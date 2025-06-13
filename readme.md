# imgalz: A Modular Library for Image Analysis by onnx

![img](https://cdn.jsdelivr.net/gh/pleb631/ImgManager@main/img/2024-01-24-14-31-32.png)

## Installation

```bash
pip install .
# or
pip install imgalz
```

## Usage

```python
import cv2
import numpy as np

from imgalz.utils import cv_imshow,imread
from imgalz.models.detector import YOLOv5

# file local path or auto download from huggingface
model = YOLOv5(model_path = "yolov5n.onnx")
# model = YOLOv5("yolov6n.onnx")
im = imread("resources/bus.jpg",1)
bboxes = model.detect(im, aug=True)
# plot box on img
for box in bboxes:
    cv2.rectangle(
        im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2
    )

cv_imshow("yolov5-det", im)

```

You can refer to the specific usage by [demo](demo/yolo.py)

## Optional Models

### Detectors

- YOLOv5/6
- YOLOv8/11
- YOLOv8pose
- YOLOv8seg

### Tracking

- ByteTrack
- Motpy
- NorFair
- OCSort

## todo

- add mmpose and vitpose inference code

## Weights

The ONNX model in the example is exported directly from the official code and can be obtained from the [huggingface](https://huggingface.co/pleb631/onnxmodel).
