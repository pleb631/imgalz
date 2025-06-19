# imgalz: A Modular Library for Image Analysis by onnx

![img](https://cdn.jsdelivr.net/gh/pleb631/ImgManager@main/img/2024-01-24-14-31-32.png)

## Usage

See below for quickstart installation and usage examples.
All guidance can refer to full [imgalz docs](https://pleb631.github.io/imgalz)

### Installation

```bash
pip install .[all]
# or
pip install imgalz[all]
```

### example

```python
import cv2
import numpy as np

import imgalz
from imgalz.models.detector import YOLOv5

# Use local path if available, otherwise download from Hugging Face
model = YOLOv5(model_path = "yolov5n.onnx")
# model = YOLOv5("yolov6n.onnx")
im = imgalz.imread("resources/bus.jpg",1)
bboxes = model.detect(im, aug=True)
# plot box on img
for box in bboxes:
    cv2.rectangle(
        im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2
    )

imgalz.cv_imshow("yolov5-det", im)

```

You can refer to the specific usage by [demo](https://github.com/pleb631/imgalz/tree/main/demo)

## Optional Models

### Detector

- YOLOv5/6
- YOLOv8/11
- YOLOv8pose
- YOLOv8seg

### Tracker

- ByteTrack
- Motpy
- NorFair
- OCSort

### Pose

- ViT-Pose

## todo

- add mmpose inference code
- add more tool for image processing
- add more tracker and refact code

## Weights

The ONNX model in the example is exported directly from the official code and can be obtained from the [huggingface](https://huggingface.co/pleb631/onnxmodel).
