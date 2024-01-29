# imgalz: A Modular Library for Image Analysis

![2024-01-24-14-31-32](https://cdn.jsdelivr.net/gh/pleb631/ImgManager@main/img/2024-01-24-14-31-32.png)

## Installation

```shell
pip install -r requirements.txt
pip install -e .
```

## Running imgalz

### Predicting on Videos

Run `main.py` to test the tracker on all videos in the `resources` folder.

```bash
python tools/main.py cfg/video_track.py
```

If you want to run it on a webcam, use:

```bash
python tools/main.py cfg/video_track.py --cfg-options data.data_root=webcam
```

### Predicting/Validating on Images

If you want to predict on images, use:

```bash
python tools/main.py cfg/imgdet.py
```

If you want to validate the model trained on COCO in the YOLO dataset, refer to the configuration in `cfg/imgdet_val.py`.

### Validating on a Custom YOLO Dataset/Model

Define the class names using `class_name=[0:'person',1:'car'....]` and pass `class_name` to `hook.detval`.

## Optional Models

### Detectors

- YOLOv5
- YOLOv8
- YOLOv8pose(person)
- YOLOv8seg

### Tracking

- ByteTrack
- Motpy
- NorFair
- OCSort

### Weights

The ONNX model in the example is exported directly from the official code and can be obtained from the [Google Drive](https://drive.google.com/drive/folders/18X7T5nZRrQ3UEZ870_EX91Lf-oLTxKoE?usp=drive_link).

## Other Questions

1. Why did the bounding box (bbox) decoding fail?

   Please check the configuration file and ensure that `pipeline.type` is compatible with the model you intend to use.

