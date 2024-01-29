pipeline = [
    dict(
        type="YOLOv8",
        backend="onnx",
        model_path=r"weights/yolov8m.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        conf_thres=0.1,
        iou_thres=0.3,
    ),
]

data = dict(type="YoloDataset", data_root=r"resources/coco128-7")


hooks = [
    dict(
        type="run_model",
        backend="image",
        priority=50,
    ),
    dict(
        type="deteval",
        priority=55,
    ),
]

runner = dict()
