pipeline = [
    dict(
        type="YOLOv5",
        backend="onnx",
        model_path=r"weights/yolov5s.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        conf_thres=0.25,
        iou_thres=0.45,
    ),
]

data = dict(type="ImgDataset", data_root=r"resources/")


hooks = [
    dict(
        type="run_model",
        backend="image",
        priority=50,
    ),
    dict(
        type="plot_box",
        priority=60,
    ),
    dict(type="exporter",priority=90,),
]


runner = dict()
