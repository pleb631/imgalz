pipeline = [
    dict(
        type="YOLOv8",
        backend="onnx",
        model_path=r"weights/yolov8n.onnx",
        filter_class=[],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        conf_thres=0.3,
        iou_thres=0.45,
    ),
    dict(
        type="ByteTrack",
    ),
]

data = dict(type="VideoDataset", data_root=r"resources/")


hooks = [
    dict(
        type="run_model",
        backend="image",
        priority=50,
    ),
    dict(
        type="plot_box",
        plot_id=True,
        label_format="{id}",
        priority=60,
    ),
    dict(type="exporter",priority=90,),
]
runner = dict()
