pipeline = [
    dict(
        type="YOLOv8_pose",
        backend="onnx",
        model_path=r"weights/yolov8n-pose.onnx",
        filter_class=[],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        conf_thres=0.3,
        iou_thres=0.45,
    ),
    dict(
        type="ByteTrack",
    ),
]

data = dict(type="VideoDataset", data_root=r"resources\3.mp4")


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
    dict(
        type="plot_kpt",
        priority=61,
        key=0
    ),
    dict(type="exporter",priority=90,),
]
runner = dict()
