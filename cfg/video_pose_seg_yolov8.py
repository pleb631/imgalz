pipeline = [
    dict(
        type="YOLOv8_seg",
        backend="onnx",
        model_path=r"weights/yolov8n-seg.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        conf_thres=0.25,
        iou_thres=0.45,
        filter_class=[0],
    ),
    
    dict(
        type="YOLOv8_pose",
        backend="onnx",
        model_path=r"weights/yolov8n-pose.onnx",
        filter_class=[],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        conf_thres=0.3,
        iou_thres=0.3,
    ),
    dict(
        type="ByteTrack",
    ),
]

data = dict(type="VideoDataset", data_root=r"resources/3.mp4")


hooks = [
    dict(
        type="run_model",
        backend="image",
        priority=50,
    ),

    dict(
        type="plot_seg",
        priority=60,
        key=0
    ),
    dict(
        type="plot_box",
        priority=60,
        key = 2,
        plot_id = True
    ),
    dict(
        type="plot_kpt",
        priority=60,
        key=1,
    ),
    dict(type="exporter",priority=90,),
]


runner = dict()
