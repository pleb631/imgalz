from imgalz import cv_imshow, VideoReader
import numpy as np
from imgalz.models.tracker import OcSort,Motpy,NorFair
from imgalz.utils.plot import plot_box


def main():
    from imgalz.models.detector import YOLOv8

    det = YOLOv8(model_path="yolov8s.onnx")
    track = Motpy()
    vi = VideoReader(r"resources\video2.mp4", step=3)
    for frame in vi:
        if frame is not None:
            box = det.detect(frame, iou_thres=0.7)
            track_info = track.track(frame, box)
            bbox_ltrb, ids, scores = (
                track_info["bbox_ltrb"],
                track_info["ids"],
                track_info["scores"],
            )
            if len(bbox_ltrb)==0:
                continue
            bbox_ltrb = np.concatenate(
                (bbox_ltrb, scores.reshape(-1, 1), ids.reshape(-1, 1)), axis=1
            )
            for box in bbox_ltrb:
                plot_box.plt_bbox(frame, box, label_format="{id}")

            if cv_imshow("yolov5-det", frame, delay=1):
                break


main()
