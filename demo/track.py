import imgalz
from imgalz.models.tracker import OcSort, Motpy, NorFair


def main():
    from imgalz.models.detector import YOLOv8

    det = YOLOv8(model_path="yolov8s.onnx")
    track = Motpy()
    vi = imgalz.VideoReader(r"resources\video2.mp4", step=3)
    for frame in vi:
        if frame is not None:
            box = det.detect(frame, iou_thres=0.7)
            track_info = track.track(frame, box)
            bbox_ltrb, ids = (
                track_info["bbox_ltrb"],
                track_info["ids"],
            )
            if len(bbox_ltrb) == 0:
                continue

            for box, id in zip(bbox_ltrb, ids):
                imgalz.draw_bbox(frame, box, 1, id, label_format="{id}")

            if imgalz.cv_imshow("track", frame, delay=1):
                break


if __name__ == "__main__":
    main()
