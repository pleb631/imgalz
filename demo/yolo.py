import cv2
import numpy as np

import imgalz


# This script supports models exported in the same format as Ultralytics YOLOv5 and YOLOv8,
# including YOLOv5, YOLOv8, YOLOv8-Pose, YOLOv8-Seg, and other compatible variants such as yolov6,yolov11.
# Anchor-based models should use the YOLOv5 class.
# Anchor-free models should use the YOLOv8 class.


def use_yolov5_det():
    from imgalz.models.detector import YOLOv5

    model = YOLOv5(model_path="yolov5n.onnx")
    # model = YOLOv5("yolov6n.onnx")
    # model = YOLOv8("yolov8n.onnx")
    im = imgalz.imread("resources/bus.jpg", 1)
    bboxes = model.detect(im, aug=True)
    # plot box on img
    for box in bboxes:
        cv2.rectangle(
            im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2
        )

    imgalz.cv_imshow("det", im)


def use_yolov8_det():
    from imgalz.models.detector import YOLOv8
    from imgalz.utils.dataset_info import CocoConfig

    model = YOLOv8("yolov8n.onnx")
    # model = YOLOv8("yolo11n.onnx")
    im = imgalz.imread("resources/bus.jpg", 1)
    bboxes = model.detect(im)
    # plot box on img
    for box in bboxes:
        color = imgalz.compute_color_for_labels(box[5])
        label = CocoConfig.category[int(box[5])]
        im = imgalz.draw_bbox(
            im,
            box[:4],
            box[4],
            label,
            line_thickness=2,
            box_color=color,
            label_format="{id}:{score:.2f}",
        )

    imgalz.cv_imshow("yolov8-det", im)


def use_yolo_pose():
    from imgalz.models.detector import YOLOv8Pose

    model = YOLOv8Pose(model_path="yolov8n-pose.onnx")
    im = imgalz.imread("resources/bus.jpg", 1)
    bboxes = model.detect(im)
    # plot box on img
    for box in bboxes:
        cv2.rectangle(
            im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2
        )
        # plt key point
        for kpt in box[6:].reshape(-1, 3):
            cv2.circle(im, (int(kpt[0]), int(kpt[1])), 3, (0, 255, 0), -1)

    imgalz.cv_imshow("yolo-pose", im)


def use_yolo_seg():
    from imgalz.models.detector import YOLOv8Seg

    model = YOLOv8Seg(model_path="yolov8n-seg.onnx")
    im = imgalz.imread("resources/bus.jpg", 1)
    bboxes, masks = model.detect(im)
    # plot box on img
    for box in bboxes:
        cv2.rectangle(
            im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2
        )
    # plot masks on im,masks shape is N,H,W
    colors = [imgalz.palette(i) for i in range(len(masks))]
    im = imgalz.draw_masks(masks,colors,im)

    imgalz.cv_imshow("yolo-seg", im,size=(512,512))


if __name__ == "__main__":
    use_yolov5_det()
    use_yolov8_det()
    use_yolo_seg()
    use_yolo_pose()
