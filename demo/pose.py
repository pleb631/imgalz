import imgalz
from imgalz.models.detector import YOLOv8
from imgalz.models.pose import HeatmapPose

def vitpose():


    det = YOLOv8(model_path="yolov8n.onnx")
    pose = HeatmapPose("vitpose-s.onnx")
    im = imgalz.imread("resources/bus.jpg", 1)
    bboxes = det.detect(im)
    bboxes = [box for box in bboxes if box[-1] == 0]
    for box in bboxes:
        x1, y1, x2, y2 = box[:4]
        x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        res, vis = pose.detect_with_label(im, [x, y, w, h], crop_ratio=1.25)

        imgalz.cv_imshow("pose-vis", vis)

def draw_pose_on_coco():
    from imgalz.utils.dataset_info import CocoConfig
    from imgalz.utils.visualization import draw_keypoints
    det = YOLOv8(model_path="yolov8n.onnx")
    pose = HeatmapPose("vitpose-s.onnx")
    im = imgalz.imread("resources/bus.jpg", 1)
    bboxes = det.detect(im)
    bboxes = [box for box in bboxes if box[-1] == 0]
    for box in bboxes:
        x1, y1, x2, y2 = box[:4]
        x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        keypoints, vis = pose.detect_with_label(im, [x, y, w, h], crop_ratio=1.25)

        skeleton = CocoConfig.skeleton
        kpt_color = CocoConfig.kpt_color
        limb_color = CocoConfig.limb_color
        draw_keypoints(im, keypoints, skeleton, kpt_color, limb_color)
    imgalz.cv_imshow("pose-vis", im,size=(512,512))
        
        
if __name__ == "__main__":
    vitpose()
    draw_pose_on_coco()
