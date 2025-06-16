import numpy as np

from .yolov5 import YOLOv5, scale_boxes, letterbox, scale_img
from imgalz import xywh2xyxy


__all__ = ["YOLOv8"]


class YOLOv8(YOLOv5):
    """
    YOLOv8 object detection model wrapper extending YOLOv5.

    Inherits basic ONNX loading and preprocessing from YOLOv5,
    and adds support for specifying the number of classes.

    Attributes:
        model_path (Union[str, Path]): Path to the ONNX model file.
        mean (List[float], optional): Mean values for normalization. Defaults to [0, 0, 0].
        std (List[float], optional): Standard deviation values for normalization. Defaults to [1, 1, 1].
        nc (int, optional): Number of classes. Defaults to 80.
    """

    def __init__(
        self,
        model_path,
        mean=[0, 0, 0],
        std=[1, 1, 1],
        nc=80,
    ):
        super(YOLOv8, self).__init__(model_path=model_path, mean=mean, std=std)

        self.nc = nc

    def detect(self, bgr_img, conf_thres=0.3, iou_thres=0.45, aug=False):
        image = bgr_img[:, :, ::-1].copy()
        img_h, img_w = image.shape[:2]
        image = letterbox(
            image, self.img_size, stride=self.stride, auto=False, scaleFill=False
        )[0]
        if not aug:
            ort_outs = self._forward(image)
            boxes = ort_outs[0].squeeze().T
        else:
            boxes = self._forward_augment(image)

        boxes = self._post_process(boxes, conf_thres, iou_thres)
        if len(boxes) > 0:
            boxes[:, :4] = scale_boxes(
                self.img_size, boxes[:, :4], [img_h, img_w]
            ).round()

        return boxes

    def _post_process(self, boxes, conf_thres=0.30, iou_thres=0.45):
        xc = boxes[:, 4 : 4 + self.nc].max(1) > conf_thres
        boxes = boxes[xc, :]

        box, clss, mask, _ = np.split(
            boxes, [4, 4 + self.nc, boxes.shape[1] + 1], axis=1
        )

        box = xywh2xyxy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        conf = clss.max(1, keepdims=True)
        j = clss.argmax(1, keepdims=True)
        x = np.concatenate((box, conf, j, mask), 1)[conf.reshape(-1) > conf_thres]

        if len(boxes) == 0:
            return []

        return self._nms(x, iou_thres)

    def _forward_augment(self, x):
        img_size = x.shape[:2]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 1, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(np.flip(x, fi) if fi else x, si, True, gs=32)
            ort_outs = self._forward(xi)  # forward
            yi = ort_outs[0].squeeze().T

            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
            # break
        y = self._clip_augmented(y)  # clip augmented tails
        return np.concatenate(y, 0)  # augmented inference, train
