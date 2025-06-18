import cv2
import onnxruntime
import numpy as np
import math
from imgalz.utils.common import auto_download
from imgalz import nms, xywh2xyxy

__all__ = ["YOLOv5"]


def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    # Scales img(bs,3,y,x) by ratio constrained to gs-multiple
    if ratio == 1.0:
        return img
    h, w = img.shape[:2]
    s = (int(w * ratio), int(h * ratio))  # new size
    img = cv2.resize(
        img,
        s,
    )  # resize
    if not same_shape:  # pad/crop img
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    return np.pad(
        img, [(0, h - s[0]), (0, w - s[1]), (0, 0)], constant_values=0.447 * 255
    )  # value = imagenet mean


def clip_boxes(boxes, shape):
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain
        ) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def letterbox(
    im,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, ratio, (dw, dh)


class YOLOv5:
    """
    YOLOv5 object detection model wrapper using ONNX Runtime.

    This class loads a YOLOv5 model in ONNX format and prepares it for inference,
    including setting preprocessing parameters like mean and std normalization.

    Attributes:
        model_path (Union[str, Path]): Path to the ONNX model file.
        mean (List[float]): Mean values for image normalization.
        std (List[float]): Standard deviation values for image normalization.

    """

    @auto_download(category="yolo")
    def __init__(
        self,
        model_path,
        mean=[0, 0, 0],
        std=[1, 1, 1],
    ):

        self.mean = mean
        self.std = std

        self.stride = 32
        self.model = onnxruntime.InferenceSession(
            str(model_path), providers=["CUDAExecutionProvider"]
        )
        self.img_size = self.model.get_inputs()[0].shape[2:]

    def _perprocess(self, image):

        self.img_h, self.img_w = image.shape[:2]

        image = letterbox(
            image, self.img_size, stride=self.stride, auto=False, scaleFill=False
        )[0]

        image = image / 255
        image = np.array((image - self.mean) / self.std, dtype=np.float32)
        image = image.transpose(2, 0, 1)[np.newaxis, :, :, :]
        image = np.ascontiguousarray(image)

        return image

    def _forward(self, image):

        image = image / 255
        image = np.array((image - self.mean) / self.std, dtype=np.float32)
        image = image.transpose(2, 0, 1)[np.newaxis, :, :, :]
        image = np.ascontiguousarray(image)

        ort_inputs = {self.model.get_inputs()[0].name: image}
        ort_outs = self.model.run(None, ort_inputs)

        return ort_outs

    def _forward_augment(self, x):
        img_size = x.shape[:2]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 1, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(np.flip(x, fi) if fi else x, si, True, gs=32)
            ort_outs = self._forward(xi)  # forward
            yi = ort_outs[0].squeeze()

            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return np.concatenate(y, 0)  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        x, y, wh = (
            p[..., 0:1] / scale,
            p[..., 1:2] / scale,
            p[..., 2:4] / scale,
        )  # de-scale
        if flips == 0:
            y = img_size[0] - y  # de-flip ud
        elif flips == 1:
            x = img_size[1] - x  # de-flip lr
        p = np.concatenate((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = 3  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[0] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][:-i, :]  # large
        i = (y[-1].shape[0] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][i:, :]  # small
        return y

    def detect(self, bgr_img, conf_thres=0.3, iou_thres=0.45, aug=False):
        image = bgr_img[:, :, ::-1].copy()
        img_h, img_w = image.shape[:2]
        image = letterbox(
            image, self.img_size, stride=self.stride, auto=False, scaleFill=False
        )[0]
        if not aug:
            ort_outs = self._forward(image)
            boxes = ort_outs[0].squeeze()
        else:
            boxes = self._forward_augment(image)
        boxes = self._post_process(boxes, conf_thres, iou_thres)
        if len(boxes) > 0:
            boxes[:, :4] = scale_boxes(
                self.img_size, boxes[:, :4], [img_h, img_w]
            ).round()

        return boxes

    def _nms(self, x, iou_thres):
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 300000  # maximum number of boxes into torchvision.ops.nms()
        max_det = 300
        agnostic = False

        x = x[x[:, 4].argsort()[::-1][:max_nms]]

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 5]
        i = nms(boxes, scores, iou_thres)
        i = i[:max_det]

        x = x[i]
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        return x

    def _post_process(self, boxes, conf_thres, iou_thres):

        xc = boxes[:, 4] > conf_thres
        boxes = boxes[xc, :]
        if len(boxes) == 0:
            return []

        boxes[:, 5:] *= boxes[:, 4:5]

        box = xywh2xyxy(boxes[:, :4])

        conf, j = np.max(boxes[:, 5:], 1, keepdims=True), np.argmax(
            boxes[:, 5:], 1, keepdims=True
        )
        x = np.concatenate((box, conf, j), 1)[conf.reshape(-1) > conf_thres]

        if len(boxes) == 0:
            return []

        return self._nms(x, iou_thres)
