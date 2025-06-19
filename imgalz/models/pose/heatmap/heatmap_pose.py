import onnxruntime
import numpy as np
import cv2

from imgalz.utils.common import auto_download

from .utils import keypoints_from_heatmaps, vis_pose_result

__all__ = ["HeatmapPose"]


class HeatmapPose:
    @auto_download(category="pose")
    def __init__(
        self, model_path, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ):
        self.model = onnxruntime.InferenceSession(
            str(model_path), providers=["CUDAExecutionProvider"]
        )
        self.img_size = self.model.get_inputs()[0].shape[2:4]
        self.mean = mean
        self.std = std

    def _perprocess(self, bgr_img):

        ori_w, ori_h = bgr_img.shape[1], bgr_img.shape[0]
        r = min(self.img_size[1] / ori_w, self.img_size[0] / ori_h)
        new_unpad = int(round(ori_w * r)), int(round(ori_h * r))
        dw, dh = self.img_size[1] - new_unpad[0], self.img_size[0] - new_unpad[1]
        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if r > 1:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LINEAR
        img_resize = cv2.resize(bgr_img.copy(), new_unpad, interpolation=interpolation)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        assert top >= 0 and bottom >= 0 and left >= 0 and right >= 0
        image = cv2.copyMakeBorder(
            img_resize, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )  # add border

        image = image[:, :, ::-1]

        image = image / 255
        image = np.array((image - self.mean) / self.std, dtype=np.float32)
        image = image.transpose(2, 0, 1)[np.newaxis, :, :, :]
        image = np.ascontiguousarray(image)

        return image

    def detect(self, img_crop):

        input_name = self.model.get_inputs()[0].name
        img_perprocessed = self._perprocess(img_crop.copy())
        out = self.model.run(None, {input_name: img_perprocessed})[0]

        preds, prob = self._post_process(out, img_crop)

        all_preds = np.zeros((1, preds.shape[1], 3), dtype=np.float32)
        all_preds[:, :, 0:2] = preds[:, :, 0:2]
        all_preds[:, :, 2:3] = prob

        vis = vis_pose_result(img_crop, all_preds, radius=1)
        return all_preds, vis

    def _post_process(self, output, ori_img):

        h, w, _ = ori_img.shape

        c = np.array([[w * 0.5, h * 0.5]], dtype=np.float32)

        aspect_ratio = 1.0 * self.img_size[1] / self.img_size[0]

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        s = np.array([[w, h]], dtype=np.float32)

        preds, maxvals = keypoints_from_heatmaps(output, c, s)

        return preds, maxvals

    def detect_with_label(self, image, box_xywh, is_normalized=False, crop_ratio=1.25):
        h0, w0, _ = image.shape
        x, y, w, h = box_xywh[:4]
        w, h = w * crop_ratio, h * crop_ratio
        if is_normalized:
            x, y, w, h = x * w0, y * h0, w * w0, h * h0

        xmin, ymin = round(x - w / 2), round(y - h // 2)
        xmax, ymax = round(x + w / 2), round(y + h / 2)
        xmin, ymin = max(xmin, 0), max(ymin, 0)
        xmax, ymax = min(w0, xmax), min(h0, ymax)

        img_crop = image[ymin:ymax, xmin:xmax, :].copy()
        all_preds, vis = self.detect(img_crop)
        all_preds = all_preds[0] +np.array([[xmin, ymin,0]])
        return all_preds, vis
