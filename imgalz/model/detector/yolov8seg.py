import cv2
import numpy as np


from imgalz.model import MODELS
from .yolov8 import YOLOv8,scale_boxes


def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))
    return sig


def scale_masks(masks, im0_shape, ratio_pad=None):
    """
    Takes a mask, and resizes it to the original image size.

    Args:
        masks (np.ndarray): resized and padded masks/images, [h, w, num]/[h, w, 3].
        im0_shape (tuple): the original image shape
        ratio_pad (tuple): the ratio of the padding to the original image.

    Returns:
        masks (numpy.ndarray): The masks that are being returned.
    """
    # Rescale coordinates (xyxy) from im1_shape to im0_shape
    im1_shape = masks.shape
    if im1_shape[:2] == im0_shape[:2]:
        return masks
    if ratio_pad is None:  # calculate from im0_shape
        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
    else:
        # gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    top, left = (int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1)))  # y, x
    bottom, right = (int(round(im1_shape[0] - pad[1] + 0.1)), int(round(im1_shape[1] - pad[0] + 0.1)))

    masks = masks[top:bottom, left:right]
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))
    if len(masks.shape) == 2:
        masks = masks[:, :, None]
    masks = masks.transpose(2, 0, 1)
    return masks


def crop_mask(masks, boxes):
    """
    It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box.

    Args:
        masks (numpy.ndarray): [n, h, w] tensor of masks
        boxes (numpy.ndarray): [n, 4] tensor of bbox coordinates in relative point form

    Returns:
        (numpy.ndarray): The masks are being cropped to the bounding box.
    """
    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(boxes[:, :, None], 4,axis=1)  # x1 shape(n,1,1)
    r = np.arange(w)[None, None, :]  # rows shape(1,1,w)
    c = np.arange(h)[None, :, None]  # cols shape(1,h,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))



def process_mask_native(protos, masks_in, bboxes, shape):
    """
    It takes the output of the mask head, and crops it after upsampling to the bounding boxes.

    Args:
        protos (numpy.ndarray): [mask_dim, mask_h, mask_w]
        masks_in (numpy.ndarray): [n, mask_dim], n is number of masks after nms
        bboxes (numpy.ndarray): [n, 4], n is number of masks after nms
        shape (tuple): the size of the input image (h,w)

    Returns:
        masks (numpy.ndarray): The returned masks with dimensions [h, w, n]
    """
    c, mh, mw = protos.shape  # CHW
    masks = sigmoid(masks_in @ protos.reshape(c, -1)).reshape(-1, mh, mw)  # CHW
    masks = scale_masks(masks.transpose(1,2,0), shape)  # CHW
    masks = crop_mask(masks, bboxes)  # CHW
    return np.greater(masks,0.5)


@MODELS.register_module()
class YOLOv8_seg(YOLOv8):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super(YOLOv8_seg, self).__init__(*args,**kwargs)


    def forward(self, data):
        image = data["ori_img"].copy()
        image = self.perprocess(image)
        preds = self.single_forward(image)
        pred = self.PostProcess(preds[0].squeeze(), self.conf_thres, self.iou_thres)
        
        proto = preds[1].squeeze()  # second output is len 3 

        orig_img = data["ori_img"]

        if not len(pred):  # save empty boxes
            masks = None
        else:
            pred[:, :4] = scale_boxes(image.shape[2:], pred[:, :4], orig_img.shape[:2])
            masks = process_mask_native(proto, pred[:, 6:], pred[:, :4], orig_img.shape)
            

        return {"bbox_ltrb":pred[:, :6],"masks":masks}

    