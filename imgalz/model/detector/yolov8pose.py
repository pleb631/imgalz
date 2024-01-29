from imgalz.model import MODELS
from .yolov8 import YOLOv8,scale_boxes


def clip_coords(coords, shape):
    """
    Clip line coordinates to the image boundaries.

    Args:
        coords (torch.Tensor | numpy.ndarray): A list of line coordinates.
        shape (tuple): A tuple of integers representing the size of the image in the format (height, width).

    Returns:
        (torch.Tensor | numpy.ndarray): Clipped coordinates
    """

    coords[..., 0] = coords[..., 0].clip(0, shape[1])  # x
    coords[..., 1] = coords[..., 1].clip(0, shape[0])  # y
    return coords

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, normalize=False, padding=True):
    """
    Rescale segment coordinates (xy) from img1_shape to img0_shape.

    Args:
        img1_shape (tuple): The shape of the image that the coords are from.
        coords (numpy.ndarray): the coords to be scaled of shape n,2.
        img0_shape (tuple): the shape of the image that the segmentation is being applied to.
        ratio_pad (tuple): the ratio of the image size to the padded image size.
        normalize (bool): If True, the coordinates will be normalized to the range [0, 1]. Defaults to False.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.

    Returns:
        coords (numpy.ndarray): The scaled coordinates.
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        coords[..., 0] -= pad[0]  # x padding
        coords[..., 1] -= pad[1]  # y padding
    coords[..., 0] /= gain
    coords[..., 1] /= gain
    coords = clip_coords(coords, img0_shape)
    if normalize:
        coords[..., 0] /= img0_shape[1]  # width
        coords[..., 1] /= img0_shape[0]  # height
    return coords



@MODELS.register_module()
class YOLOv8_pose(YOLOv8):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super(YOLOv8_pose, self).__init__(*args,**kwargs)
        self.nc = 1
        self.kpt_shape=[17,3]

    def forward(self, data):
        image = data["ori_img"].copy()
        image = self.perprocess(image)
        pred = self.single_forward(image)[0].squeeze()
        pred = self.PostProcess(pred, self.conf_thres, self.iou_thres)
        if len(pred) > 0:
            pred[:, :4] = scale_boxes(
                image.shape[2:], pred[:, :4], data["ori_img"].shape[:2]
            ).round()

        
        pred_kpts = pred[:, 6:].reshape(len(pred), *self.kpt_shape) if len(pred) else pred[:, 6:]
        pred_kpts = scale_coords(image.shape[2:], pred_kpts, data["ori_img"].shape[:2])
            

        return {"bbox_ltrb":pred[:,:6], "kpt":pred_kpts}
    