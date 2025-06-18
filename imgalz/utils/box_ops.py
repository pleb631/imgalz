import numpy as np
from typing import Union, List, Tuple

__all__ = [
    "xyxy2ltwh",
    "ltwh2xyxy",
    "xywh2xyxy",
    "xyxy2xywh",
    "expand_box",
    "nms",
    "xywh2xyxyxyxy"
]


def xyxy2ltwh(
    boxes: Union[np.ndarray, list],
) -> np.ndarray:
    """
    Convert bounding boxes from [xmin, ymin, xmax, ymax, ...] format
    to [xmin, ymin, width, height, ...] format.
    Supports arbitrary dimensional input as long as last dimension >= 4.
    Only converts the first 4 elements per box, others remain unchanged.

    Args:
        boxes (Union[np.ndarray, list]): Input boxes, shape (..., >=4).

    Returns:
        np.ndarray: Converted boxes, same shape as input.

    Example:
        >>> xyxy2ltwh([10, 20, 30, 40])
        array([10., 20., 20., 20.])

        >>> xyxy2ltwh([[10, 20, 30, 40, 0], [5, 5, 15, 15, 1]])
        array([[10., 20., 20., 20., 0.],
               [ 5.,  5., 10., 10., 1.]])
    """
    arr = np.asarray(boxes)
    if arr.shape[-1] < 4:
        raise ValueError("The last dimension must be at least 4")

    converted = arr.copy()
    converted[..., 2] = converted[..., 2] - converted[..., 0]  # width = xmax - xmin
    converted[..., 3] = converted[..., 3] - converted[..., 1]  # height = ymax - ymin

    return converted


def ltwh2xyxy(boxes: Union[np.ndarray, list]) -> np.ndarray:
    """
    Convert bounding boxes from [xmin, ymin, w, h, ...] format
    to [xmin, ymin, xmax, ymax, ...] format.
    Supports arbitrary dimensional input as long as last dimension >= 4.
    Only converts the first 4 elements in the last dimension.

    Args:
        boxes (Union[np.ndarray, list]): Input boxes, shape (..., >=4).

    Returns:
        np.ndarray: Converted boxes, same shape as input.

    Example:
        >>> ltwh2xyxy([10, 20, 20, 20])
        array([10, 20, 30, 40])

        >>> ltwh2xyxy([[10, 20, 20, 20, 0], [5, 5, 10, 10, 1]])
        array([[10, 20, 30, 40, 0],
               [ 5,  5, 15, 15, 1]])

        >>> ltwh2xyxy(np.array([[[10,20,20,20],[5,5,10,10]], [[1,2,2,2],[6,7,2,2]]]))
        array([[[10, 20, 30, 40],
                [ 5,  5, 15, 15]],

               [[ 1,  2,  3,  4],
                [ 6,  7,  8,  9]]])
    """
    arr = np.asarray(boxes)
    if arr.shape[-1] < 4:
        raise ValueError("The last dimension must be at least 4")

    converted = arr.copy()
    converted[..., 2] = converted[..., 0] + converted[..., 2]  # xmax = xmin + w
    converted[..., 3] = converted[..., 1] + converted[..., 3]  # ymax = ymin + h

    return converted


def xywh2xyxy(boxes: Union[np.ndarray, list]) -> np.ndarray:
    """
    Convert bounding boxes from [x_center, y_center, w, h, ...] format
    to [xmin, ymin, xmax, ymax, ...] format.
    Supports arbitrary dimensional input as long as last dimension >= 4.
    Only converts the first 4 elements in the last dimension.

    Args:
        boxes (Union[np.ndarray, list]): Input boxes, shape (..., >=4).

    Returns:
        np.ndarray: Converted boxes, same shape as input.

    Example:
        >>> xywh2xyxy([50, 50, 20, 20])
        array([40., 40., 60., 60.])

        >>> xywh2xyxy([[50, 50, 20, 20, 0], [10, 10, 4, 6, 1]])
        array([[40., 40., 60., 60.,  0.],
               [ 8.,  7., 12., 13.,  1.]])
    """
    arr = np.asarray(boxes, dtype=np.float32)
    if arr.shape[-1] < 4:
        raise ValueError("The last dimension must be at least 4")

    converted = arr.copy()
    converted[..., 0] = arr[..., 0] - arr[..., 2] / 2  # xmin = x_center - w/2
    converted[..., 1] = arr[..., 1] - arr[..., 3] / 2  # ymin = y_center - h/2
    converted[..., 2] = arr[..., 0] + arr[..., 2] / 2  # xmax = x_center + w/2
    converted[..., 3] = arr[..., 1] + arr[..., 3] / 2  # ymax = y_center + h/2

    return converted


def xyxy2xywh(boxes: Union[list, np.ndarray]) -> np.ndarray:
    """
    Convert bounding boxes from [xmin, ymin, xmax, ymax] format
    to [x_center, y_center, width, height] format.
    Supports arbitrary dimensional input as long as the last dimension is at least 4.
    Keeps any additional trailing elements unchanged.

    Args:
        boxes (list or np.ndarray): Input boxes with shape (..., >=4).

    Returns:
        np.ndarray: Converted boxes with the same shape as input.

    Example:
        >>> xyxy2xywh([10, 20, 30, 40])
        array([20., 30., 20., 20.])

        >>> xyxy2xywh([[10, 20, 30, 40, 1], [5, 5, 15, 15, 2]])
        array([[20., 30., 20., 20.,  1.],
               [10., 10., 10., 10.,  2.]])
    """
    arr = np.asarray(boxes, dtype=np.float32)
    if arr.shape[-1] < 4:
        raise ValueError("The last dimension must be at least 4")

    converted = arr.copy()
    converted[..., 0] = (arr[..., 0] + arr[..., 2]) / 2  # x_center
    converted[..., 1] = (arr[..., 1] + arr[..., 3]) / 2  # y_center
    converted[..., 2] = arr[..., 2] - arr[..., 0]  # width
    converted[..., 3] = arr[..., 3] - arr[..., 1]  # height

    return converted


def expand_box(
    xyxy: Union[np.ndarray, Tuple[float, float, float, float]],
    ratio: Union[float, Tuple[float, float]],
    w: int,
    h: int,
) -> np.ndarray:
    """
    Expand bounding box size by given ratio and clip it within image dimensions.

    Args:
        xyxy (array-like): Bounding box coordinates in format [xmin, ymin, xmax, ymax].
        ratio (float or tuple of float): Expansion ratio.
            - If float, both width and height are scaled by this ratio.
            - If tuple of two floats, width and height are scaled separately.
        w (int): Image width, used to clip bounding box.
        h (int): Image height, used to clip bounding box.

    Returns:
        np.ndarray: Expanded and clipped bounding box in format [xmin, ymin, xmax, ymax].
    """
    xyxy = np.array(xyxy, dtype=np.float32)
    if isinstance(ratio, (float, int)):
        ratio = np.array([ratio, ratio], dtype=np.float32)
    else:
        ratio = np.array(ratio, dtype=np.float32)
        if ratio.size != 2:
            raise ValueError("ratio must be a float or a tuple/list of length 2")

    x1, y1, x2, y2 = xyxy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = (x2 - x1) * ratio[0], (y2 - y1) * ratio[1]

    new_x1 = np.clip(cx - bw / 2, 0, w)
    new_y1 = np.clip(cy - bh / 2, 0, h)
    new_x2 = np.clip(cx + bw / 2, 0, w)
    new_y2 = np.clip(cy + bh / 2, 0, h)

    return np.array([new_x1, new_y1, new_x2, new_y2], dtype=np.float32)


def nms(boxes, probs, overlapThresh=0.3):

    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    idxs = np.argsort(probs)

    while len(idxs) > 0:

        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        )

    # integer data type
    return pick

def xywh2xyxyxyxy(center):
    """
    Convert oriented bounding boxes (OBB) from [cx, cy, w, h, angle] format
    to 4 corner points [x1, y1, x2, y2, x3, y3, x4, y4].

    Args:
        center (np.ndarray): Input array of shape (..., 5), last dimension is [cx, cy, w, h, angle in degrees].

    Returns:
        np.ndarray: Output array of shape (..., 8), each element is [x1, y1, x2, y2, x3, y3, x4, y4].

    Example:
        >>> box = np.array([100, 100, 40, 20, 45])
        >>> xyxy = xywh2xyxyxyxy(box)
        >>> print(xyxy.shape)  # (8,)

        >>> batch_boxes = np.random.rand(2, 3, 5) * 100
        >>> xyxy_batch = xywh2xyxyxyxy(batch_boxes)
        >>> print(xyxy_batch.shape)  # (2, 3, 8)
    """
    center = np.asarray(center, dtype=np.float32)
    assert center.shape[-1] == 5, "The last dimension of input must be 5: [cx, cy, w, h, angle]"

    cx, cy, w, h, angle = np.moveaxis(center, -1, 0)
    angle = np.deg2rad(angle)

    dx = w / 2
    dy = h / 2

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    dx_cos = dx * cos_a
    dx_sin = dx * sin_a
    dy_cos = dy * cos_a
    dy_sin = dy * sin_a

    x1 = cx - dx_cos - dy_sin
    y1 = cy + dx_sin - dy_cos
    x2 = cx + dx_cos - dy_sin
    y2 = cy - dx_sin - dy_cos
    x3 = cx + dx_cos + dy_sin
    y3 = cy - dx_sin + dy_cos
    x4 = cx - dx_cos + dy_sin
    y4 = cy + dx_sin + dy_cos

    corners = np.stack([x1, y1, x2, y2, x3, y3, x4, y4], axis=-1)
    return corners
