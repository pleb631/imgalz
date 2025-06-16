import cv2
import numpy as np
from typing import Union, List, Tuple


__all__ = ["draw_bbox","compute_color_for_labels"]


def compute_color_for_labels(label):
    palette = (2**11 - 1, 2**15 - 1, 2**20 - 1)
    color = [int((p * (label**2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_bbox(
    img: np.ndarray,
    box: Union[List[float], np.ndarray],  # [x1, y1, x2, y2]
    score: float = 1.0,
    obj_id: str = None,
    line_thickness: Union[int, None] = None,
    label_format: str = "{score:.2f} {id}",
    txt_color: Tuple[int, int, int] = (255, 255, 255),
    box_color: Union[List[int], Tuple[int, int, int]] = [255, 0, 0],
) -> np.ndarray:
    """
    Draws a bounding box with optional label on the image.

    Args:
        img (np.ndarray): The image on which to draw.
        box (List[float] or np.ndarray): Bounding box in [x1, y1, x2, y2] format.
        score (float, optional): Confidence score for the object.
        obj_id (int, optional): Object ID or class index.
        line_thickness (int, optional): Line thickness of the box.
        label_format (str, optional): Format string for label. Use '{score}' and '{id}'.
        txt_color (Tuple[int, int, int], optional): Text color in BGR format.
        box_color (List[int] or Tuple[int, int, int], optional): Box color in BGR.

    Returns:
        np.ndarray: Image with bounding box and label drawn.
    """
    box = box.tolist() if isinstance(box, np.ndarray) else box
    tl = line_thickness or round(0.001 * (img.shape[0] + img.shape[1]) / 2) + 1  # auto thickness

    # Draw rectangle
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, p1, p2, box_color, thickness=tl)

    # Draw label
    if label_format:
        tf = max(tl - 1, 1)
        sf = tl / 3

        label = label_format.format(score=score, id=obj_id)
        w, h = cv2.getTextSize(label, 0, fontScale=sf, thickness=tf)[0]
        outside = p1[1] - h >= 3
        p2_label = (p1[0] + w, p1[1] - h - 3) if outside else (p1[0] + w, p1[1] + h + 3)

        cv2.rectangle(img, p1, p2_label, box_color, -1, cv2.LINE_AA)
        cv2.putText(
            img,
            label,
            (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
            0,
            sf,
            txt_color,
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

    return img


def draw_pose(
    image: np.ndarray,
    keypoints: np.ndarray,
    skeleton: list,
    kpt_color: np.ndarray,
    limb_color: np.ndarray,
    image_shape: tuple = None,
    radius: int = 5,
    draw_limb: bool = True,
    conf_threshold: float = 0.5,
):
    """
    Draw keypoints and skeletons on the image.

    Args:
        image (np.ndarray): Input image.
        keypoints (np.ndarray): Keypoints array with shape (17, 3), format [x, y, conf].
        skeleton (list): List of index pairs defining limb connections.
        kpt_color (np.ndarray): Color array for each keypoint.
        limb_color (np.ndarray): Color array for each limb.
        image_shape (tuple): Optional, (h, w). Defaults to image.shape[:2].
        radius (int): Radius of keypoint circles.
        draw_limb (bool): Whether to draw connecting lines between keypoints.
        conf_threshold (float): Minimum confidence to render a keypoint or limb.
    """
    if image_shape is None:
        image_shape = image.shape[:2]

    nkpt = keypoints.shape[0]
    for i in range(nkpt):
        x, y = keypoints[i][:2]
        conf = keypoints[i][2] if keypoints.shape[1] >= 3 else 1.0
        if conf < conf_threshold:
            continue
        if not (0 < x < image_shape[1] and 0 < y < image_shape[0]):
            continue
        color = tuple(int(c) for c in kpt_color[i % len(kpt_color)])
        cv2.circle(image, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)

    if draw_limb:
        for i, (a, b) in enumerate(skeleton):
            if a > nkpt or b > nkpt:
                continue
            x1, y1, c1 = keypoints[a - 1][:3]
            x2, y2, c2 = keypoints[b - 1][:3]
            if min(c1, c2) < conf_threshold:
                continue
            if not (0 < x1 < image_shape[1] and 0 < y1 < image_shape[0]):
                continue
            if not (0 < x2 < image_shape[1] and 0 < y2 < image_shape[0]):
                continue
            color = tuple(int(c) for c in limb_color[i % len(limb_color)])
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2, cv2.LINE_AA)
