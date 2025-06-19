import cv2
import numpy as np
import os
from pathlib import Path
from typing import Union, List, Tuple, Literal, Optional, Iterator, Any
import colorsys
import random
from .common import is_url, url_to_image, Cache

__all__ = [
    "imread",
    "imwrite",
    "cv_imshow",
    "draw_bbox",
    "draw_keypoints",
    "draw_masks",
    "compute_color_for_labels",
    "VideoReader",
    "palette"
]


def cv_imshow(
    title: str,
    image: np.ndarray,
    color_type: Literal["bgr", "rgb"] = "bgr",
    delay: int = 0,
    size: Optional[Union[int,List]] = None,
) -> Optional[bool]:
    """
    Display an image in a window. Converts color if needed.
    Optionally resizes the image to fit screen if too large.

    Args:
        title (str): Window title or filename prefix if saving.
        image (np.ndarray): Image array.
        color_type (Literal['bgr', 'rgb'], optional): Input image color space.
            Defaults to 'bgr'.
        delay (int, optional): Delay in milliseconds for display.
            If 0, waits indefinitely.Defaults to 0.
    """
    if color_type == "rgb":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if size is not None:
        if isinstance(size, int):
            size = [size, size]
        target_w, target_h = size
        h, w = image.shape[:2]

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, target_w, target_h)
    else:
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)

    cv2.imshow(title, image)
    if delay > 0:
        cv2.waitKey(delay)
        key = cv2.waitKey(delay) & 0xFF
        if key == 27:
            return True
        else:
            return False
    else:
        cv2.waitKey(0)
        cv2.destroyWindow(title)


def imwrite(
    filename: Union[str, Path],
    img: np.ndarray,
) -> bool:
    """
    Saves an image to a file, supporting paths with non-ASCII characters.

    Args:
        filename (Union[str, Path]): Path to save the image.
        img (np.ndarray): Image data array.


    Returns:
        bool: True if the image is successfully saved, False otherwise.
    """
    filename = str(filename)
    try:
        ext = os.path.splitext(filename)[1]  # file extension with dot, e.g. '.jpg'
        result, encoded_img = cv2.imencode(ext, img)
        if not result:
            return False
        encoded_img.tofile(filename)
        return True
    except Exception:
        return False


def imread(path: Union[str, Path], flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """
    Reads an image from a file, supporting paths with non-ASCII characters.

    Args:
        path (Union[str, Path]): Path to the image file.
        flags (int, optional): Flags specifying the color type of a loaded image.
            Defaults to cv2.IMREAD_COLOR.

    Returns:
        np.ndarray: The loaded image array.
    """
    path = str(path)
    if is_url(path):
        return url_to_image(path, readFlag=flags)
    try:
        image_array = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(image_array, flags)
        if image is None:
            print(f"Failed to decode image from file: {path}")
        return image
    except Exception as e:
        print(f"Failed to read image from file {path}: {e}")
        return None


class VideoReader:
    def __init__(self, filename: str, cache_capacity: int = 10, step: int = 1):
        if not filename.startswith(("http://", "https://")):
            if not os.path.isfile(filename):
                raise FileNotFoundError(f"Video file not found: {filename}")
        if cache_capacity <= 0:
            raise ValueError("cache_capacity must be a positive integer")
        if step <= 0:
            raise ValueError("step must be a positive integer")

        self._vcap = cv2.VideoCapture(filename)
        if not self._vcap.isOpened():
            raise RuntimeError(f"Failed to open video: {filename}")

        self._cache = Cache(cache_capacity)
        self._step = step
        self._position = 0
        self._width = int(self._vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = self._vcap.get(cv2.CAP_PROP_FPS)
        self._frame_cnt = int(self._vcap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fourcc = self._vcap.get(cv2.CAP_PROP_FOURCC)

    # ----------- Properties -----------
    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def resolution(self):
        return (self._width, self._height)

    @property
    def fps(self):
        return self._fps

    @property
    def frame_cnt(self):
        return self._frame_cnt

    @property
    def fourcc(self):
        return self._fourcc

    @property
    def position(self):
        return self._position

    @property
    def step(self):
        return self._step

    def _query_frame_position(self) -> int:
        return int(round(self._vcap.get(cv2.CAP_PROP_POS_FRAMES)))

    def _seek_frame_safely(self, frame_id: int) -> None:
        self._vcap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        actual = self._query_frame_position()
        for _ in range(frame_id - actual):
            self._vcap.read()
        self._position = frame_id

    def _decode_frame(self, frame_id: int) -> Optional[Any]:
        cached = self._cache.get(frame_id)
        if cached is not None:
            self._position = frame_id + self._step
            return cached

        self._seek_frame_safely(frame_id)
        ret, frame = self._vcap.read()
        if ret:
            self._cache.put(frame_id, frame)
            self._position += self._step
            return frame
        return None

    def read(self) -> Optional[Any]:
        return self._decode_frame(self._position)

    def get_frame(self, frame_id: int) -> Optional[Any]:
        if not (0 <= frame_id < self._frame_cnt):
            raise IndexError(f"frame_id must be between 0 and {self._frame_cnt - 1}")
        return self._decode_frame(frame_id)

    def current_frame(self) -> Optional[Any]:
        if self._position == 0:
            return None
        return self._cache.get(self._position - self._step)

    # ----------- Python Magic Methods -----------

    def __len__(self) -> int:
        return self._frame_cnt

    def __getitem__(self, index: Union[int, slice]) -> Union[Any, List[Any]]:
        if isinstance(index, slice):
            return [self.get_frame(i) for i in range(*index.indices(self._frame_cnt))]
        if index < 0:
            index += self._frame_cnt
        if index < 0 or index >= self._frame_cnt:
            raise IndexError("index out of range")
        return self.get_frame(index)

    def __iter__(self) -> Iterator[Any]:
        self._seek_frame_safely(0)
        return self

    def __next__(self) -> Any:
        frame = self.read()
        if frame is None:
            raise StopIteration
        return frame

    next = __next__  # Optional for Py2 style

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._vcap.release()

    def __del__(self):
        if hasattr(self, "_vcap") and self._vcap.isOpened():
            self._vcap.release()


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
    tl = (
        line_thickness or round(0.001 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # auto thickness

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


def draw_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    skeleton: list,
    kpt_color: np.ndarray,
    limb_color: np.ndarray,
    image_shape: tuple = None,
    radius: int = 5,
    draw_limb: bool = True,
    conf_threshold: float = 0.3,
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

    Returns:
        np.ndarray: same shape as input image, dtype uint8.

    Example:
        >>> from imgalz.utils.dataset_info import CocoConfig
        >>> import numpy as np
        >>> image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        >>> keypoints = np.random.rand((17,3))
        >>> kpts[:, 0] *= 640  # x
        >>> kpts[:, 1] *= 640  # y
        >>> kpts[:, 2] = 1.0   # conf
        >>> skeleton = CocoConfig.skeleton
        >>> kpt_color = CocoConfig.kpt_color
        >>> limb_color = CocoConfig.limb_color
        >>> draw_keypoints(image, keypoints, skeleton, kpt_color, limb_color)
    """
    if image_shape is None:
        image_shape = image.shape[:2]

    image = image.copy()
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
            cv2.line(
                image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2, cv2.LINE_AA
            )
    return image

def draw_masks(masks: np.ndarray, colors: Union[list,np.ndarray], image: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay multiple binary masks onto an image with given colors and alpha blending.

    Args:
        masks (np.ndarray): Boolean or float masks of shape (N, H, W), each mask is in [0, 1] or {0, 1}.
        colors (np.ndarray|list): RGB colors of shape (N, 3), each color is [R, G, B] in [0, 255].
        image (np.ndarray): Original image of shape (H, W, 3), dtype uint8, values in [0, 255].
        alpha (float, optional): Opacity of each mask, between 0 (transparent) and 1 (opaque). Default is 0.5.

    Returns:
        np.ndarray: Image with masks overlaid, same shape as input image, dtype uint8.

    Example:
        >>> output = draw_masks(masks, colors, image, alpha=0.5)
        >>> cv2.imshow("Masked", output)
    """
    if isinstance(colors,list):
        colors = np.array(colors)
    
    assert masks.ndim == 3, "Masks must be of shape (N, H, W)"
    assert colors.shape[0] == masks.shape[0], "Each mask must have a corresponding color"
    assert image.ndim == 3 and image.shape[2] == 3, "Image must be HxWx3"

    image = image.copy()
    image = image.astype(np.float64)  # convert for float ops
    colors = colors[:, None, None, :]  # (N, 1, 1, 3)
    masks = masks[..., None]  # (N, H, W, 1)

    colored_masks = masks * (colors * alpha)  # (N, H, W, 3)

    inv_alpha_masks = (1 - masks * alpha).cumprod(axis=0)  # (N, H, W, 1)

    merged_masks = colored_masks.max(axis=0)  # (H, W, 3)

    image = image * inv_alpha_masks[-1] + merged_masks
    image = np.clip(image, 0, 255)

    return image.astype(np.uint8)

def generate_distinct_colors(n: int = 20) -> list:
    """
    Generate n visually distinct hex colors using HCL-like HSV sampling.

    Args:
        n (int): Number of distinct colors.

    Returns:
        List[str]: List of hex color strings, e.g. ["#FF0000", "#00FF00", ...]
    """
    colors = []
    for i in range(n):
        hue = i / n
        lightness = 0.6
        saturation = 0.8
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))
        colors.append(hex_color)

    return colors


class Colors:
    """
    Attributes:
        palette (list of tuple): List of RGB color values.
        n (int): The number of colors in the palette.
    """

    def __init__(self,hex_list: List[str]):

        self.palette = [self.hex2rgb(f"{c}") for c in hex_list]
        self.n = len(self.palette)

    def __call__(self, index: int, bgr=False):
        """Converts hex color codes to RGB values."""
        c = self.palette[int(index) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h:str):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))

colors = generate_distinct_colors(20)
random.seed(0)
random.shuffle(colors)
palette = Colors(colors)