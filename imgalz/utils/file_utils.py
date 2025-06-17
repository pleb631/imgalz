# -*- coding: UTF-8 -*-
import os
import json
import csv
import pickle
import yaml
import numpy as np
from pathlib import Path
from typing import Union, Literal, Any, Mapping, Optional, Iterable, List


__all__ = [
    "read_json",
    "save_json",
    "read_yaml",
    "save_yaml",
    "read_csv",
    "save_csv",
    "read_pkl",
    "save_pkl",
    "read_txt",
    "save_txt",
    "list_files",
    "read_yolo_txt",
    "xywh2xyxyxyxy",
]


def read_json(
    json_path: Union[str, Path], mode: Literal["all", "line"] = "all"
) -> list[Any]:
    """
    Reads JSON content from a file.

    Supports reading the entire file as a JSON object or reading line-by-line
    for JSONL (JSON Lines) formatted files.

    Args:
        json_path (Union[str, Path]): The path to the JSON file.
        mode (Literal['all', 'line'], optional):
            The mode to read the file.
            - 'all': Read the entire file as a single JSON object.
            - 'line': Read the file line by line, each line being a JSON object.
            Defaults to 'all'.

    Returns:
        list[Any]: A list of JSON-parsed Python objects. For 'all' mode, the list will contain the root JSON object(s).
                   For 'line' mode, the list will contain one object per line.
    """
    json_path = Path(json_path)
    json_data = []

    with json_path.open("r", encoding="utf-8") as json_file:
        if mode == "all":
            json_data = json.load(json_file)
        elif mode == "line":
            for line in json_file:
                json_line = json.loads(line)
                json_data.append(json_line)
        else:
            raise ValueError(f"Unsupported mode '{mode}'. Use 'all' or 'line'.")

    if not isinstance(json_data, list):
        json_data = [json_data]

    return json_data


def save_json(
    json_path: Union[str, Path],
    info: Any,
    indent: int = 4,
    mode: Literal["w", "a"] = "w",
    with_return_char: bool = False,
) -> None:
    """
    Saves a Python object to a JSON file.

    Args:
        json_path (Union[str, Path]): Path to the JSON file to write.
        info (Any): The Python object to serialize as JSON.
        indent (int, optional): Number of spaces to use for indentation. Defaults to 4.
        mode (Literal['w', 'a'], optional): File write mode.
            - 'w': Overwrite the file.
            - 'a': Append to the file.
            Defaults to 'w'.
        with_return_char (bool, optional): Whether to append a newline character at the end. Defaults to False.

    Returns:
        None
    """
    json_path = Path(json_path)
    json_str = json.dumps(info, indent=indent, ensure_ascii=False)

    if with_return_char:
        json_str += "\n"

    with json_path.open(mode, encoding="utf-8") as json_file:
        json_file.write(json_str)


def read_yaml(yaml_path: Union[str, Path]) -> Any:
    """
    Reads and parses a YAML file.

    Args:
        yaml_path (Union[str, Path]): Path to the YAML file.

    Returns:
        Any: The parsed Python object from the YAML file, usually a dict or list.
    """
    yaml_path = Path(yaml_path)

    with yaml_path.open("r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.load(yaml_file, Loader=yaml.FullLoader)

    return yaml_data


def save_yaml(
    yaml_path: Union[str, Path], data: Mapping[str, Any], header: str = ""
) -> None:
    """
    Saves a dictionary to a YAML file.

    Converts any unsupported value types to strings to ensure YAML serialization.

    Args:
        yaml_path (Union[str, Path]): The path to save the YAML file.
        data (Mapping[str, Any]): The dictionary to be saved.
        header (str, optional): An optional header string to be written before the YAML content. Defaults to ''.

    Returns:
        None
    """
    yaml_path = Path(yaml_path)

    # Define types safe for YAML dumping
    valid_types = (int, float, str, bool, list, tuple, dict, type(None))

    # Convert non-serializable values to strings
    safe_data = {
        k: (v if isinstance(v, valid_types) else str(v)) for k, v in data.items()
    }

    with yaml_path.open("w", encoding="utf-8", errors="ignore") as f:
        if header:
            f.write(header)
        yaml.safe_dump(safe_data, f, sort_keys=False, allow_unicode=True)


def read_csv(
    csv_path: Union[str, Path],
    delimiter: str = ",",
    skip_empty_lines: bool = True,
) -> List[List[str]]:
    """
    Reads a CSV file and returns its content as a list of rows.

    Args:
        csv_path (Union[str, Path]): Path to the CSV file.
        delimiter (str, optional): Delimiter used in the CSV file. Defaults to ','.
        skip_empty_lines (bool, optional): Whether to skip empty lines. Defaults to True.

    Returns:
        List[List[str]]: A list of rows, where each row is a list of strings.

    """
    csv_path = Path(csv_path)
    rows: List[List[str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.reader(csv_file, delimiter=delimiter)
        for row in reader:
            if skip_empty_lines and not any(cell.strip() for cell in row):
                continue
            rows.append(row)

    return rows


def save_csv(
    csv_path: Union[str, Path],
    info: List[List[Any]],
    mode: Literal["w", "a"] = "w",
    header: Optional[List[str]] = None,
) -> None:
    """
    Saves a 2D list to a CSV file.

    Args:
        csv_path (Union[str, Path]): Path to the CSV file.
        info (List[List[Any]]): Data to write, each sublist is a row.
        mode (Literal['w', 'a'], optional): Write mode.
            - 'w': Overwrite the file.
            - 'a': Append to the file.
            Defaults to 'w'.
        header (Optional[List[str]], optional): Optional column headers.
            Will be written as the first line if provided and mode is 'w'.
            Defaults to None.

    Returns:
        None
    """
    csv_path = Path(csv_path)

    with csv_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if header and mode == "w":
            writer.writerow(header)
        writer.writerows(info)


def read_txt(txt_path: Union[str, Path]) -> List[str]:
    """
    Reads a text file and returns a list of lines without trailing newline characters.

    Args:
        txt_path (Union[str, Path]): Path to the text file.

    Returns:
        List[str]: List of lines with trailing newline characters removed.
    """
    txt_path = Path(txt_path)
    with txt_path.open("r", encoding="utf-8") as txt_file:
        return [line.rstrip("\n") for line in txt_file]


def save_txt(txt_path: Union[str, Path], info: List[str], mode: str = "w") -> None:
    """
    Saves a list of strings to a text file, adding a newline character after each line.

    Args:
        txt_path (Union[str, Path]): Path to the text file.
        info (List[str]): List of strings to write, each string will be one line.
        mode (str, optional): File open mode, defaults to write mode 'w'.
    """
    txt_path = Path(txt_path)
    with txt_path.open(mode, encoding="utf-8") as txt_file:
        for line in info:
            txt_file.write(line + "\n")


def read_pkl(pkl_path: Union[str, Path]) -> Any:
    """
    Reads a pickle file and returns the deserialized data.

    Args:
        pkl_path (Union[str, Path]): Path to the pickle file.

    Returns:
        Any: The deserialized Python object stored in the pickle file.
    """
    pkl_path = Path(pkl_path)
    with pkl_path.open("rb") as pkl_file:
        pkl_data = pickle.load(pkl_file)
    return pkl_data


def save_pkl(pkl_path: Union[str, Path], pkl_data: Any) -> None:
    """
    Saves Python object data to a pickle file.

    Args:
        pkl_path (Union[str, Path]): Path to the pickle file to write.
        pkl_data (Any): Python object to serialize and save.

    Returns:
        None
    """
    pkl_path = Path(pkl_path)
    with pkl_path.open("wb") as pkl_file:
        pickle.dump(pkl_data, pkl_file)


def list_files(
    base_path: Union[str, Path],
    valid_exts: Optional[Union[str, List[str], tuple]] = None,
    contains: Optional[str] = None,
) -> Iterable[str]:
    """
    Recursively lists files in a directory, filtering by file extension and substring in filename.

    Args:
        base_path (Union[str, Path]): Directory path to search for files.
        valid_exts (Optional[Union[str, List[str], tuple]], optional):
            File extensions to filter by (e.g., '.jpg', ['.png', '.jpg']).
            Case insensitive. If None, no filtering by extension.
            Defaults to None.
        contains (Optional[str], optional): Substring that filenames must contain.
            If None, no filtering by substring.
            Defaults to None.

    Yields:
        Iterator[str]: Full file paths matching the criteria.
    """
    base_path = Path(base_path)

    for root_dir, _, filenames in os.walk(base_path):
        for filename in filenames:
            if contains is not None and contains not in filename:
                continue

            ext = os.path.splitext(filename)[1].lower()

            if valid_exts is None:
                matched = True
            elif isinstance(valid_exts, (list, tuple)):
                matched = ext in [e.lower() for e in valid_exts]
            else:
                matched = ext == valid_exts.lower()

            if matched:
                yield os.path.join(root_dir, filename)


def read_yolo_txt(txt_path: Union[str, Path], width:int, height:int):
    """
    Read YOLO-format annotation file and convert boxes to [x1, y1, x2, y2, class_id] format.

    Args:
        txt_path (str or Path): Path to the YOLO annotation text file.
        width (int or float): Width of the image the boxes are relative to.
        height (int or float): Height of the image the boxes are relative to.

    Returns:
        np.ndarray: Array of shape (N, 5), where each row is [x1, y1, x2, y2, class_id].

    Example:
        >>> boxes = read_yolo_txt("label.txt", 640, 480)
    """
    txt_path = Path(txt_path)
    boxes = []

    with txt_path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue  # skip invalid lines

            cls_id = int(parts[0])
            cx = float(parts[1]) * width
            cy = float(parts[2]) * height
            w = float(parts[3]) * width
            h = float(parts[4]) * height

            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            boxes.append([x1, y1, x2, y2, cls_id])

    return np.array(boxes, dtype=np.float32)



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

