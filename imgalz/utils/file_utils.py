# -*- coding: UTF-8 -*-
import os
import json
import csv
import pickle
import yaml
import numpy as np
from pathlib import Path
from typing import Union, Literal, Any, Sequence, Optional, Iterable, List


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
    "save_yolo_txt",
]


def read_json(
    json_path: Union[str, Path], mode: Literal["all", "line"] = "all"
) -> List[Any]:
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
        List[Any]: A list of JSON-parsed Python objects. For 'all' mode, the list will contain the root JSON object(s).
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


def sanitize_for_yaml(value: Any) -> Any:
    """Recursively sanitize a Python object for YAML dumping."""
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    elif isinstance(value, (list, tuple)):
        return [sanitize_for_yaml(v) for v in value]
    elif isinstance(value, dict):
        return {str(k): sanitize_for_yaml(v) for k, v in value.items()}
    else:
        return str(value)


def save_yaml(yaml_path: Union[str, Path], data: Any, header: str = "") -> None:
    """
    Saves any YAML-serializable Python data (dict, list, etc.) to a YAML file.

    Args:
        yaml_path (str or Path): The path to the output YAML file.
        data (Any): The Python data to save (dict, list, etc.).
        header (str): Optional header string to prepend to the file.
    """
    yaml_path = Path(yaml_path)
    safe_data = sanitize_for_yaml(data)

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


def read_yolo_txt(txt_path: Union[str, Path], width: int, height: int):
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


def save_yolo_txt(
    box: np.ndarray,
    cls: Union[np.ndarray, Sequence[int]],
    save_path: Union[str, Path],
    format: Literal["xywh", "xyxy"] = "xyxy",
    is_normalized: bool = False,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> None:
    """
    Save bounding boxes in YOLO format to a .txt file.

    Args:
        box (np.ndarray): Array of shape (N, 4), each row is either:
                          - [x1, y1, x2, y2] if format='xyxy'
                          - [cx, cy, w, h] if format='xywh'
        cls (np.ndarray or list): Array/list of class IDs, shape (N,).
        save_path (str or Path): Output path for the YOLO-format .txt file.
        format (str): Format of input boxes: "xywh" or "xyxy".
        input_is_normalized (bool): True if input box values are already normalized.
        width (int, optional): Image width (required if input_is_normalized=False).
        height (int, optional): Image height (required if input_is_normalized=False).
    """
    save_path = Path(save_path)
    if np.any(box < 0):
        raise ValueError(
            "Boxes contain negative values while input_is_normalized=False"
        )
    if not is_normalized:
        if np.all(box <= 1):
            raise ValueError("Input boxes look normalized but is_normalized=False.")
        if width is None or height is None:
            raise ValueError(
                "Image width and height must be provided if is_normalized=False"
            )
    else:
        if np.any(box > 1):
            raise ValueError(
                "Boxes contain values outside [0, 1] range while is_normalized=True"
            )

    if len(box) != len(cls):
        raise ValueError("box and cls must have the same length")

    lines = []
    for b, c in zip(box, cls):
        if format == "xyxy":
            x1, y1, x2, y2 = b
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
        elif format == "xywh":
            cx, cy, w, h = b
        else:
            raise ValueError("format must be either 'xyxy' or 'xywh'")

        if not is_normalized:
            cx /= width
            cy /= height
            w /= width
            h /= height

        line = f"{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        lines.append(line)

    save_txt(save_path, lines)
