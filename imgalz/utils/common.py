import cv2
import numpy as np
import os
from pathlib import Path
import requests
from urllib import parse, request
from PIL import Image
from typing import Union, Optional, Any
from collections import OrderedDict
import json
import inspect
from functools import wraps
from huggingface_hub import hf_hub_download



__all__ = [
    "is_url",
    "url_to_image",
    "is_valid_image",
]


def is_url(url: str, check: bool = False) -> bool:
    """
    Validate if the given string is a URL and optionally check if the URL exists online.

    Args:
        url (str): The string to be validated as a URL.
        check (bool, optional): If True, performs an additional check to see if the URL exists online.

    Returns:
        (bool): True for a valid URL. If 'check' is True, also returns True if the URL exists online.

    Examples:
        >>> valid = is_url("https://www.example.com")
        >>> valid_and_exists = is_url("https://www.example.com", check=True)
    """
    try:
        url = str(url)
        result = parse.urlparse(url)
        assert all([result.scheme, result.netloc])  # check if is url
        if check:
            with request.urlopen(url) as response:
                return response.getcode() == 200  # check if exists online
        return True
    except Exception:
        return False


def url_to_image(url: str, readFlag: int = cv2.IMREAD_COLOR,headers=None) -> Optional[np.ndarray]:
    """
    Download an image from a URL and decode it into an OpenCV image.

    Args:
        url (str): URL of the image to download.
        readFlag (int, optional): Flag specifying the color type of a loaded image.
            Defaults to cv2.IMREAD_COLOR.

    Returns:
        Optional[np.ndarray]: Decoded image as a numpy array if successful, else None.
    """
    if headers is None:
        headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    try:
        response = requests.get(url,headers=headers, timeout=10)
        response.raise_for_status()
        image_array = np.frombuffer(response.content, dtype=np.uint8)
        image = cv2.imdecode(image_array, readFlag)
        return image
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except Exception as e:
        print(f"Image decode failed: {e}")
        return None

def is_valid_image(path: Union[str, Path]) -> bool:
    """
    Checks whether the given file is a valid image by attempting to open and verify it.

    Args:
        path (Union[str, Path]): Path to the image file.

    Returns:
        bool: True if the image is valid, False otherwise.

    Raises:
        None: All exceptions are caught internally and False is returned.
    """
    try:
        with Image.open(path) as img:
            img.verify()  # Verify that it is, in fact, an image
        return True
    except:
        return False


class Cache:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self._capacity = capacity
        self._cache = OrderedDict()

    def put(self, key: Any, value: Any) -> None:
        if key in self._cache:
            return
        if len(self._cache) >= self._capacity:
            self._cache.popitem(last=False)
        self._cache[key] = value

    def get(self, key: Any, default: Optional[Any] = None) -> Any:
        return self._cache.get(key, default)


def auto_download(category, local_cache_dir="./ckpt"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            mapping_json_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../cfg/mapping.json")
            )

            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            model_key = bound_args.arguments.get("model_path")
            if model_key is None:
                raise ValueError("model_path is not provided")

            if os.path.exists(model_key):
                return func(*args, **kwargs)
            model_key = Path(model_key).stem

            with open(mapping_json_path, "r", encoding="utf-8") as f:
                mapping = json.load(f).get(category, {})

            if model_key not in mapping:
                raise ValueError(
                    f"model key '{model_key}' is not found in the mapping of {category}"
                )

            hf_info = mapping[model_key]
            repo_id = hf_info["repo_id"]
            filename = hf_info["filename"]

            os.makedirs(local_cache_dir, exist_ok=True)

            local_path = os.path.join(local_cache_dir, filename)
            if not os.path.exists(local_path):
                print(f"Downloading {filename} from {repo_id}...")
                hf_hub_download(
                    repo_id=repo_id, filename=filename, local_dir=local_cache_dir
                )
            else:
                print(f"Found existing model file: {local_path}")

            bound_args.arguments["model_path"] = local_path
            return func(*bound_args.args, **bound_args.kwargs)

        return wrapper

    return decorator
