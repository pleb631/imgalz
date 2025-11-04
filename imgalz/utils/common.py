import cv2
import numpy as np

from pathlib import Path
import requests
from urllib import parse, request
from PIL import Image
from typing import Union, Optional, Any
from collections import OrderedDict
from multiprocessing import Pool, Manager
from tqdm import tqdm

__all__ = [
    "is_url",
    "url_to_image",
    "is_valid_image",
    "numpy_to_pillow",
    "pillow_to_numpy",
    "parallel_process",
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


def url_to_image(
    url: str, readFlag: int = cv2.IMREAD_COLOR, headers=None
) -> Optional[np.ndarray]:
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
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    try:
        response = requests.get(url, headers=headers, timeout=10)
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


def pillow_to_numpy(img):
    img_numpy = np.asarray(img)
    if not img_numpy.flags.writeable:
        img_numpy = np.array(img)
    return img_numpy


def numpy_to_pillow(img, mode=None):
    return Image.fromarray(img, mode=mode)



def _process_wrapper(args):
    func, item, lock, results, store = args
    result = func(item)
    if store:
        with lock:
            results.append(result)
    return result

def parallel_process(func, data, num_workers=4, store_results=True):
    """
    General-purpose function for parallel processing using multiple processes.

    Args:
        func (Callable): The function to execute in parallel.
        data (Iterable): An iterable of input items, each of which will be passed to `func`.
        num_workers (int, optional): The number of worker processes to use. Defaults to 4.
        store_results (bool, optional): Whether to store results in a shared list. 
            If False, results are not saved. Defaults to True.

    Returns:
        Tuple[List[Any] | None, float]: 
            - results: A list containing the results of all tasks if `store_results` is True; otherwise, None.
            - duration: Total execution time in seconds.

    Raises:
        None: All exceptions are caught internally; errors in child processes do not interrupt the main process.
    """
    if not callable(func):
        raise TypeError("func must be a callable function.")


    manager = Manager()
    results = manager.list() if store_results else None
    lock = manager.Lock()

    with Pool(processes=num_workers) as pool:
        args_iter = [(func, item, lock, results, store_results) for item in data]
        for _ in tqdm(pool.imap_unordered(_process_wrapper, args_iter), total=len(data)):
            pass

    return list(results) if store_results else None