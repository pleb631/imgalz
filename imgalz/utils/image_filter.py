import glob
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from PIL import Image
import numpy as np
import imagehash
import os
from typing import Union, Literal, Iterable, List
from collections import defaultdict
import math
import random
from functools import lru_cache


from .common import parallel_process

try:
    from datasketch import MinHash, MinHashLSH

    _HAS_DATASKETCH = True
except ImportError:
    _HAS_DATASKETCH = False


from imgalz.utils import is_valid_image

try:
    from .cpp import hashfilter_cpp
except:
    hashfilter_cpp = None


__all__ = ["ImageFilter", "ImageHasher"]


class ImageHasher:
    def __init__(self, method="ahash", num_perm=128, perprocess=None):
        self._method = method.lower()
        self.num_perm = num_perm
        self._hash_size = 8

        self.perprocess = []

        if perprocess is not None:
            if callable(perprocess):
                self.perprocess.append(perprocess)
            elif isinstance(perprocess, (list, tuple)):
                for p in perprocess:
                    if callable(p):
                        self.perprocess.append(p)
                    else:
                        raise ValueError(f"Invalid perprocess: {p}")
            else:
                raise ValueError("Invalid perprocess: {}".format(perprocess))

    def hash(self, image_path):
        image = Image.open(image_path)
        for p in self.perprocess:
            image = p(image)

        if self._method == "ahash":
            return int(str(imagehash.average_hash(image, self._hash_size)), 16)
        elif self._method == "phash":
            return int(str(imagehash.phash(image, self._hash_size)), 16)
        elif self._method == "dhash":
            return int(str(imagehash.dhash(image, self._hash_size)), 16)
        elif self._method == "whash":
            return int(str(imagehash.whash(image, self._hash_size)), 16)
        elif self._method == "minhash":
            return self._minhash(image)
        else:
            raise ValueError(f"Unsupported hash method: {self._method}")

    def _minhash(self, image):
        image = image.resize((8, 8)).convert("L")
        pixels = np.array(image).flatten()
        avg = pixels.mean()
        bits = (pixels > avg).astype(int)
        m = MinHash(num_perm=self.num_perm)
        for i, b in enumerate(bits):
            if b:
                m.update(str(i).encode("utf-8"))
        return m

    @property
    def method(self):
        return self._method

    @property
    def hash_size(self):
        return self._hash_size * self._hash_size


@lru_cache(maxsize=100_000)
def _compute(h1, h2):
    return bin(h1 ^ h2).count("1")


def _hamming(h1, h2):
    if h1 > h2:
        h1, h2 = h2, h1

    return _compute(h1, h2)


def int256_to_uint64_list(x):
    """将一个 256 位整数拆成 4 个 uint64_t 列表（高位在前）"""
    mask64 = (1 << 64) - 1
    return [(x >> 192) & mask64, (x >> 128) & mask64, (x >> 64) & mask64, x & mask64]


def _filter_similar_hashes(
    image_hashes, show_progress, max_distance, similarity_func, bits_len, leave=True
):
    if hashfilter_cpp and similarity_func is _hamming:
        if bits_len == 64:
            return hashfilter_cpp.filter_similar_hashes(image_hashes, max_distance)
        elif bits_len == 256:
            data = [
                (path, int256_to_uint64_list(hash_int))
                for path, hash_int in image_hashes
            ]
            return hashfilter_cpp.filter_similar_hashes256(data, max_distance)

    removed = set()
    keep = []
    for i, (p1, h1) in enumerate(
        tqdm(image_hashes, desc="Filtering similar images", leave=leave)
        if show_progress
        else image_hashes
    ):
        if p1 in removed:
            continue
        for j in range(i + 1, len(image_hashes)):
            p2, h2 = image_hashes[j]

            if p2 in removed:
                continue
            if similarity_func(h1, h2) <= max_distance:
                removed.add(p2)
        keep.append(p1)
    return keep, removed


class ImageFilter:
    """
    A utility class for detecting and filtering duplicate or similar images
    using perceptual or MinHash-based hashing.

    Args:
        hash (:class:`str` or :class:`ImageHasher`):
            Hashing method to use. Supported options:

            - 'ahash': Average Hash
            - 'phash': Perceptual Hash
            - 'dhash': Difference Hash
            - 'whash': Wavelet Hash
            - 'minhash': MinHash (for scalable set similarity)


    Example:

        .. code-block:: python

            from imgalz import ImageFilter
            deduper = ImageFilter(hash="ahash")
            keep = deduper.run(src_dir="/path/to/src", threshold=5)
    """

    hash_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif")

    def __init__(self, hash: Union[ImageHasher, str] = "ahash"):
        if isinstance(hash, str):
            self.hasher = ImageHasher(method=hash)
        elif isinstance(hash, ImageHasher):
            self.hasher = hash
        else:
            raise ValueError("hash must be a string or an ImageHasher instance")

        if self.hasher.method == "minhash":
            if not _HAS_DATASKETCH:
                raise RuntimeError(
                    "MinHash mode requires the datasketch library. Please install it with: pip install datasketch"
                )
            self._lsh = MinHashLSH(threshold=0.8, num_perm=self.hasher.num_perm)

    def _hash_img(self, image_path: Union[str, Path]):
        if not is_valid_image(image_path):
            return None
        hash = self.hasher.hash(image_path)
        return (str(image_path), hash)

    def compute_hashes(
        self,
        image_paths: Iterable,
        use_threads: bool = False,
        num_workers: int = 1,
        show_progress: bool = False,
    ):

        return parallel_process(
            self._hash_img,
            image_paths,
            use_threads=use_threads,
            num_workers=num_workers,
            show_progress=show_progress,
            store_results=True,
            prog_desc="Computing hashes",
        )[0]

    def _build_lsh_index(self):
        for path, h in tqdm(self.image_hashes, desc="Building LSH index"):
            self._lsh.insert(path, h)

    def filter_similar(
        self,
        image_hashes: List[tuple[str, int]],
        threshold: float = 5,
        show_progress=False,
        bucket_bit: Union[int, Literal["auto"], None] = None,
        n_tables: int = 1,
    ):
        keep = set()
        removed = set()

        if self.hasher.method == "minhash":
            self._build_lsh_index()
            if show_progress:
                image_hashes = tqdm(image_hashes, desc="Filtering similar images")
            for path, h in image_hashes:
                if path in removed:
                    continue
                near_dups = self.lsh.query(h)
                near_dups = [p for p in near_dups if p != path]
                removed.update(near_dups)
                keep.add(path)
        else:
            compare_func = _hamming

            hash_bits_len = self.hasher.hash_size
            if bucket_bit is None:
                keep, _ = _filter_similar_hashes(
                    image_hashes, show_progress, threshold, compare_func, hash_bits_len
                )
            else:

                if bucket_bit == "auto":
                    b = int(math.log2(len(image_hashes))) - 4
                    bucket_bit = max(8, min(hash_bits_len // 2, b))

                # LSH

                masks = [
                    random.sample(range(hash_bits_len), bucket_bit)
                    for _ in range(n_tables)
                ]

                tables = [defaultdict(list) for _ in range(n_tables)]
                for path, h in image_hashes:
                    bitstring = f"{h:0{hash_bits_len}b}"
                    for t, mask in zip(tables, masks):
                        key = "".join(bitstring[i] for i in mask)
                        t[key].append((path, h))

                for table_i, t in enumerate(
                    tqdm(tables, desc="lsh filtering") if show_progress else tables
                ):
                    for _, items in (
                        tqdm(t.items(), desc="Filtering buckets", leave=False)
                        if show_progress
                        else t.items()
                    ):
                        if table_i > 0:
                            items = [(p, h) for p, h in items if p not in removed]

                        if len(items) < 1:
                            continue

                        keep_i, removed_i = _filter_similar_hashes(
                            items,
                            show_progress,
                            threshold,
                            compare_func,
                            hash_bits_len,
                            leave=False,
                        )
                        removed.update(removed_i)
                        keep.update(keep_i)

        return keep

    @staticmethod
    def copy_files(
        keep_paths: list,
        image_dir: Path,
        save_dir: Path,
        show_progress=False,
        num_workers=8,
    ):
        image_dir = Path(image_dir)
        save_dir = Path(save_dir)

        def copy_file(path):
            target_path = save_dir / Path(path).relative_to(image_dir)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target_path)
            return 0

        return parallel_process(
            copy_file,
            keep_paths,
            use_threads=True,
            num_workers=num_workers,
            store_results=False,
            show_progress=show_progress,
            prog_desc="Copying files",
        )

    @staticmethod
    def get_img_paths(src_dir, recursive=True):
        image_paths = []
        for ext in ImageFilter.hash_exts:
            src_pattern = os.path.join(
                src_dir, f"**/*{ext}" if recursive else f"*{ext}"
            )
            image_paths.extend(glob.glob(src_pattern, recursive=recursive))
        return image_paths

    def run(
        self,
        src_dir: str,
        threshold=5,
        recursive=True,
        bucket_bit: Union[int, Literal["auto"], None] = None,
        n_tables: int = 1,
        show_progress=True,
        num_workers=1,
        use_threads=False,
    ):
        """
        src_dir (Union[str, Path]): Path to the directory containing input images to be filtered.

        threshold (float): Similarity threshold to determine duplicates.

        bucket_bit (Union[int, Literal["auto"], None]): Number of high-order bits of the image hash used for LSH bucketing.This balances memory usage, computation, and recall without manual tuning.
            - None: Disable bucket-based filtering; all images will be compared in a single group.
            - int: Manually specify the number of bits to use for bucketing. Smaller values create fewer, larger buckets
            (more comparisons, higher recall), while larger values create more, smaller buckets (fewer comparisons,
            potential misses).
            - "auto": Automatically determine an appropriate number of bucket bits based on the number of images to be filtered.


        n_tables (int): Number of LSH tables to use for bucket-based filtering.

        show_progress (bool): Whether to display a progress bar during processing.

        num_workers (int): Number of parallel processes to use for processing.

        use_threads (bool): Whether to use threads for parallel processing.

        """
        image_paths = self.get_img_paths(src_dir, recursive)

        image_hashes = self.compute_hashes(
            image_paths,
            use_threads=use_threads,
            show_progress=show_progress,
            num_workers=num_workers,
        )

        keep = self.filter_similar(
            image_hashes,
            threshold=threshold,
            show_progress=show_progress,
            bucket_bit=bucket_bit,
            n_tables=n_tables,
        )

        return keep
