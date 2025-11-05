import glob
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from PIL import Image
import numpy as np
import imagehash
import os
from typing import Union,Literal
from collections import defaultdict
import math

try:
    from datasketch import MinHash, MinHashLSH

    _HAS_DATASKETCH = True
except ImportError:
    _HAS_DATASKETCH = False


from imgalz.utils import is_valid_image


__all__ = ["ImageFilter", "ImageHasher"]


class ImageHasher:
    def __init__(self, method="ahash", num_perm=128, perprocess=None):
        self.method = method.lower()
        self.num_perm = num_perm
        self.perprocess = []
        if perprocess is not None:
            if callable(perprocess):
                self.perprocess.append(perprocess)
            elif isinstance(perprocess, (list, tuple)):
                for p in perprocess:
                    if callable(perprocess):
                        self.perprocess.append(p)
                    else:
                        raise ValueError(f"Invalid perprocess: {p}")
            else:
                raise ValueError("Invalid perprocess: {}".format(perprocess))

    def hash(self, image_path):
        image = Image.open(image_path)
        for p in self.perprocess:
            image = p(image)
        if self.method == "ahash":
            return int(str(imagehash.average_hash(image)), 16)
        elif self.method == "phash":
            return int(str(imagehash.phash(image)), 16)
        elif self.method == "dhash":
            return int(str(imagehash.dhash(image)), 16)
        elif self.method == "whash":
            return int(str(imagehash.whash(image)), 16)
        elif self.method == "minhash":
            return self._minhash(image)
        else:
            raise ValueError(f"Unsupported hash method: {self.method}")

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


def _hamming(h1, h2):
    return bin(h1 ^ h2).count("1")


def filter_hash(image_hashes, show_progress, threshold):
    removed = set()
    keep = []
    for i, (p1, h1) in enumerate(
        tqdm(image_hashes, desc="Filtering similar images...",leave=False)
        if show_progress
        else image_hashes
    ):
        if p1 in removed:
            continue
        for j in range(i + 1, len(image_hashes)):
            p2, h2 = image_hashes[j]
            
            
            if p2 in removed:
                continue
            if _hamming(h1, h2) <= threshold:
                removed.add(p2)
        keep.append(p1)
    return keep


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

        max_workers (int):
            Maximum number of threads for parallel image hashing.

        src_dir (Union[str, Path]): Path to the directory containing input images to be filtered.
        save_dir (Union[str, Path]): Path where filtered
        threshold (float): Similarity threshold to determine duplicates.
            For non-Minhash methods, this is a Hamming distance threshold.
        bucket_bit (Union[int, Literal["auto"], None]): Number of high-order bits of the image hash used for LSH bucketing.This balances memory usage, computation, and recall without manual tuning.
            - None: Disable bucket-based filtering; all images will be compared in a single group.
            - int: Manually specify the number of bits to use for bucketing. Smaller values create fewer, larger buckets 
            (more comparisons, higher recall), while larger values create more, smaller buckets (fewer comparisons, 
            potential misses).
            - "auto": Automatically determine an appropriate number of bucket bits based on the number of images to be filtered.
        show_progress (bool): Whether to display a progress bar during processing.
            


    Example:

        .. code-block:: python

            from imgalz import ImageFilter
            deduper = ImageFilter(hash="ahash", max_workers=8)
            keep = deduper.run(src_dir="/path/to/src", threshold=5)
    """

    hash_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif")

    def __init__(self, hash: Union[ImageHasher, str] = "ahash", max_workers=4):
        if isinstance(hash, str):
            self.hasher = ImageHasher(method=hash)
        elif isinstance(hash, ImageHasher):
            self.hasher = hash
        else:
            raise ValueError("hash must be a string or an ImageHasher instance")
        self.max_workers = max_workers

        if self.hasher.method == "minhash":
            if not _HAS_DATASKETCH:
                raise RuntimeError(
                    "MinHash mode requires the datasketch library. Please install it with: pip install datasketch"
                )
            self.lsh = MinHashLSH(threshold=0.8, num_perm=self.hasher.num_perm)

    def compute_hashes(self, image_paths: list, show_progress=False):
        valid_paths = [p for p in image_paths if is_valid_image(p)]
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            if show_progress:
                executor = tqdm(
                    executor.map(self.hasher.hash, valid_paths),
                    total=len(valid_paths),
                    desc="Computing image hashes...",
                )
                hashes = list(executor)
            else:
                hashes = list(executor.map(self.hasher.hash, valid_paths))

        image_hashes = list(zip(valid_paths, hashes))

        return image_hashes

    def _build_lsh_index(self):
        for path, h in tqdm(self.image_hashes, desc="Building LSH index..."):
            self.lsh.insert(path, h)

    def filter_similar(
        self,
        image_hashes: list,
        threshold: float = 5,
        show_progress=False,
        bucket_bit: Union[int, Literal["auto"], None] = None,
    ):
        keep = []
        removed = set()

        if self.hasher.method == "minhash":
            self._build_lsh_index()
            if show_progress:
                image_hashes = tqdm(image_hashes, desc="Filtering similar images...")
            for path, h in image_hashes:
                if path in removed:
                    continue
                near_dups = self.lsh.query(h)
                near_dups = [p for p in near_dups if p != path]
                removed.update(near_dups)
                keep.append(path)
        else:
            if bucket_bit is None:
                keep = filter_hash(image_hashes, show_progress, threshold)
            else:
                if bucket_bit == "auto":
                    bucket_bit = min(16, max(8, int(math.log2(len(image_hashes))) - 4))
                buckets = defaultdict(list)
                for path, h in image_hashes:
                    bucket_key = h >> (64 - bucket_bit)
                    buckets[bucket_key].append((path, h))
                for _, items in (
                    tqdm(buckets.items(), desc="Filtering buckets...")
                    if show_progress
                    else buckets.items()
                ):
                    keep.extend(filter_hash(items, show_progress, threshold))

        return keep

    @staticmethod
    def copy_images(
        keep_paths: list,
        image_dir: Path,
        save_dir: Path,
        show_progress=False,
        max_workers=8,
    ):
        image_dir = Path(image_dir)
        save_dir = Path(save_dir)

        def copy_file(path):
            target_path = save_dir / Path(path).relative_to(image_dir)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target_path)
            return 0

        if show_progress:
            for _ in tqdm(
                ThreadPoolExecutor(max_workers=max_workers).map(copy_file, keep_paths),
                total=len(keep_paths),
                desc="Copying files...",
            ):
                pass

        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                executor.map(copy_file, keep_paths)

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
        show_progress=True,
    ):
        image_paths = self.get_img_paths(src_dir, recursive)

        image_hashes = self.compute_hashes(image_paths, show_progress=show_progress)

        keep = self.filter_similar(
            image_hashes,
            threshold=threshold,
            show_progress=show_progress,
            bucket_bit=bucket_bit,
        )

        return keep
