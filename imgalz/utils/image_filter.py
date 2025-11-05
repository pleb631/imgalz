import glob
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from PIL import Image
import numpy as np
import imagehash
import os
from typing import Union
from collections import defaultdict
from typing import Literal, Optional
import math

try:
    from datasketch import MinHash, MinHashLSH

    _HAS_DATASKETCH = True
except ImportError:
    _HAS_DATASKETCH = False


from imgalz.utils import is_valid_image


__all__ = ["ImageFilter", "ImageHasher"]


class ImageHasher:
    def __init__(self, method="ahash", num_perm=128, hash_size=8,perprocess=None):
        self.method = method.lower()
        self.num_perm = num_perm
        self.hash_size = hash_size
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
            return int(str(imagehash.average_hash(image, self.hash_size)), 16)
        elif self.method == "phash":
            return int(str(imagehash.phash(image, self.hash_size)), 16)
        elif self.method == "dhash":
            return int(str(imagehash.dhash(image, self.hash_size)), 16)
        elif self.method == "whash":
            return int(str(imagehash.whash(image, self.hash_size)), 16)
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

def hamming_distance_matrix(batch_hashes, single_hash):
    """
    Compute Hamming distances between a single hash and a batch of hashes.
    Supports arbitrary-length integer hashes.
    """

    max_bits = max(max(batch_hashes).bit_length(), single_hash.bit_length())
    n_bytes = (max_bits + 7) // 8

    batch_bytes = np.array([np.frombuffer(h.to_bytes(n_bytes, 'big'), dtype=np.uint8) for h in batch_hashes], dtype=np.uint8)
    single_bytes = np.frombuffer(single_hash.to_bytes(n_bytes, 'big'), dtype=np.uint8)

    xor = batch_bytes ^ single_bytes
    dist = np.unpackbits(xor, axis=1).sum(axis=1)
    return dist

def filter_hash(image_hashes, show_progress, threshold, window: Optional[int] = 1000):
    paths = [p for p, _ in image_hashes]
    hashes = [h for _, h in image_hashes]

    n = len(paths)
    if window is None or window > n:
        window = n

    removed = set()
    keep = []

    range_iter = (
        tqdm(range(n), desc="Filtering similar images...", leave=False)
        if show_progress
        else range(n)
    )

    for i in range_iter:
        if paths[i] in removed:
            continue
        keep.append(paths[i])
        j = i + 1
        batch_hashes = []
        batch_paths = []
        while j < n:
            while len(batch_hashes) < window and j < n:

                if paths[j] not in removed:
                    batch_paths.append(paths[j])
                    batch_hashes.append(hashes[j])
                j += 1

            if len(batch_paths) > 0:
                hamming_distances = hamming_distance_matrix(batch_hashes, hashes[i])
                for k, d in zip(batch_paths, hamming_distances):
                    if d <= threshold:
                        removed.add(k)
            batch_hashes.clear()
            batch_paths.clear()

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

        window (int, optional):
            Number of images processed in each batch window.

    Example:

        .. code-block:: python

            from imgalz import ImageFilter
            deduper = ImageFilter(hash="ahash", max_workers=8)
            keep = deduper.run(src_dir="/path/to/src", threshold=5)
    """

    hash_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif")

    def __init__(
        self, hash: Union[ImageHasher, str] = "ahash", max_workers=4, window=None
    ):
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

        self.window = window

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
        window: Optional[int] = None,
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
                keep = filter_hash(image_hashes, show_progress, threshold, window)
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
                    keep.extend(filter_hash(items, show_progress, threshold, window))

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
            window=self.window,
        )

        return keep
