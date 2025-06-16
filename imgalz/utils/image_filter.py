import glob
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from PIL import Image
import numpy as np
import imagehash

try:
    from datasketch import MinHash, MinHashLSH

    _HAS_DATASKETCH = True
except ImportError:
    _HAS_DATASKETCH = False


from imgalz.utils import is_valid_image


__all__ = ["ImageFilter"]


class ImageHasher:
    def __init__(self, method="ahash", num_perm=128):
        self.method = method.lower()
        self.num_perm = num_perm

    def hash(self, image_path):
        image = Image.open(image_path)
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


class ImageFilter:
    """
    A utility class for detecting and filtering duplicate or similar images
    based on perceptual or MinHash-based hashing.

    Args:
        image_dir (Union[str, Path]): Path to the directory containing input images to be filtered.
        save_dir (Union[str, Path]): Path where filtered (non-duplicate) images will be saved.
        hash (str): Hashing method to use. Supported options are:
            - 'ahash': Average Hash
            - 'phash': Perceptual Hash
            - 'dhash': Difference Hash
            - 'whash': Wavelet Hash
            - 'minhash': MinHash (for scalable set similarity)
        threshold (int): Similarity threshold to determine duplicates.
            For non-Minhash methods, this is a Hamming distance threshold.
        max_workers (int): Maximum number of threads for parallel image hashing.


    Example:
        ```python
        from imgalz import ImageFilter


        deduper = ImageFilter(
            image_dir="/path/to/src",
            save_dir="/path/to/dst",
            hash="ahash",
            threshold=5,
            max_workers=8
        )
        deduper.run()
        ```
    """

    hash_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif")

    def __init__(self, image_dir, save_dir, hash="ahash", threshold=5, max_workers=8):

        self.image_dir = Path(image_dir)
        self.save_dir = Path(save_dir)
        self.hasher = ImageHasher(method=hash)
        self.threshold = threshold
        self.max_workers = max_workers

        self.image_hashes = []
        if self.hasher.method == "minhash":
            if not _HAS_DATASKETCH:
                raise RuntimeError(
                    "MinHash mode requires the datasketch library. Please install it with: pip install datasketch"
                )
            self.lsh = MinHashLSH(threshold=0.8, num_perm=self.hasher.num_perm)

        image_paths = []
        for ext in self.hash_exts:
            image_paths.extend(glob.glob(f"{self.image_dir}/**/*{ext}", recursive=True))

        self.image_paths = image_paths

    def _compute_hashes(self):
        print("Computing image hashes...")
        valid_paths = [p for p in self.image_paths if is_valid_image(p)]
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            hashes = list(
                tqdm(
                    executor.map(self.hasher.hash, valid_paths), total=len(valid_paths)
                )
            )
        self.image_hashes = list(zip(valid_paths, hashes))

    def _hamming(self, h1, h2):
        return bin(h1 ^ h2).count("1")

    def _build_lsh_index(self):
        print("Building LSH index...")
        for path, h in tqdm(self.image_hashes):
            self.lsh.insert(path, h)

    def _filter_similar(self):
        print("Filtering similar images...")
        keep = []
        removed = set()

        if self.hasher.method == "minhash":
            self._build_lsh_index()
            for path, h in tqdm(self.image_hashes):
                if path in removed:
                    continue
                near_dups = self.lsh.query(h)
                near_dups = [p for p in near_dups if p != path]
                removed.update(near_dups)
                keep.append(path)
        else:
            for i, (p1, h1) in enumerate(tqdm(self.image_hashes)):
                if p1 in removed:
                    continue
                for j in range(i + 1, len(self.image_hashes)):
                    p2, h2 = self.image_hashes[j]
                    if p2 in removed:
                        continue
                    if self._hamming(h1, h2) <= self.threshold:
                        removed.add(p2)
                keep.append(p1)

        return keep

    def _copy_images(self, keep_paths):
        print("Copying images to save directory...")
        for path in tqdm(keep_paths):
            target_path = self.save_dir / Path(path).relative_to(self.image_dir)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target_path)

    def run(self):
        self._compute_hashes()
        keep = self._filter_similar()
        self._copy_images(keep)
