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
    hash_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif")

    def __init__(self, image_dir, save_dir, hash, threshold=5, max_workers=8):
        """
        Initialize the deduplicator.

        Args:
            image_dir (str or Path): Path to the input image directory.
            save_dir (str or Path): Path where deduplicated images will be saved.
            hash (str): Hashing method to use. Options: 'ahash', 'phash', 'dhash', 'whash', 'minhash'.
            threshold (int): Hamming distance threshold for similarity (only for non-Minhash methods).
            max_workers (int): Number of threads to use for hashing images.
        """
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

    def is_valid_image(self, path):
        try:
            with Image.open(path) as img:
                img.verify()
            return True
        except:
            return False

    def compute_hashes(self):
        print("Computing image hashes...")
        valid_paths = [p for p in self.image_paths if self.is_valid_image(p)]
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            hashes = list(
                tqdm(
                    executor.map(self.hasher.hash, valid_paths), total=len(valid_paths)
                )
            )
        self.image_hashes = list(zip(valid_paths, hashes))

    def hamming(self, h1, h2):
        return bin(h1 ^ h2).count("1")

    def build_lsh_index(self):
        print("Building LSH index...")
        for path, h in tqdm(self.image_hashes):
            self.lsh.insert(path, h)

    def filter_similar(self):
        print("Filtering similar images...")
        keep = []
        removed = set()

        if self.hasher.method == "minhash":
            self.build_lsh_index()
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
                    if self.hamming(h1, h2) <= self.threshold:
                        removed.add(p2)
                keep.append(p1)

        return keep

    def copy_images(self, keep_paths):
        print("Copying images to save directory...")
        for path in tqdm(keep_paths):
            target_path = self.save_dir / Path(path).relative_to(self.image_dir)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target_path)

    def run(self):
        self.compute_hashes()
        keep = self.filter_similar()
        self.copy_images(keep)


# Example Usage
if __name__ == "__main__":
    deduper = ImageFilter(
        image_dir=r"src", save_dir=r"dst", hash="minhash", threshold=5, max_workers=8
    )
    deduper.run()
