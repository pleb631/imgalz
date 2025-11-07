from imgalz import ImageFilter, ImageHasher, numpy_to_pillow, pillow_to_numpy
from pathlib import Path


def func1(image):
    img_numpy = pillow_to_numpy(image)
    # crop
    image = img_numpy[500:900, 1000:1700, :]
    return numpy_to_pillow(image)

if __name__ == "__main__":
    f = ImageFilter()
    hash = ImageHasher()
    # hash = ImageHasher(perprocess=func1)
    hash._hash_size = 8
    f = ImageFilter(hash=hash, max_workers=8)

    image_paths = Path("./test_images").rglob("*.jpg")
    # image_paths = ImageFilter.get_img_paths("./test_images")
    image_hashes = f.compute_hashes_mp(image_paths, show_progress=True, n_procs=3)

    keep = f.filter_similar(
        image_hashes,
        threshold=5,
        show_progress=True,
        bucket_bit=8,
        n_tables=2,
    )
    ImageFilter.copy_files(keep, "./test_images", "./save_dir", True)
