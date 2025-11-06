from imgalz import ImageFilter, ImageHasher, numpy_to_pillow, pillow_to_numpy


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
    f = ImageFilter(hash=hash)

    keep = f.run("./test_images", 1, True,bucket_bit=None)
    ImageFilter.copy_files(keep, "./test_images", "./save_dir", True)
