from imgalz import ImageFilter, ImageHasher, numpy_to_pillow, pillow_to_numpy


def func1(image):
    img_numpy = pillow_to_numpy(image)
    # crop
    image = img_numpy[500:900, 1000:1700, :]
    return numpy_to_pillow(image)
def func2(i):
        print(i)
        import time
        time.sleep(1)
        return 0

if __name__ == "__main__":
    f = ImageFilter()
    # hash = ImageHasher(perprocess=func1)
    # f = ImageFilter(hash=hash)

    keep = f.run("./test_images", 1, True)
    ImageFilter.copy_images(keep, "./test_images", "./save_dir", True)
