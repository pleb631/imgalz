import numpy as np
import cv2

from .hook import HOOKS, Hook
from imgalz.utils import imread

class VideoCapture:
    def __init__(self, image_infos):
        video_path = image_infos["video_path"]
        self.video = cv2.VideoCapture(video_path)
        assert self.video.isOpened(), f"Fail to open video:{video_path}"
        self.video_path = video_path
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.i = 0
        self.extra = image_infos

    def __iter__(self):
        return self

    def __next__(self):
        while self.video.isOpened():
            is_success, img = self.video.read()
            if not is_success:
                break
            self.i += 1
            info = self.extra.copy()
            info["ori_img"] = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB,
            )
            info["times"] = self.i
            return info
        raise StopIteration()


class ImgGenerator:
    def __init__(self, img_info):
        if isinstance(img_info, dict):
            img_info = [img_info]
        self.img_infos = iter(img_info)

    def __iter__(self):
        return self

    def __next__(self):
        image_info = next(self.img_infos)
        image = imread(image_info["image_path"])
        image_info["ori_img"] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_info["wh"] = image.shape[:-1][::-1]

        return image_info


@HOOKS.register_module()
class run_model(Hook):
    def __init__(self, work_dir=None, **kwargs) -> None:
        self.save_folder = work_dir

    def before_run(self, runner):
        self.base = runner.base
        # work_dir = runner.work_dir
        # if not self.save_folder:
        #     self.save_folder = work_dir
        # os.makedirs(work_dir, exist_ok=True)

    def before_epoch(self, runner):
        if self.base == "image":
            image_infos = runner.image_infos
            runner.generator = ImgGenerator(image_infos)

        elif self.base == "video":
            cap_img = VideoCapture(runner.image_infos)
            runner.generator = cap_img
    
    def after_epoch(self, runner):
        runner.model.restart()
