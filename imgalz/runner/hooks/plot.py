import cv2
import numpy as np
from imgalz.utils import imread

from .hook import HOOKS, Hook
from imgalz.utils.colors import Colors

@HOOKS.register_module()
class plot(Hook):
    def __init__(self, key=None,**kwargs) -> None:
        self.kwargs = kwargs
        self.colors = Colors()
        self.key = str(key) if not key is None else None
        
    def before_run(self, runner):
        self.class_name = runner.class_name
        self.base = runner.base
        
    def before_iter(self, runner):
        if "plot_img" not in runner.image_info:
            image_info = runner.image_info
            image = image_info.get("ori_img", image_info.get("image_path", None))
            if isinstance(image, np.ndarray):
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image = imread(image)
            image_info["plot_img"] = image.copy()
    
        

