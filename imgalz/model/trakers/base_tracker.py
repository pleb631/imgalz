from abc import ABC, abstractmethod

class BaseTracker(ABC):
    def __init__(self, model_name=None,require_keys=-1,*args, **kwargs):
        self.model_name = model_name
        self.require_keys = require_keys

    @abstractmethod
    def forward(self):
        pass

    def __call__(self, data):
        key = data["result"]['name'][self.require_keys]
        info = data["result"][key]
        dets_xyxy = info["bbox_ltrb"]
        image = data["ori_img"]
        inindata = {"bbox_ltrb":dets_xyxy,"ori_img":image}
        result_dict = self.forward(inindata)
        return {"track_info":result_dict}
