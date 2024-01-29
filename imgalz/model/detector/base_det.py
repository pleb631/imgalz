from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict

class BaseDet(ABC):
    def __init__(self, filter_class=[], model_name=None,require_keys=-1,*args, **kwargs):
        filter_class = list(map(int, filter_class))
        self.filter_class = filter_class
        self.model_name = model_name
        self.require_keys = require_keys

    @abstractmethod
    def forward(self):
        pass

    def __call__(self, data):
        result_dict = self.forward(data)
        if len(self.filter_class) > 0:
            boxes = result_dict["bbox_ltrb"]
            items= result_dict.items()
            
            filter_item = defaultdict(list)
            for i,box in enumerate(boxes):
                if int(box[5]) in self.filter_class:
                    for item in items:
                        filter_item[item[0]].append(item[1][i])
            
            result_dict = dict(filter_item)
            for key in result_dict.keys():
                result_dict[key] = np.array(result_dict[key])

        return result_dict
