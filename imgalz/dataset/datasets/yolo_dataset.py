import os
import numpy as np


from .img_dataset import ImgDataset
from ..builder import DATASETS
from imgalz.utils import xywh2xyxy


def read_yolo_txt(txt_path):
    txt_file = open(txt_path, "r")
    label_xywh = []
    for line in txt_file:
        line = line.split()
        cls_id = int(line[0])
        box_x = float(line[1])
        box_y = float(line[2])
        box_w = float(line[3])
        box_h = float(line[4])
        xyxy = xywh2xyxy([box_x, box_y, box_w, box_h])

        label_xywh.append([cls_id, *xyxy])
    txt_file.close()

    return label_xywh


@DATASETS.register_module()
class YoloDataset(ImgDataset):
    def __init__(self, data_root, *args, **kwargs):
        super(YoloDataset, self).__init__(data_root, *args, **kwargs)

        self.data_root = data_root

    def prepare_data(self, idx):
        image_path = str(self.data_infos[idx])
        label_path = image_path.replace(
            f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
        )
        label_path = os.path.splitext(label_path)[0] + ".txt"
        info = {"image_path": image_path, "data_root": self.data_root}
        if os.path.exists(label_path):
            info["label_path"] = label_path
            label_cxyxy = read_yolo_txt(label_path)
            info["label_cxyxy"] = np.array(label_cxyxy)

        return info
