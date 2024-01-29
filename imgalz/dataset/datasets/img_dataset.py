import glob
from itertools import chain
import os

from ..builder import DATASETS


@DATASETS.register_module()
class ImgDataset:
    formats = [".jpg", ".png", ".jpeg"]
    base = "image"

    def __init__(self, data_root, *args, **kwargs):
        super(ImgDataset, self).__init__()

        self.data_root = data_root
        self.data_infos = self.load_data_list()

    def load_data_list(
        self,
    ):
        if os.path.isfile(self.data_root):
            return [self.data_root]
        data_infos = sorted(
            chain(
                *[
                    glob.glob(
                        os.path.join(self.data_root, f"**{os.sep}*{f}"), recursive=True
                    )
                    for f in ImgDataset.formats
                ]
            )
        )
        return data_infos

    def prepare_data(self, idx):
        image_path = str(self.data_infos[idx])
        info = {"image_path": image_path, "data_root": self.data_root}

        return info

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        return self.prepare_data(index)
