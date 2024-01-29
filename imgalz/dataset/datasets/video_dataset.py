import glob
from itertools import chain
import os

from ..builder import DATASETS


@DATASETS.register_module()
class VideoDataset:
    formats = [".mp4", ".avi", ".mkv"]
    base = "video"

    def __init__(self, data_root, *args, **kwargs):
        super(VideoDataset, self).__init__()

        self.data_root = data_root
        self.data_infos = self.load_data_list()

    def load_data_list(
        self,
    ):
        if os.path.isfile(self.data_root):
            return [self.data_root]
        if self.data_root == "webcam":
            return [0]
        data_infos = sorted(
            chain(
                *[
                    glob.glob(
                        os.path.join(self.data_root, f"**{os.sep}*{f}"), recursive=True
                    )
                    for f in VideoDataset.formats
                ]
            )
        )
        return data_infos

    def prepare_data(self, idx):
        video_path = self.data_infos[idx]
        info = {"video_path": video_path, "data_root": self.data_root}

        return info

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        return self.prepare_data(index)
