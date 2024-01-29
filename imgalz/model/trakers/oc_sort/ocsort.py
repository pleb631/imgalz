import numpy as np
from .tracker.ocsort import OCSort

from imgalz.model import MODELS
from imgalz.model.trakers.base_tracker import BaseTracker


@MODELS.register_module()
class OcSort(BaseTracker):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tracker = OCSort(det_thresh=0.2)

    def forward(self, data) -> tuple:
        dets_xyxy = data["bbox_ltrb"]
        image = data["ori_img"]

        if isinstance(dets_xyxy, np.ndarray) and len(dets_xyxy) > 0:
            dets = self.tracker.update(dets_xyxy)
            bbox_xyxy = dets[:, :4]
            ids = dets[:, 4]
            class_ids = dets[:, 5]
            scores = dets[:, 6]

            track_info = {
                "bbox_ltrb": bbox_xyxy,
                "ids": ids,
                "scores": scores,
                "class_ids": class_ids,
            }

        return track_info
