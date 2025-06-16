import numpy as np
from .tracker.ocsort import OCSort


class OcSort:
    def __init__(self) -> None:

        self.tracker = OCSort(det_thresh=0.2)

    def track(self, bgr_img,dets_xyxy) -> tuple:

        image = bgr_img

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
