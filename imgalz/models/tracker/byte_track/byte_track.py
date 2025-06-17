import numpy as np

from .tracker.byte_tracker import BYTETracker
from imgalz import ltwh2xyxy


class ByteTrack:
    def __init__(
        self, min_box_area: int = 10, aspect_ratio_thresh: float = 3.0,*args,**kwargs
    ) -> None:

        self.min_box_area = min_box_area
        self.aspect_ratio_thresh = aspect_ratio_thresh
        self.min_box_area = min_box_area

        self.tracker = BYTETracker(*args,**kwargs)

    def track(self, bgr_img, dets_xyxy) -> tuple:

        if isinstance(dets_xyxy, np.ndarray) and len(dets_xyxy) > 0:
            bboxes_xyxy, ids, scores, cls = self._tracker_update(
                dets_xyxy,
                bgr_img,
            )
        track_info = {
            "bbox_ltrb": bboxes_xyxy,
            "ids": ids,
            "scores": scores,
            "class_ids": cls,
        }
        return track_info

    def _tracker_update(self, dets: np.ndarray, bgr_img: np.ndarray):
        online_targets = []

        if dets is not None:
            online_targets = self.tracker.update(dets)

        online_xyxys = []
        online_ids = []
        online_scores = []
        online_clss = []
        for online_target in online_targets:
            tlwh = online_target.tlwh
            track_id = online_target.track_id
            vertical = tlwh[2] / tlwh[3] > self.aspect_ratio_thresh
            if tlwh[2] * tlwh[3] > self.min_box_area and not vertical:
                online_xyxys.append(ltwh2xyxy(tlwh))
                online_ids.append(track_id)
                online_scores.append(online_target.score)
                online_clss.append(online_target.cls)
        return (
            np.array(online_xyxys),
            np.array(online_ids),
            np.array(online_scores),
            np.array(online_clss),
        )
