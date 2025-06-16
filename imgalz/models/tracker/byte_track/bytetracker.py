import numpy as np

from .tracker.byte_tracker import BYTETracker



def tlwh_to_xyxy(tlwh):
    """ " Convert tlwh to xyxy"""
    x1 = tlwh[0]
    y1 = tlwh[1]
    x2 = tlwh[2] + x1
    y2 = tlwh[3] + y1
    return [x1, y1, x2, y2]



class ByteTrack:
    def __init__(
        self, min_box_area: int = 10, aspect_ratio_thresh: float = 3.0) -> None:

        self.min_box_area = min_box_area
        self.aspect_ratio_thresh = aspect_ratio_thresh
        self.min_box_area = min_box_area

        self.tracker = BYTETracker(frame_rate=30)

    def track(self, bgr_img,dets_xyxy) -> tuple:

        image = bgr_img
        image_info = {"width": image.shape[0], "height": image.shape[1]}
        class_ids = []
        ids = []
        bboxes_xyxy = []
        scores = []

        if isinstance(dets_xyxy, np.ndarray) and len(dets_xyxy) > 0:
            bboxes_xyxy, ids, scores = self._tracker_update(
                dets_xyxy,
                image_info,
            )
        track_info = {
            "bbox_ltrb": bboxes_xyxy,
            "ids": ids,
            "scores": scores,
            "class_ids": class_ids,
        }
        return track_info

    def _tracker_update(self, dets: np.ndarray, image_info: dict):
        online_targets = []
        class_id = 0
        if dets is not None:
            online_targets = self.tracker.update(dets)

        online_xyxys = []
        online_ids = []
        online_scores = []
        for online_target in online_targets:
            tlwh = online_target.tlwh
            track_id = online_target.track_id
            vertical = tlwh[2] / tlwh[3] > self.aspect_ratio_thresh
            if tlwh[2] * tlwh[3] > self.min_box_area and not vertical:
                online_xyxys.append(tlwh_to_xyxy(tlwh))
                online_ids.append(track_id)
                online_scores.append(online_target.score)
        return np.array(online_xyxys), np.array(online_ids), np.array(online_scores)
