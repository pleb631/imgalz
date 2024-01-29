from norfair import Detection, Tracker
import numpy as np

from imgalz.model import MODELS
from imgalz.model.trakers.base_tracker import BaseTracker

@MODELS.register_module()
class NorFair(BaseTracker):
    def __init__(self, max_distance_between_points=30,*args,**kwargs) -> None:
        super().__init__(*args,**kwargs)
        self.tracker = Tracker(
            distance_function=self._euclidean_distance,
            distance_threshold=max_distance_between_points,
        )

    def _euclidean_distance(self, detection, tracked_object):
        return np.linalg.norm(detection.points - tracked_object.estimate)

    def forward(self, data) -> tuple:
        dets_xyxy = data["bbox_ltrb"]
        image = data["ori_img"]

        class_ids = []
        ids = []
        bboxes_xyxy = []
        scores = []

        if isinstance(dets_xyxy, np.ndarray) and len(dets_xyxy) > 0:
            dets_xyxy = [
                Detection(
                    np.array([(box[2] + box[0]) / 2, (box[3] + box[1]) / 2]), data=box
                )
                for box in dets_xyxy
                # if box[-1] == 2
            ]

            bboxes_xyxy, ids, scores, class_ids = self._tracker_update(dets_xyxy)

        track_info = {
            "bbox_ltrb": bboxes_xyxy,
            "ids": ids,
            "scores": scores,
            "class_ids": class_ids,
        }

        return track_info

    def _tracker_update(self, dets_xyxy: list):
        bboxes_xyxy = []
        class_ids = []
        scores = []
        ids = []

        tracked_objects = self.tracker.update(detections=dets_xyxy)

        for obj in tracked_objects:
            det = obj.last_detection.data
            bboxes_xyxy.append(det[:4])
            class_ids.append(int(det[-1]))
            scores.append(int(det[-2]))
            ids.append(obj.id)
        return np.array(bboxes_xyxy), ids, scores, class_ids
