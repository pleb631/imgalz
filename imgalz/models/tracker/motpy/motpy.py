try:
    from motpy import Detection, MultiObjectTracker
except:
    raise ImportError(
        "Motpy is not installed. Please install it with `pip install motpy`"
    ) from None
import numpy as np




class Motpy:
    def __init__(self, dt=0.1) -> None:
        self.tracker = MultiObjectTracker(dt=dt)
        self.obj_count = 0
        self.uuids = {}

    def track(self, bgr_img,dets_xyxy) -> tuple:

        image = bgr_img
        
        class_ids = []
        ids = []
        bboxes_xyxy = []
        scores = []
        if isinstance(dets_xyxy, np.ndarray) and len(dets_xyxy) > 0:
            self.tracker.step(
                detections=[
                    Detection(box=box[:4], score=box[4], class_id=box[5])
                    for box in dets_xyxy
                ]
            )
            bboxes_xyxy, ids, scores, class_ids = self._tracker_update()

        track_info = {
            "bbox_ltrb": bboxes_xyxy,
            "ids": ids,
            "scores": scores,
            "class_ids": class_ids,
        }

        return track_info

    def _tracker_update(self):
        bboxes_xyxy = []
        class_ids = []
        scores = []
        ids = []

        tracked_objects = self.tracker.active_tracks()
        for obj in tracked_objects:
            if obj[0] in self.uuids:
                obj_id = self.uuids[obj[0]]
            else:
                self.obj_count += 1
                self.uuids[obj[0]] = self.obj_count
                obj_id = self.uuids[obj[0]]

            bboxes_xyxy.append(obj[1:2][0].tolist())
            class_ids.append(obj[3])
            scores.append(obj[2])
            ids.append(obj_id)
        return np.array(bboxes_xyxy), np.array(ids), np.array(scores), np.array(class_ids)
