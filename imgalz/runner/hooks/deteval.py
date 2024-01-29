import numpy as np
import pprint
from imgalz.utils.metrics.DetMetrics import DetMetrics, process_batch
from imgalz.utils import COCO_METAINFO, xywh2xyxy

from .hook import HOOKS, Hook


@HOOKS.register_module()
class deteval(Hook):
    def __init__(self, key=None,**kwargs) -> None:
        self.stats = []
        self.class_name = None
        self.metrics = None
        self.kwargs = kwargs
        self.key = str(key) if not key is None else -1

    def before_run(self, runner):
        self.class_name = runner.class_name
        self.metrics = DetMetrics(names=self.class_name, plot=True, **self.kwargs)

    def after_iter(self, runner):
        if isinstance(self.key,int):
            self.key = runner.result["result"]['name'][self.key]

        result = runner.result["result"][self.key]
        
        preds = result["bbox_ltrb"]

        gt = []
        if "label_cxyxy" in runner.result:
            label_cxyxy = runner.result["label_cxyxy"]
            w, h = runner.result["wh"]
            label_cxyxy = label_cxyxy * [1, w, h, w, h]
            gt = label_cxyxy

        correct_bboxes = np.zeros((len(preds), 10))
        matches = np.empty((0, 2))
        stat = (correct_bboxes, np.empty(0), np.empty(0), np.empty(0))

        if len(gt) > 0 and len(preds) == 0:
            stat = (correct_bboxes, np.empty(0), np.empty(0), gt[:, 5])
        elif len(preds) > 0 and len(gt) > 0:
            correct_bboxes, matches = process_batch(preds, gt)
            stat = (correct_bboxes, preds[:, 4], preds[:, 5], gt[:, 0])
        elif len(preds) > 0 and len(gt) == 0:
            stat = (correct_bboxes, preds[:, 4], preds[:, 5], np.empty(0))

        self.stats.append(stat)

    def after_run(self, runner):
        stats = [np.concatenate(x, 0) for x in zip(*self.stats)]
        if len(stats) and stats[0].any():
            self.metrics.process(*stats)

        result = self.metrics.results_dict

        pprint.pprint(result)
