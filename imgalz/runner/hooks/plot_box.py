import cv2
import numpy as np

from .hook import HOOKS
from .plot import plot


def compute_color_for_labels(label):
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def plt_bbox(img, box, obj_threshold=0, line_thickness=None,label_format="{score:.2f} {id}",txt_color=(255,255,255),box_color=[255,0,0],**kwargs):
    
    if box[4] < obj_threshold:
        return 0
    if isinstance(box, np.ndarray):
        box = box.tolist()
    
    tl = line_thickness or round(
        0.001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img,p1,p2,box_color,tl)
    if label_format:
        tf = max(tl - 1, 1)  # font thickness
        sf = tl / 3  # font scale
        
        score = box[4]
        id = int(box[5])
        label = label_format.format(score=score,id=id)
        
        w, h = cv2.getTextSize(label, 0, fontScale=sf, thickness=tf)[0]
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(img, p1, p2, box_color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
                img,
                label,
                (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                sf,
                txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA,
            )
    return img


@HOOKS.register_module()
class plot_box(plot):
    def __init__(self, plot_id=False,*args,**kwargs) -> None:
        super().__init__(*args,**kwargs)


        self.plot_id = plot_id
        
       
    def after_iter(self, runner):
        if self.key is None:
            self.key = runner.result["result"]['name'][-1]
            
        result = runner.result["result"][self.key]

            
        plot_img = runner.result['plot_img']

        pred = result.get("bbox_ltrb", [])
        if self.plot_id:
            if "track_info" in result:
                ids = result["track_info"]["ids"]
                scores = result["track_info"]["scores"]
                bbox_ltrb = result["track_info"]["bbox_ltrb"]
                if len(bbox_ltrb) > 0:
                    pred = np.concatenate(
                        (
                            np.array(bbox_ltrb),
                            np.array(scores)[..., None],
                            np.array(ids)[..., None],
                        ),
                        1,
                    )
                else:
                    pred = []
        for box in pred:
            if self.plot_id:
                color = compute_color_for_labels(int(box[5]))
            else:
                color = self.colors(int(box[5]), True)
            plot_img = plt_bbox(plot_img, box,box_color=color,**self.kwargs)
            
        runner.result['plot_img'] = plot_img
        
