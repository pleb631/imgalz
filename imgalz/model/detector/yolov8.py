import torch
import numpy as np
import torchvision
from imgalz.model import MODELS
from .yolov5 import YOLOv5,xywh2xyxy,scale_boxes



@MODELS.register_module()
class YOLOv8(YOLOv5):
    def __init__(
        self,
        nc=80,
        *args,
        **kwargs
    ):
        super(YOLOv8, self).__init__(*args,**kwargs)

        self.nc = nc
        
    
    def PostProcess(self,boxes, conf_thres=0.24, iou_thres=0.45):
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 300000  # maximum number of boxes into torchvision.ops.nms()
        max_det = 300
        agnostic = False

        xc = boxes[4:4+self.nc,:].max(0) > conf_thres
        boxes = boxes[:,xc].T

        box, clss,mask,_ = np.split(boxes,[4,4+self.nc,boxes.shape[1]+1],axis=1)
        
        

        box = xywh2xyxy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        conf = clss.max(1, keepdims=True)
        j = clss.argmax(1, keepdims=True)
        x = np.concatenate((box, conf, j,mask), 1)[conf.reshape(-1) > conf_thres]
        # Check shape
        n = x.shape[0]  # number of boxes
        # if not n:  # no boxes
        #     continue
        x = x[x[:, 4].argsort()[::-1][:max_nms]]  # sort by confidence and remove excess boxes
        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        boxes = torch.from_numpy(boxes)
        scores = torch.from_numpy(scores)
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        x =  x[i]
        if len(x.shape)==1:
            x=x.reshape(1,-1)
            
        return x
