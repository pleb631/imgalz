from mmengine import Registry

MODELS = Registry("model")


from .detector import YOLOv5,YOLOv8,YOLOv8_seg,YOLOv8_pose
from .trakers import ByteTrack,Motpy,NorFair,OCSort
from .model_pipline import ModelPipline

