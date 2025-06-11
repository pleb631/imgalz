import cv2
import numpy as np



def xywh2xyxy(xywh):
    xmin = xywh[0] - xywh[2] / 2
    ymin = xywh[1] - xywh[3] / 2
    xmax = xmin + xywh[2]
    ymax = ymin + xywh[3]
    return [xmin, ymin, xmax, ymax]


def xyxy2xywh(xyxy):
    x_center = (xyxy[0] + xyxy[2]) / 2
    y_center = (xyxy[1] + xyxy[3]) / 2
    width = xyxy[2] - xyxy[0]
    height = xyxy[3] - xyxy[1]
    return [x_center, y_center, width, height]



def imread(path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(path, np.uint8), flags)


def cv_show_image(title,image,type='bgr'):
    if type=='rgb':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try:
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyWindow(title)
    except:
        cv2.imwrite(title+'.jpg', image)