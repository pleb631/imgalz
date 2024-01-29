import numpy as np
from .hook import HOOKS
from .plot import plot



def plt_masks(masks, colors, image, alpha=0.5):
    """
    Plot masks on image.

    Args:
        masks (array): Predicted masks, shape: [n, h, w]
        colors (List[List[Int]]): Colors for predicted masks, [[r, g, b] * n]
        image (array):  shape: [h, w,3], range: [0, 255]
        alpha (float): Mask transparency: 0.0 fully transparent, 1.0 opaque
    """

    image = image.astype(np.float64)

    colors = colors[:, None, None]  # shape(n,1,1,3)

    masks = masks[...,None]  # shape(n,h,w,1)
    masks_color = masks * (colors * alpha)  # shape(n,h,w,3)

    inv_alpha_masks = (1 - masks * alpha).cumprod(0)  # shape(n,h,w,1)
    mcs = masks_color.max(0)  # shape(n,h,w,3)

    image = image * inv_alpha_masks[-1]+ mcs
    image = np.clip(image, 0, 255)
        

    return image.astype(np.uint8)
    
    
             
@HOOKS.register_module()
class plot_seg(plot):
    def __init__(self,**kwargs) -> None:
        super().__init__(**kwargs)
        
       
    def after_iter(self, runner):
        if self.key is None:
            self.key = runner.result["result"]['name'][-1]

        result = runner.result["result"][self.key]
        
        plot_img = runner.result['plot_img']
        masks = result.get('masks',None)
        if masks is None:
            return
        bbox_ltrb = result.get("bbox_ltrb", None)
        if bbox_ltrb is None:
            idx =  range(len(masks))
        else:  
            idx = bbox_ltrb[:,5].tolist()
        color=np.array([self.colors(x, True) for x in idx])

        plot_img = plt_masks(masks,color,plot_img,)
        runner.result['plot_img'] = plot_img



