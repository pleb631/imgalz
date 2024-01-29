import cv2

from .hook import HOOKS, Hook

@HOOKS.register_module()
class exporter(Hook):
    def __init__(self, save_dir=None,**kwargs) -> None:
        self.save_folder = save_dir
        self.kwargs = kwargs

        
    def before_run(self, runner):
        self.base = runner.base
        
    
    def after_iter(self, runner):
        
        plot_img = runner.result.get('plot_img',None)
        if not plot_img is None:
            
            cv2.imshow("im", plot_img)
            if self.base == "image":
                cv2.waitKey(0)
            else:
                cv2.waitKey(1)

