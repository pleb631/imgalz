import tqdm
import mmengine


from .priority import get_priority
from .hooks import HOOKS, Hook
from imgalz.utils import COCO_METAINFO

class Runner(object):
    def __init__(self, model,dataset,class_name=None,**kwargs):
        super(Runner, self).__init__()
        
        self.model = model
        self.dataset = dataset
        self.base = dataset.base
        self.kwargs = kwargs
        self.stats = []
        
        if class_name is None:
            class_name = {i: name for i, name in enumerate(COCO_METAINFO)}

        self.class_name = class_name
        self._hooks=[]


    def run(self):
        img_num = len(self.dataset)
        self.call_hook('before_run')
        for i in tqdm.tqdm(range(img_num)):
            image_infos = self.dataset[i]
            self.image_infos = image_infos
            self.call_hook('before_epoch')
            for image_info in self.generator:
                self.image_info = image_info
                self.call_hook('before_iter')
                self.result = self.model(image_info)
                self.call_hook('after_iter')
            self.call_hook('after_epoch')
        self.call_hook('after_run')
               
    def register_hook(self,
                      hook: Hook,
                      priority = 'NORMAL') -> None:
        """Register a hook into the hook list.

        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.

        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority  # type: ignore
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:  # type: ignore
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)
            
            
    def register_hook_from_cfg(self, hook_cfg) -> None:
        """Register a hook from its cfg.

        Args:
            hook_cfg (dict): Hook config. It should have at least keys 'type'
              and 'priority' indicating its type and priority.

        Note:
            The specific hook class to register should not use 'type' and
            'priority' arguments during initialization.
        """
        hook_cfg = hook_cfg.copy()
        priority = hook_cfg.pop('priority', 'NORMAL')
        hook = mmengine.build_from_cfg(hook_cfg, HOOKS)
        self.register_hook(hook, priority=priority)
            
    def call_hook(self, fn_name: str) -> None:
        """Call all hooks.

        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
        """
        self.run_mode = "continue"
        for hook in self._hooks:
            if self.run_mode=="pass":
                break
            getattr(hook, fn_name)(self)
        