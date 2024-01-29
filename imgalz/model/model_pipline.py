from mmengine import build_from_cfg
from collections.abc import Sequence
from . import MODELS
from imgalz.model.trakers import ByteTrack, Motpy, NorFair, OCSort


class ModelPipline:
    """Compose a data pipeline with a sequence of model.

    Args:
        model (list[dict | callable]): Either config
          dicts of model or model objects.
    """

    def __init__(self, model):
        assert isinstance(model, Sequence)
        self.models = []
        self.tl = model
        self.need_restart = False
        self.load()

    def load(
        self,
    ):
        name = []
        for i, transform in enumerate(self.tl):
            if transform in [ByteTrack, Motpy, NorFair, OCSort]:
                self.need_restart = True
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, MODELS)
                if not getattr(transform, "model_name"):
                    transform.model_name = str(i)
                else:
                    if str(transform.model_name) in name:
                        raise IOError(f"{transform.model_name} is repeated")
                name.append(transform.model_name)
                self.models.append(transform)
            # elif callable(transform):
            #     self.models.append(transform)
            else:
                raise TypeError(
                    "model must be callable or a dict, but got" f" {type(transform)}"
                )

    def __call__(self, data):
        """Call function to apply model sequentially.

        Args:
            data (dict): A result dict contains the data to model.

        Returns:
            dict: Transformed data.
        """
        if data is None:
            return None
        data["result"] = dict(name=[])
        for t in self.models:
            res = t(data.copy())
            data["result"][t.model_name] = res
            data["result"]["name"].append(t.model_name)

        return data


    def restart(
        self,
    ):
        for i, (transform, model) in enumerate(zip(self.tl, self.models)):
            if isinstance(model, (ByteTrack, Motpy, NorFair, OCSort)):
                m = build_from_cfg(transform, MODELS)
                m.model_name = self.models[i].model_name
                self.models[i] = m
