import torch


class FakeTransformation(object):
    def __init__(self):
        pass

    @property
    def summarize(self):
        return {
            "name": "FakeTransformation",
            "params": {}
        }

    def __call__(self, x: torch.Tensor):
        return x


__all__ = ["FakeTransformation"]
