

import torch
import torchvision
from torch import Tensor, nn

from . import log, device


class PreTrainedModels:
    """
    Pretrained parent model: this is placed outside of child module to exclude from its graph

    To see all the available models: https://pytorch.org/vision/stable/models.html#classification
    """
    # lazy loading
    __cache = {}

    @classmethod
    def get_model(cls, name, pretrained=True):
        assert hasattr(torchvision.models, name) and callable(getattr(torchvision.models, name)), \
            f'{name} is either unknown or invalid'
        return getattr(torchvision.models, name)(pretrained=pretrained)

    @classmethod
    def cache(cls, name):
        if name not in cls.__cache:
            log.info(f"initializing parent model: {name}")
            model = cls.get_model(name=name, pretrained=True)
            model.eval()
            model = model.to(device)
            cls.__cache[name] = model
        return cls.__cache[name]


class ImageClassifier(nn.Module):

    def __init__(self, n_classes, parent, pre_classes=1000):
        super().__init__()
        self.fc = nn.Linear(pre_classes, n_classes)
        self.parent = parent

    def forward(self, xs: Tensor):
        with torch.no_grad():
            feats = PreTrainedModels.cache(self.parent)(xs)
        return self.fc(feats)

