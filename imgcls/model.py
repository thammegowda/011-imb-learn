

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

    last_layer = dict(
        resnext50_32x4d = 'fc',
        resnet18 = 'fc',
        wide_resnet50_2 = 'fc',
        inception_v3 = 'fc',
        googlenet='fc',
        shufflenet_v2_x1_0='fc',
        mobilenet_v2='fc',
        mobilenet_v3_large='fc',
        densenet161 = 'classifier',
        alexnet = 'classifier[-1]',
        vgg16 = 'classifier[-1]',
        mobilenet_v3_small = 'classifier[-1]',
        mnasnet1_0 = 'classifier[-1]',
    )
    @classmethod
    def get_model(cls, name, pretrained=True, remove_last_layer=True):
        assert name in cls.last_layer, f'{name} unknown. Options are : {cls.last_layer.keys()}'
        model = getattr(torchvision.models, name)(pretrained=pretrained)
        removed_layer = None
        if remove_last_layer:
            layer_name = cls.last_layer[name]
            last_index = False
            if '[-1]' in layer_name:
                layer_name = layer_name.replace('[-1]', '')
                last_index = True
            assert hasattr(model, layer_name), f'expected {layer_name} attrib in {type(model)}'
            removed_layer = getattr(model, layer_name)
            if last_index:
                removed_layer = removed_layer[-1]
                getattr(model, layer_name)[-1] = nn.Identity()
            else:
                setattr(model, layer_name, nn.Identity())
            assert isinstance(removed_layer, nn.Linear)
        return model, removed_layer

    @classmethod
    def cache(cls, name):
        if name not in cls.__cache:
            log.info(f"initializing parent model: {name}")
            model, last_layer = cls.get_model(name=name, pretrained=True, remove_last_layer=True)
            model.eval()
            model = model.to(device)
            cls.__cache[name] = model
        return cls.__cache[name]


class ImageClassifier(nn.Module):

    def __init__(self, n_classes, parent, dropout=0.3, intermediate=None):
        super().__init__()
        intermediate = intermediate or 2 * n_classes
        _, last_layer = PreTrainedModels.get_model(name=parent, pretrained=False, remove_last_layer=True)
        pre_classes = last_layer.in_features
        log.info(f"Removed last layer from {parent}; input dimension = {pre_classes}")
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(pre_classes, intermediate),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(intermediate, n_classes)
        )

        self.fc = nn.Linear(pre_classes, n_classes)
        self.parent = parent

    def forward(self, xs: Tensor):
        with torch.no_grad():
            feats = PreTrainedModels.cache(self.parent)(xs)
        return self.fc(feats)

