#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 5/7/21

from torch import optim
from torch.nn import CrossEntropyLoss
import functools
import re

MODEL = 'model'
OPTIMIZER = 'optimizer'
SCHEDULE = 'schedule'
LOSS = 'loss'

registry = {
    MODEL: dict(),
    OPTIMIZER: dict(
        adam=optim.Adam,
        sgd=optim.SGD,
        adagrad=optim.Adagrad,
        adam_w=optim.AdamW,
        adadelta=optim.Adadelta,
        sparse_adam=optim.SparseAdam),
    SCHEDULE: dict(),
    LOSS: dict(
        cross_entropy=CrossEntropyLoss),
}


def snake_case(word):
    """
    Converts a word (from CamelCase) to snake Case
    :param word:
    :return:
    """
    word = re.sub(r"([A-Z]+)([A-Z][a-z])", r'\1_\2', word)
    word = re.sub(r"([a-z\d])([A-Z])", r'\1_\2', word)
    word = word.replace("-", "_")
    return word.lower()


def register(kind, name=None):
    """
    A decorator for registering modules
    :param kind: what kind of component :py:const:MODEL, :py:const:OPTIMIZER, :py:const:SCHEDULE
    :param name: (optional) name for this component
    :return:
    """
    assert kind in registry

    def _wrap_func(func):
        name2 = name or snake_case(func.__name__)
        registry[kind][name2] = func

        @functools.wraps(func)
        def _wrap_args(*args, **kwargs):
            return func(*args, **kwargs)

        return _wrap_args

    return _wrap_func


def register_all():
    # import so register() calls can happen; Not sure if there is a better way to accomplish
    from importlib import import_module
    modules = [
        'imblearn.common.schedule',
        'imblearn.common.loss',
        'imblearn.imgcls.model'
    ]
    for name in modules:
        import_module(name)