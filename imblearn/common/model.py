#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 7/19/21

from torch import nn
from abc import abstractmethod, ABC

from imblearn.common.exp import BaseTrainer


class BaseModel(nn.Module, ABC):

    @classmethod
    @abstractmethod
    def Trainer(cls) -> BaseTrainer:
        raise NotImplemented()
