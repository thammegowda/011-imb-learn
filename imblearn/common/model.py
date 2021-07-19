#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 7/19/21

from torch import nn
from abc import abstractmethod

class BaseModel(nn.Module):

    @classmethod
    @abstractmethod
    def Trainer(cls):
        raise NotImplemented()
