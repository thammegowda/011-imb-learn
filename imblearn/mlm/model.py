#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 5/19/21

from torch import nn
from imblearn import register, MODEL, log
from transformers import RobertaForMaskedLM, RobertaConfig


class BalBert(RobertaForMaskedLM):

    @staticmethod
    @register(kind=MODEL, name='bal_bert')
    def new_roberta(*args, **kwargs):
        config = RobertaConfig(**kwargs)
        return BalBert(*args, config=config)
