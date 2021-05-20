#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 5/19/21

from torch import nn
from imblearn import register, MODEL, log
from transformers import RobertaForMaskedLM, RobertaConfig

@register(MODEL)
class BalBert(RobertaForMaskedLM):

    def __init__(self, **kwargs):
        config = RobertaConfig(**kwargs)
        super().__init__(config=config)