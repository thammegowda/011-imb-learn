#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 5/3/21

import torch
from typing import List
from  torch.nn.modules.loss import CrossEntropyLoss
from imblearn import log, register, LOSS


@register(LOSS)
class CrossEntropy(CrossEntropyLoss):

    def __init__(self, exp, *args, weight_by=None, weight=None, **kwargs):
        if weight_by:
            assert weight is None, f'weight_by and weight are mutually exclusive'
            cls_freqs: List[int] = exp.cls_freqs
            freqs = torch.tensor(cls_freqs, dtype=torch.float, requires_grad=False)
            min_freq = freqs.min()
            assert min_freq > 1  # at least 2 samples
            known = ('inverse_frequency', 'inverse_log', 'information_content')
            if weight_by == 'inverse_frequency':
                weight = freqs.median() / freqs
            elif weight_by == 'inverse_log':
                weight = 1 / freqs.log()
            elif weight_by == 'information_content':
                # https://en.wikipedia.org/wiki/Information_content
                probs = freqs / freqs.sum()
                weight = -probs.log2()
            elif isinstance(weight_by, list) and len(weight_by) == self.n_classes:
                weight = torch.tensor(weight_by, dtype=torch.float)
            else:
                raise Exception(f'criterion.args.weight={weight_by} unknown; known={known}')
            log.info(f'class weights = {dict(zip(exp.classes, weight.tolist()))}')
        print(weight, '<<')
        super().__init__(*args, weight=weight, **kwargs)
