#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 5/3/21

from typing import List, Union
import math
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.loss import CrossEntropyLoss, _Loss, _WeightedLoss
from imblearn import log, register, LOSS
import abc

DEF_EFF_BETA = 0.999  # effective number of samples


class Loss(_Loss):

    @property
    @abc.abstractmethod
    def input_type(self) -> str:
        raise NotImplementedError()


class WeightedLoss(_WeightedLoss, Loss):

    def __init__(self, exp, reduction='mean', weight_by=None, weight=None,
                 eff_frequency: bool = False, eff_beta: float = DEF_EFF_BETA):
        self.exp = exp
        if weight_by:
            assert weight is None, f'weight_by and weight are mutually exclusive'
            if isinstance(weight, str):
                raise Exception(f'weight={weight} is invalid; do you mean weight_by={weight}?')

            freqs: Tensor = torch.tensor(exp.cls_freqs, dtype=torch.float, requires_grad=False)
            if eff_frequency:
                # https://arxiv.org/abs/1901.05555
                assert 0 <= eff_beta < 1, f'eff_beta should be in [0,1), but given {eff_beta}'
                log.info(f"Using 'effective number of samples' as frequencies with β={eff_beta}")
                eff_beta_pow_freq = torch.tensor(eff_beta).pow(freqs)
                freqs = (1 - eff_beta_pow_freq) / (1 - eff_beta)

            min_freq = freqs.min()
            assert min_freq > 1  # at least ore than 1, just so log(f) and 1/f functions are happy
            known = ('inverse_frequency', 'inverse_log', 'inverse_sqrt', 'information_content')
            if weight_by == 'inverse_frequency':
                weight = freqs.median() / freqs
            elif weight_by == 'inverse_log':
                logs = freqs.log()
                weight = logs.median() / logs
            elif weight_by == 'inverse_sqrt':
                sqrts = freqs.sqrt()
                weight = sqrts.median() / sqrts
            elif weight_by == 'information_content':
                # https://en.wikipedia.org/wiki/Information_content
                probs = freqs / freqs.sum()
                weight = -probs.log2()
            elif isinstance(weight_by, list) and len(weight_by) == self.n_classes:
                weight = torch.tensor(weight_by, dtype=torch.float)
            else:
                raise Exception(f'criterion.args.weight={weight_by} unknown; known={known}')

        assert weight is None or isinstance(weight, Tensor)
        super(WeightedLoss, self).__init__(weight=weight, reduction=reduction)
        self.weight_by = weight_by
        if weight is not None:
            top = 20
            msg = ', '.join(f'{c}: {w:g}' for c, w in zip(exp.classes[:top], weight[:top]))
            if len(self.exp.classes) > top:
                msg += '...[truncated]'
            log.info(f'class weights = {msg}')


@register(LOSS)
class CrossEntropy(WeightedLoss):

    def __init__(self, *args, ignore_index=-100, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_index = ignore_index

    @property
    def input_type(self) -> str:
        return 'logits'

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)


@register(LOSS)
class FocalLoss(nn.Module):
    """
    Implements Lin etal https://arxiv.org/abs/1708.02002
    """

    def __init__(self, exp, reduction='mean', gamma: float = 0.0):
        super().__init__()
        self.exp = exp
        assert reduction in ('none', None, 'mean')
        self.reduction = reduction
        self.gamma = gamma
        assert gamma > 0.0

    @property
    def input_type(self):
        return 'logits'

    def forward(self, input: Tensor, target: Tensor):
        r"""
        :param input: :math:`(N, C)` where `C = number of classes`
        :param target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`
        :return:
        """
        input = input.softmax(dim=1)
        # extract probs for the correct class (i.e y==1 in one-hot)
        target = target.view(-1, 1)
        probs = input.gather(dim=1, index=target)
        losses = -((1 - probs).pow(self.gamma) * probs.log())
        if self.reduction in ('none', None):
            return losses
        elif self.reduction == 'mean':
            return losses.mean()
        else:
            raise Exception('duh, this should not be happening...')



def smooth_labels(target, C):
    assert len(target.shape) == 1
    N = len(target)
    assert len(target) == N
    assert target.max() < C
    assert 0 <= target.min()

    # expand [N] -> [N,C]
    full = torch.zeros(N, C, dtype=torch.float)
    full.scatter_(0, target.view(1, -1), 1.0)
    return full
