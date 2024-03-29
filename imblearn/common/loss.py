#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 5/3/21

from typing import List, Union
import math
from dataclasses import dataclass
from pathlib import Path
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.loss import CrossEntropyLoss, _Loss, _WeightedLoss
from imblearn import log, register, LOSS, registry, device
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
        self.eff_frequency = eff_frequency
        self.eff_beta = eff_beta

        if weight_by:
            assert weight is None, f'weight_by and weight are mutually exclusive'
            if isinstance(weight, str):
                raise Exception(f'weight={weight} is invalid; do you mean weight_by={weight}?')
            weight = self.get_weight(weight_by)

        if weight is not None:
            assert isinstance(weight, Tensor)
            weight = weight.to(device)
        super(WeightedLoss, self).__init__(weight=weight, reduction=reduction)
        self.weight_by = weight_by

        if weight is not None:
            msg = '\n'.join(f'{c}\t{f}\t{w:g}' for c, f, w in zip(exp.classes, exp.cls_freqs, weight))
            log.info(f'class weights =\n {msg}')
            wt_file: Path = self.exp.work_dir / 'class-weights.tsv'
            if not wt_file.exists():
                log.info(f"Creating {wt_file}")
                wt_file.write_text(msg)


    def get_weight(self, weight_by):
        freqs: Tensor = torch.tensor(self.exp.cls_freqs, dtype=torch.float, requires_grad=False)
        if self.eff_frequency:
            # https://arxiv.org/abs/1901.05555
            assert 0 <= self.eff_beta < 1, f'eff_beta should be in [0,1), but given {self.eff_beta}'
            log.info(f"Using 'effective number of samples' as frequencies with β={self.eff_beta}")
            eff_beta_pow_freq = torch.tensor(self.eff_beta).pow(freqs)
            freqs = (1 - eff_beta_pow_freq) / (1 - self.eff_beta)
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
        return weight


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
class SmoothCrossEntropy(CrossEntropy):

    def __init__(self, *args, ignore_index=-100, smooth_epsilon=0.1, smooth_weight_by=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_index = ignore_index
        self.smooth_epsilon = smooth_epsilon
        self.smooth_weight = self.get_weight(weight_by=smooth_weight_by)\
            if smooth_weight_by else None

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        N, C = input.shape
        target = self.smooth_labels(target, C=C, epsilon=self.smooth_epsilon,
                                    weight=self.smooth_weight)
        assert input.shape == target.shape
        log_probs = input.log_softmax(dim=1)
        #[N, C] :  y_c * log(p_c)
        prod = torch.mul(target, log_probs)
        if self.weight is not None:
            assert len(self.weight) == C
            # [N, C] w_c *  y_c * log(p_c)
            prod = torch.mul(self.weight.view(1, C), prod)
        return -prod.mean()


    @classmethod
    def smooth_labels(cls, labels, C, epsilon, weight=None):
        """
        :param labels: labels [N] where each item is in range {0, 1, 2,... C-1}
        :param C: total number of classes
        :param epsilon: the magnitude of smoothing
        :param weight: distribute epsilon as per the weights
        :return:
        """
        assert len(labels.shape) == 1
        assert labels.max() < C
        assert 0 <= labels.min()
        assert 0 <= epsilon <= 1

        N = len(labels)
        labels = labels.view(N, 1)
        device = labels.device
        if weight is None:
            # take out epsilon and distribute evenly to all but one
            fill_value = epsilon / (C - 1)
            # expand [N] -> [N, C]
            full = torch.full((N, C), fill_value=fill_value, dtype=torch.float, device=device)
            full.scatter_(1, labels.type(torch.int64), 1 - epsilon)
        else:
            assert len(weight) == C
            weight = weight.to(device)
            full = (weight * epsilon).expand(N, C)            # [C] -> [N, C]
            peaks = torch.tensor(1 - epsilon, device=device).expand(N, 1)  # [N, 1]
            full = full.scatter_add(1, labels, peaks)             # inplace add
        return full


@register(LOSS)
class MacroCrossEntropy(SmoothCrossEntropy):

    def __init__(self, exp=None, smooth_epsilon=0.0, weight_by=None, *args, **kwargs):
        assert weight_by is None, 'weight_by is not supported for MacroCrossEntropy'
        super().__init__(exp=exp, *args, smooth_epsilon=smooth_epsilon, **kwargs)
        assert 0.0 <= smooth_epsilon < 1
        self.is_smoothing = smooth_epsilon > 0.
        log.info(f"Smoothing enabled? {self.is_smoothing} ; smooth_epsilon={smooth_epsilon}")
        assert self.weight is None

    def input_type(self) -> str:
        return 'logits'

    @staticmethod
    def one_hot(indices, n_classes):
        assert len(indices.shape) == 1
        N = indices.shape[0]
        C = n_classes
        # expand target [N] -> [N, C]  i.e., one hot vector
        return torch.zeros(N, C, dtype=torch.float, device=indices.device).scatter(1, indices.view(N, 1), 1.0)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        N, C = input.shape
        log_probs = input.log_softmax(dim=1)
        if self.is_smoothing:
            target = self.smooth_labels(target, C=C, epsilon=self.smooth_epsilon,
                                        weight=self.smooth_weight)
        else:
            # expand target [N] -> [N, C]  i.e., one hot vector
            target = self.one_hot(indices=target, n_classes=C).to(input.device)

        assert input.shape == target.shape

        # [N, C] :  y_c * log(p_c)
        losses = torch.mul(target, log_probs)
        # sum along dim 0 :  top-down each column
        col_sum = losses.sum(dim=0)   # [N, C] -> [C]
        col_norm = target.sum(dim=0)  # [N, C] -> [C]
        assert len(col_sum.shape) == 1 and col_sum.shape[0] == C
        assert col_sum.shape == col_norm.shape

        epsilon = 1e-9  # to avoid divide by zero
        losses = col_sum / (col_norm + epsilon)
        return -losses.sum()   # sum along the column


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

