#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 4/26/21
from dataclasses import dataclass

from imblearn import register, SCHEDULE


@dataclass()
class LRSchedule:

    def __call__(self, *args, **kwargs) -> float:
        return self.rate(*args, **kwargs)

    def rate(self, step) -> float:
        raise NotImplementedError()


@register(SCHEDULE, 'noam')
@dataclass()
class Noam(LRSchedule):
    warmup: int
    scaler: int = 1
    model_dim: int = 100

    def rate(self, step) -> float:
        return self.scaler * self.model_dim ** -0.5 * min(step ** -0.5, step * self.warmup ** -1.5)


@register(SCHEDULE, 'inverse_sqrt')
@dataclass()
class InverseSqrt(LRSchedule):
    warmup: int
    peak_lr: float

    def rate(self, step) -> float:
        return min(step * self.peak_lr / self.warmup,
                   self.peak_lr * self.warmup ** 0.5 * step ** -0.5)
