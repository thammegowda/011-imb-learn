#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 6/14/21

import torch
import random
from .loss import MacroCrossEntropy
import tqdm


def test_macro_CE_one_hot():
    mce = MacroCrossEntropy(smooth_epsilon=0.0)
    assert mce.input_type() == 'logits'
    for _ in tqdm.trange(10):  # N tests
        C = random.randint(2, 64)
        N = random.randint(2, 256)
        target = torch.randint(low=0, high=C - 1, size=(N,))
        logits = torch.rand(N, C)
        loss = mce(logits, target)
        assert loss > 0
        assert not torch.isnan(loss)

def test_macro_CE_label_smooth():

    for _ in tqdm.trange(10):  # N tests
        epsilon = random.randint(1, 50) / 100
        mce = MacroCrossEntropy(smooth_epsilon=epsilon)
        assert mce.input_type() == 'logits'
        C = random.randint(2, 64)
        N = random.randint(2, 256)
        target = torch.randint(low=0, high=C - 1, size=(N,))
        logits = torch.rand(N, C)
        loss = mce(logits, target)
        assert loss > 0
        assert not torch.isnan(loss)