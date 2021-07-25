#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 5/7/21
__version__ = '0.2'
__description__ = 'Machine Learning Toolkit that focus on imbalanced learning'

from ruamel.yaml import YAML
from .common.log import Logger
yaml = YAML()
log = Logger()

# this one required yaml to be init properly
from .common.tune import TunableParam, find_tunable_params
from .common.metric import ClsMetric
from .registry import register, registry, MODEL, SCHEDULE, LOSS, OPTIMIZER
from .common.schedule import LRSchedule

import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cpu_device = torch.device('cpu')
log.info(f"Default device={device}")

## register all
from .registry import register_all as __load_modules
__load_modules()




