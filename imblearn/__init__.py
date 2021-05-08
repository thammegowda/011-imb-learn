#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 5/7/21
__version__ = '0.2'

from .common.metric import ClsMetric
from .common.log import Logger
from .registry import register, registry, MODEL, SCHEDULE, LOSS, OPTIMIZER
from .common.schedule import LRSchedule

from ruamel.yaml import YAML
import torch

log = Logger()
yaml = YAML()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cpu_device = torch.device('cpu')
log.info(f"Default device={device}")

## register all
from .registry import register_all as __load_modules
__load_modules()




