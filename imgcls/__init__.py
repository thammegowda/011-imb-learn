#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 4/23/21
__version__ = '0.1'

import logging
from .log import Logger
from ruamel.yaml import YAML
import torch
from .metric import ClsMetric


log = Logger(console_level=logging.DEBUG)
yaml = YAML()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cpu_device = torch.device('cpu')
log.info(f"Default device={device}")