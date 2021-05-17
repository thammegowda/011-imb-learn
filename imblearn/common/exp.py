#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 5/7/21
from pathlib import Path
from typing import Optional
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from copy import copy


from imblearn import device, yaml, log, registry, MODEL, OPTIMIZER, SCHEDULE, LOSS
from .schedule import LRSchedule


class BaseExperiment:

    def __init__(self, work_dir: Path, device=device, log_file=None, model=None):
        self.work_dir = work_dir
        self._conf_file = work_dir / 'conf.yml'
        assert self._conf_file.exists()
        self.conf = yaml.load(self._conf_file)
        self.device = device
        self.models_dir = work_dir / 'models'
        self.models_dir.mkdir(parents=True, exist_ok=True)
        if not log_file:
            logs_dir = work_dir / 'logs'
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_file = logs_dir / 'log.log'
        log.update_file_handler(log_file)
        self.model = (model or self._get_model()).to(device)
        self.class_stats = self.work_dir / 'classes.csv'
        if self.class_stats.exists():
            self.classes = []
            for line_num, line in enumerate(self.class_stats.read_text().splitlines()):
                idx, name = line.strip().split(",")[:2]
                assert line_num == int(idx)
                self.classes.append(name)

        self._trained_flag = self.work_dir / '_TRAINED'  # fully trained

    @property
    def best_checkpt(self) -> Optional[Path]:
        return self.models_dir / 'checkpt_best.pkl'

    @property
    def last_checkpt(self) -> Optional[Path]:
        return self.models_dir / 'checkpt_last.pkl'

    def _get_component(self, kind, name=None, override=None, **args):
        """

        :param kind: component kind py::MODEL, py::OPTIMIZER etc
        :param name: name of component (optional; will be picked from conf)
        :param override: override these args over the args from conf (optional)
        :param args: args for component (optional; will be picked from conf)
        :return: instance of component
        """
        name = name or self.conf[kind]['name']
        args = args or self.conf[kind].get('args', {})
        if override:
            args = copy(args)
            args.update(override)
        assert name in registry[kind], f'kind={name} unknown; Known={list(registry[kind].keys())}'
        log.info(f"Initialising {kind}={name}   args={dict(args)}")
        return registry[kind][name](**args)

    def _get_model(self) -> torch.nn.Module:
        return self._get_component(MODEL)

    def _get_optimizer(self, params=None) -> torch.optim.Optimizer:
        params = params or self.model.parameters()
        return self._get_component(OPTIMIZER, override=dict(params=params))

    def _get_schedule(self) -> LRSchedule:
        return self._get_component(SCHEDULE)

    def _get_loss_func(self):
        return self._get_component(LOSS, override=dict(exp=self))



class BaseTrainer(BaseExperiment):
    def __init__(self, work_dir, *args, **kwargs):
        logs_dir = work_dir / 'logs'
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / 'trainer.log'
        super().__init__(work_dir=work_dir, device=device, log_file=log_file)

        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_schedule()
        self._loss_func = None  # requires lazy initialization

        self._state = dict(step=0, epoch=0, best_step=0,
                           train_metric=dict(loss=[]),
                           val_metric=dict(loss=[]))
        self._state_file = self.work_dir / 'state.yml'
        if self._state_file.exists() and self._state_file.stat().st_size > 0:
            self._state = yaml.load(self._state_file)
            log.info(f"Restored state info from state={self._state_file}")

        self.step = self._state['step']
        self.epoch = self._state['epoch']
        if self.step > 0:
            checkpt_file = self.last_checkpt if self.last_checkpt.exists() else self.best_checkpt
            assert checkpt_file.exists(), \
                f'{self._state_file} exists with step > 0 but no checkpoint found at' \
                f' {self.last_checkpt} and {self.best_checkpt}'

            chkpt = torch.load(checkpt_file, map_location=device)
            self.step, self.epoch = chkpt['step'], chkpt['epoch']
            log.info(f'Restoring model state from checkpoint')
            self.model.load_state_dict(chkpt['model_state'])
            if 'optimizer_state' in chkpt:
                log.info('Restoring optimizer state from checkpoint')
                self.optimizer.load_state_dict(chkpt['optimizer_state'])

        tbd_dir = self.work_dir / 'tensorboard'
        tbd_dir.mkdir(exist_ok=True, parents=True)
        self.tbd = SummaryWriter(log_dir=str(tbd_dir))

    @property
    def loss_function(self):
        if not self._loss_func:
            self._loss_func = self._get_loss_func()
        return self._loss_func