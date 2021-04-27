#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 4/23/21
import logging as log
from pathlib import Path
from typing import Optional, Dict
from copy import copy

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from . import log, yaml, device, ClsMetric
from .model import ImageClassifier
from .scheduler import InverseSqrtSchedule, NoamSchedule, LRSchedule


class Registry:
    models = dict(ImageClassifier=ImageClassifier)
    optimizers = dict(Adam=optim.Adam, SGB=optim.SGD, Adagrad=optim.Adagrad, AdamW=optim.AdamW,
                      Adadelta=optim.Adadelta, SparseAdam=optim.SparseAdam)
    schedulers = dict(inverse_sqrt=InverseSqrtSchedule, noam=NoamSchedule)


class Experiment:

    def __init__(self, work_dir: Path, model=None, optimizer=None, device=device):

        self.work_dir = work_dir
        self.conf = yaml.load(work_dir / 'conf.yml')
        self.sanity_check_conf(self.conf)
        self.device = device
        self.models_dir = work_dir / 'models'
        self.models_dir.mkdir(parents=True, exist_ok=True)
        (self.work_dir / 'logs').mkdir(parents=True, exist_ok=True)
        log.update_file_handler(work_dir / 'logs/trainer.log')

        # these magic numbers are required for imagenet pretrained models
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.train_transform = T.Compose([T.RandomResizedCrop(224),
                                          T.RandomHorizontalFlip(),
                                          T.ToTensor(), self.normalize])
        self.eval_transform = T.Compose([T.Resize(256), T.CenterCrop(224),
                                         T.ToTensor(), self.normalize])

        self.train_data = ImageFolder(self.conf['train']['data'], transform=self.train_transform)
        self.val_data = ImageFolder(self.conf['validation']['data'], transform=self.eval_transform)
        self.classes = self.train_data.classes
        assert self.classes == self.val_data.classes, 'Training and val classes mismatch'

        self.n_classes = self.conf['model']['args'].get('n_classes')
        assert len(self.classes) == self.n_classes, \
            f'Dataset has {len(self.classes)}, but in conf has model.n_classes={self.n_classes}'

        # todo: extract training and validation frequencies per each class
        class_stats = self.work_dir / 'classes.csv'
        if not class_stats.exists():
            txt = "\n".join(f"{i},{name}" for i, name in enumerate(self.classes))
            class_stats.write_text(txt + '\n')

        self.criterion = nn.CrossEntropyLoss()
        self.model = (model or self._get_model()).to(device)
        self.optimizer = optimizer or self._get_optimizer()
        self.scheduler = self._get_lr_scheduler()

        self._state = dict(step=0, epoch=0, last_checkpt=None,
                           train_metric=dict(loss=[], accuracy=[]),
                           val_metric=dict(loss=[], accuracy=[]))
        self._state_file = self.work_dir / 'state.yml'
        if self._state_file.exists() and self._state_file.stat().st_size > 0:
            self._state = yaml.load(self._state_file)
            log.info(f"state={self._state}")

        self.step = self._state['step']
        self.epoch = self._state['epoch']
        if self.step > 0:
            assert self.best_checkpt.exists(), f'{self._state_file} exists with step > 0 but' \
                                               f' no checkpoint found at {self.best_checkpt}'
            chkpt = torch.load(self.best_checkpt, map_location=device)
            log.info(f'Restoring model state from checkpoint')
            self.model.load_state_dict(chkpt['model_state'])
            if 'optimizer_state' in chkpt:
                log.info('Restoring optimizer state from checkpoint')
                self.optimizer.load_state_dict(chkpt['optimizer_state'])

    def _get_model(self, name=None, **args) -> nn.Module:
        name = name or self.conf['model']['name']
        args = args or self.conf['model']['args']
        assert name in Registry.models, f'model={name} is unknown; known={Registry.models.keys()}'
        log.info(f"Model {name}; args: {args}")

        return Registry.models[name](**args)

    def _get_optimizer(self, name=None, **args) -> torch.optim.Optimizer:
        name = name or self.conf['optimizer']['name']
        args = args or self.conf['optimizer']['args']
        log.info(f"Optimizer {name}; args: {args}")
        assert name in Registry.optimizers, f'{name} unknown; known={list(Registry.optimizers.keys())}'
        log.info(f"Initializing {Registry.optimizers[name]} with args: {args}")
        return Registry.optimizers[name](params=self.model.parameters(), **args)

    def _get_lr_scheduler(self, name=None, **args) -> LRSchedule:
        name = name or self.conf['scheduler']['name']
        args = args or self.conf['scheduler']['args']

        assert name in Registry.schedulers, f'scheduler {name} unknown;' \
                                            f' known: {Registry.schedulers.keys()}'
        scheduler = Registry.schedulers[name](**args)
        log.info(f"Scheduler: {scheduler}")
        return scheduler

    @property
    def best_checkpt(self) -> Optional[Path]:
        checkpt = self.models_dir / 'checkpt_best.pkl'
        # assert checkpt.exists(), f'Invalid state: {checkpt} expected but not found'
        return checkpt

    @classmethod
    def sanity_check_conf(cls, conf):
        def _safe_get(key):
            root = conf
            for p in key.split('.'):
                assert root.get(p), f'{key} is required but not found in conf'
                root = root[p]
            return root

        dirs = ['train.data', 'validation.data']
        pwd = Path(".").absolute()
        for d in dirs:
            path = _safe_get(d)
            assert Path(path).exists(), f'{d}={path} is invalid or doesnt exist; PWD={pwd}'

        parent = _safe_get('model.args.parent')
        assert hasattr(torchvision.models, parent), f'parent model {parent} unknown to torchvision'
        assert callable(getattr(torchvision.models, parent)), f'parent model {parent} is invalid'

    def _checkpt_name(self, step, train_loss, val_loss):
        return f'model_{step:06d}_{train_loss:.5f}_{val_loss:.5f}.pkl'

    def _checkpoint(self, train_metrics, val_loader) -> bool:
        with torch.no_grad():
            self.model.eval()
            val_loss, val_met = self.validate(val_loader)
            val_metric_msg = val_met.format(confusion=False)
            log.info(f"Validation at step {self.step}:\n{val_metric_msg}")
            metric_file = self.models_dir / f'validation-{self.step:04d}.txt'
            metric_file.write_text(val_metric_msg)
            val_metrics = dict(loss=val_loss, accuracy=val_met.accuracy,
                               macro_f1=val_met.macro_f1,
                               macro_precision=val_met.macro_precision,
                               macro_recall=val_met.macro_recall)
            self.model.train()

        # YAML has trouble serializing numpy float64 types
        for key, metrics in [('train_metric', train_metrics), ('val_metric', val_metrics)]:
            for name, val in metrics.items():
                if not name in self._state[key]:
                    self._state[key][name] = []
                self._state[key][name].append(float(val))

        by = self.conf['validation'].get('by', 'loss').lower()
        assert by in val_metrics, f'validation.by={by} unknown; known={list(val_metrics.keys())}'
        log.info(f"Validation by {by}")
        patience = self.conf['validation'].get('patience', -1)
        minimizing = by in ('loss',)  # else maximising
        this_score = val_metrics[by]
        all_scores = self._state['val_metric'][by]

        if minimizing:
            is_best = this_score <= min(all_scores)
        else:
            is_best = this_score >= max(all_scores)

        log.info(f"Epoch={self.epoch} Step={self.step}"
                 f"\ntrain:{train_metrics}\nvalidation:{val_metrics}")
        if is_best:
            checkpt_name = 'checkpt_best.pkl'
            checkpt_path = self.models_dir / checkpt_name

            checkpt = dict(
                model_state=self.model.state_dict(),
                optimizer_state=self.optimizer.state_dict(),
                step=self.step,
                epoch=self.epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics)
            log.info('saving this checkpoint')
            torch.save(checkpt, checkpt_path)
            self._state.update(dict(step=self.step, epoch=self.epoch, recent_skips=0))
        else:
            log.warning('This checkpoint was not an improvement; Not saving it')
            self._state['recent_skips'] += 1
            log.info(f'Patience={patience};  recent skips={self._state["recent_skips"]}')
        yaml.dump(self._state, self._state_file)
        return self._state['recent_skips'] > patience  # stop training

    def validate(self, val_loader):
        losses = []
        accs = []
        assert not self.model.training
        predictions = []
        truth = []
        for xs, ys in tqdm(val_loader, desc=f'Ep:{self.epoch} Step:{self.step}'):
            xs, ys = xs.to(self.device), ys.to(self.device)
            output = self.model(xs)
            loss = self.criterion(output, ys)
            losses.append(loss.item())
            accs.append(accuracy(output.data, ys).item())
            _, top_idx = output.detach().max(dim=1)
            predictions.append(top_idx)
            truth.append(ys)
        metric = ClsMetric(prediction=torch.cat(predictions), truth=torch.cat(truth),
                           clsmap=self.classes)
        return np.mean(losses), metric

    def learning_rate_adjust(self) -> float:
        rate = self.scheduler(self.step)
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        return rate

    def train(self, max_step=10 ** 6, max_epoch=10 ** 3, batch_size=1,
              num_threads=0, checkpoint=1000):

        train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True,
                                  num_workers=num_threads, pin_memory=True)
        val_batch_size = self.conf.get('validation', {}).get('batch_size', batch_size)
        val_loader = DataLoader(self.val_data, batch_size=val_batch_size, shuffle=False,
                                num_workers=num_threads, pin_memory=True)

        if self.step > 0:
            log.info(f'resuming from step {self.step}; max_steps:{max_step}')

        force_stop = False  # early stop when val metric goes up
        while not force_stop and self.step <= max_step and self.epoch <= max_epoch:
            train_losses = []
            train_accs = []
            log.info(f"Epoch={self.epoch}")
            with tqdm(train_loader, initial=self.step, total=max_step, unit='batch',
                      dynamic_ncols=True, desc=f'Ep:{self.epoch}') as pbar:
                for xs, ys in pbar:
                    xs, ys = xs.to(self.device), ys.to(self.device)
                    output = self.model(xs)
                    loss = self.criterion(output, ys)

                    train_losses.append(loss.item())
                    train_accs.append(accuracy(output.data, ys).item())

                    # compute gradient and do SGD step
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.step += 1
                    lr = self.learning_rate_adjust()
                    self.optimizer.step()

                    p_msg = f'Lr={lr:g} Loss={train_losses[-1]:g} Acc:{train_accs[-1]:.2f}%'
                    pbar.set_postfix_str(p_msg, refresh=False)

                    if self.step % checkpoint == 0:
                        metrics = dict(loss=np.mean(train_losses), accuracy=np.mean(train_accs))
                        force_stop = self._checkpoint(metrics, val_loader)
                        train_losses.clear()
                        train_accs.clear()
                        if force_stop:
                            log.info("Force early stopping the training")
                            break

                    if self.step > max_step:
                        log.info("Max steps reached;")
                        break

            if not force_stop and self.step < max_step:
                self.epoch += 1

    def run(self):
        train_args: Dict = copy(self.conf['train'])
        train_args.pop('data', None)
        self.train(**train_args)


def accuracy(output, target):
    """Computes accuracy"""
    batch_size = target.size(0)
    _, top_idx = output.max(dim=1)
    correct = top_idx.eq(target).float().sum()
    return 100.0 * correct / batch_size


def parse_args():
    import argparse
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('work_dir', type=Path, help='Path of experiment dir having conf.yml file')
    return p.parse_args()


def main(**args):
    args = args or vars(parse_args())
    exp = Experiment(args['work_dir'])
    exp.run()


if __name__ == '__main__':
    main()
