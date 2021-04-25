#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 4/23/21

# %%

import logging as log
from pathlib import Path
from typing import Optional, Dict
from copy import copy

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from . import log, yaml, device
from .model import ImageClassifier


class Experiment:

    def __init__(self, work_dir: Path, model=None, optimizer=None, device=device):


        self.work_dir = work_dir
        self.models_dir = work_dir / 'models'
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.conf = yaml.load(work_dir / 'conf.yml')
        self.sanity_check_conf(self.conf)
        log.update_file_handler(work_dir / 'trainer.log')

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

        self.n_classes = self.conf.get('model', {}).get('n_classes')
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
            assert self.last_checkpt and self.last_checkpt.exists()
            chkpt = torch.load(self.last_checkpt, map_location=device)
            self.model.load_state_dict(chkpt['model_state'])
            if 'optimizer_state' in chkpt:
                self.optimizer.load_state_dict(chkpt['optimizer_state'])

    def _get_model(self, **args) -> nn.Module:
        args = args or self.conf['model']
        return ImageClassifier(**args)

    def _get_optimizer(self, args=None) -> torch.optim.Optimizer:
        args = args or self.conf['optimizer']
        name, o_args = args['name'].lower(), dict(args['args'])
        from torch import optim
        lookup = dict(adam=optim.Adam, sgd=optim.SGD, adagrad=optim.Adagrad,
                      adadelta=optim.Adadelta, sparseadam=optim.SparseAdam)
        assert name in lookup, f'{name} unknown; known={list(lookup.keys())}'
        log.info(f"Initializing {lookup[name]} with args: {o_args}")
        return lookup[name](params=self.model.parameters(), **o_args)

    @property
    def last_checkpt(self) -> Optional[Path]:
        checkpt = self._state.get('last_checkpt')
        if checkpt:
            checkpt = self.models_dir / checkpt
            assert checkpt.exists(), f'Invalid state: {checkpt} expected but not found'
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

        parent = _safe_get('model.parent')
        assert hasattr(torchvision.models, parent), f'parent model {parent} unknown to torchvision'
        assert callable(getattr(torchvision.models, parent)), f'parent model {parent} is invalid'

    def _checkpt_name(self, step, train_loss, val_loss):
        return f'model_{step:06d}_{train_loss:.5f}_{val_loss:.5f}.pkl'

    def _checkpoint(self, train_metrics, val_loader) -> bool:
        with torch.no_grad():
            self.model.eval()
            val_loss, val_acc = self.validate(val_loader)
            val_metrics = dict(loss=val_loss, accuracy=val_acc)
            self.model.train()


        # YAML has trouble serializing numpy float64 types
        for key, metrics in [('train_metrics', train_metrics), ('val_metrics', val_metrics)]:
            for name, val in metrics:
                if not name in self._state[key]:
                    self._state[key][name] = []
                self._state[key][name].append(float(val))

        by = self.conf['validation'].get('by', 'loss')
        patience = self.conf['validation'].get('patience', -1)
        minimizing = by in ('loss',)   # else maximising
        this_score = val_metrics[by]
        all_scores = self._state['val_metrics'][by]

        if minimizing:
            is_best = this_score <= min(all_scores)
        else:
            is_best = this_score >= max(all_scores)

        if is_best:
            checkpt_name = 'checkpt_best.pkl'
            checkpt_path = self.models_dir / checkpt_name

            checkpt = dict(
                model_state=self.model.state_dict(),
                optimizer_state=self.optimizer.state_dict(),
                step=self.step,
                epoch=self.epoch,
                train_metrics=train_metrics,
                val_metrics=dict(loss=val_loss, accuracy=val_acc))
            log.info(f"Checkpointing: step={self.step} epoch={self.epoch}"
                     f"\ntrain:{train_metrics}\nvalidation:{val_metrics}")
            torch.save(checkpt, checkpt_path)
            self._state.update(dict(step=self.step, epoch=self.epoch, recent_skips=0))
        else:
            self._state['recent_skips'] += 1
        yaml.dump(self._state, self._state_file)
        return self._state['recent_skips'] > patience # stop training

    def validate(self, val_loader):
        losses = []
        accs = []
        assert not self.model.training
        for xs, ys in tqdm(val_loader):
            output = self.model(xs)
            loss = self.criterion(output, ys)
            losses.append(loss.item())
            accs.append(accuracy(output.data, ys))
        return np.mean(losses), np.mean(accs)

    def train(self, max_step=10 ** 6, max_epoch=10 ** 3, batch_size=1,
              num_threads=0, checkpoint=1000):

        train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True,
                                  num_workers=num_threads, pin_memory=True)
        val_loader = DataLoader(self.val_data, batch_size=batch_size, shuffle=True,
                                num_workers=num_threads, pin_memory=True)

        if self.step > 0:
            log.info(f'resuming from step {self.step}; max_steps:{max_step}')

        force_stop = False  # early stop when val metric goes up
        while not force_stop and self.step <= max_step and self.epoch <= max_epoch:
            train_losses = []
            train_accs = []
            for xs, ys in tqdm(train_loader):

                output = self.model(xs)
                loss = self.criterion(output, ys)

                train_losses.append(loss.item())
                train_accs.append(accuracy(output.data, ys))

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.step += 1
                #
                if self.step % checkpoint == 0:
                    metrics = dict(loss=np.mean(train_losses), accuracy=np.mean(train_accs))
                    force_stop = self._checkpoint(metrics, val_loader)
                    train_losses.clear()
                    train_accs.clear()

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
