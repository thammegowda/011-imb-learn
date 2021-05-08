#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 4/23/21
import logging as log
from pathlib import Path
from typing import Dict, List
from copy import copy
from datetime import datetime
import numpy as np
import torch

from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from imblearn import log, yaml, device, ClsMetric
from imblearn.common.exp import BaseTrainer


normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
eval_transform = T.Compose([T.Resize(256), T.CenterCrop(224),
                                         T.ToTensor(), normalize])

class Trainer(BaseTrainer):

    def __init__(self, work_dir, *args, **kwargs):
        self.work_dir = work_dir

        super().__init__(*args, work_dir=work_dir, **kwargs)
        self.sanity_check_conf(self.conf)

        # these magic numbers are required for imagenet pretrained models
        self.train_transform = T.Compose([T.RandomResizedCrop(224),
                                          T.RandomHorizontalFlip(),
                                          T.ToTensor(), normalize])

        self.train_data = ImageFolder(self.conf['train']['data'], transform=self.train_transform)
        self.val_data = ImageFolder(self.conf['validation']['data'], transform=eval_transform)
        self.classes = self.train_data.classes
        assert self.classes == self.val_data.classes, 'Training and val classes mismatch'

        self.n_classes = self.conf['model']['args'].get('n_classes')
        assert len(self.classes) == self.n_classes, \
            f'Dataset has {len(self.classes)}, but in conf has model.n_classes={self.n_classes}'
        self.cls_freqs = self.get_train_freqs()

    def get_train_freqs(self):
        class_stats = self.work_dir / 'classes.csv'
        if not class_stats.exists():
            cls_freqs = self.get_class_frequencies(Path(self.conf['train']['data']))
            lines = []
            for idx, (cls_name, freq) in enumerate(zip(self.classes, cls_freqs)):
                lines.append(f"{idx},{cls_name},{freq}")
            class_stats.write_text("\n".join(lines))
        # third column has frequency
        return [int(line.strip().split(',')[2])
                          for line in class_stats.read_text().splitlines()]

    def get_class_frequencies(self, img_folder: Path) -> List[int]:
        freqs = [-1] * self.n_classes
        for idx, cls_name in enumerate(self.classes):
            cls_dir = img_folder / cls_name
            assert cls_dir.exists()
            freqs[idx] = sum(1 for _ in cls_dir.glob('*.*'))
        return freqs

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

        if conf.get('tests'):
            for name, path in conf['tests'].items():
                assert Path(path).exists(), f'test dir {name}={path} does not exist. PWD={pwd}'

    def _checkpt_name(self, step, train_loss, val_loss):
        return f'model_{step:06d}_{train_loss:.5f}_{val_loss:.5f}.pkl'

    def _checkpoint(self, train_metrics, val_loader) -> bool:
        with torch.no_grad():
            self.model.eval()
            val_loss, val_met = self.validate(val_loader)
            val_metric_msg = val_met.format(confusion=False)
            log.info(f"Validation at step {self.step}:\n{val_metric_msg}")
            metric_file = self.models_dir / f'validation-{self.step:04d}.txt'
            val_metric_msg = val_met.format(confusion=True)
            metric_file.write_text(val_metric_msg)
            val_metrics = dict(loss=val_loss, accuracy=val_met.accuracy,
                               macro_f1=val_met.macro_f1,
                               macro_precision=val_met.macro_precision,
                               macro_recall=val_met.macro_recall)
            self.model.train()
        self.store_metrics(train_metrics, val_metrics)
        if self.n_classes <= 40:  # Too many classes will be a mess
            self.tbd_val_cls_metrics(val_met)

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
                 f"\ntrain loss:{train_metrics['loss']:g} validation loss:{val_metrics['loss']:g}")
        chkpt_state = dict(
            model_state=self.model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            step=self.step,
            epoch=self.epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics)

        torch.save(chkpt_state, self.last_checkpt)
        self._state.update(dict(step=self.step, epoch=self.epoch))

        if is_best:
            log.info('saving this checkpoint as best')
            torch.save(chkpt_state, self.best_checkpt)
            self._state.update(dict(best_step=self.step, recent_skips=0))
        else:
            log.warning('This checkpoint was not an improvement')
            self._state['recent_skips'] = self._state.get('recent_skips', 0) + 1
            log.info(f'Patience={patience};  recent skips={self._state["recent_skips"]}')
        yaml.dump(self._state, self._state_file)
        return self._state['recent_skips'] > patience  # stop training

    def tbd_val_cls_metrics(self, val_metric: ClsMetric):
        f1s = dict(zip(val_metric.clsmap, val_metric.f1))
        precisions = dict(zip(val_metric.clsmap, val_metric.precision))
        recalls = dict(zip(val_metric.clsmap, val_metric.recall))
        self.tbd.add_scalars('ValF1', f1s, self.step)
        self.tbd.add_scalars('ValPrecision', precisions, self.step)
        self.tbd.add_scalars('ValRecall', recalls, self.step)

    def store_metrics(self, train_metrics, val_metrics):
        head = ['Time', 'Epoch', 'Step', 'TrainLoss', 'TrainAccuracy', 'ValLoss', 'ValAccuracy',
                'ValMacroF1']
        row = [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%3d' % self.epoch, '%5d' % self.step,
               '%.6f' % train_metrics['loss'], '%.2f' % train_metrics['accuracy'],
               '%.6f' % val_metrics['loss'], '%.2f' % val_metrics['accuracy'],
               '%.2f' % val_metrics['macro_f1']]
        scores_file = self.work_dir / 'scores.tsv'
        new_file = not scores_file.exists()
        with scores_file.open('a') as f:
            if new_file:
                f.write('\t'.join(head) + '\n')
            f.write('\t'.join(row) + '\n')
        # YAML has trouble serializing numpy float64 types
        for key, metrics in [('train_metric', train_metrics), ('val_metric', val_metrics)]:
            for name, val in metrics.items():
                if not name in self._state[key]:
                    self._state[key][name] = []
                self._state[key][name].append(float(val))

        self.tbd.add_scalars('Loss', dict(train=train_metrics['loss'], val=val_metrics['loss']),
                             global_step=self.step)
        self.tbd.add_scalars('Performance', dict(train_acc=train_metrics['accuracy'],
                                                 val_acc=val_metrics['accuracy'],
                                                 val_macrof1=val_metrics['macro_f1']),
                             global_step=self.step)

    def validate(self, val_loader):
        losses = []
        accs = []
        assert not self.model.training
        predictions = []
        truth = []
        for xs, ys in tqdm(val_loader, desc=f'Ep:{self.epoch} Step:{self.step}'):
            xs, ys = xs.to(self.device), ys.to(self.device)
            output = self.model(xs)
            loss = self.loss_function(output, ys)
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
        if self.step >= max_step or self._trained_flag.exists():
            log.warning("Skipping training.")
            if self.step >= max_step:
                log.warning(f'Already trained till {self.step}. Tip: Increase max_steps')
            elif self._trained_flag.exists():
                log.warning(f"Training was previously converged. Tip: rm {self._trained_flag}")
            return
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
                    loss = self.loss_function(output, ys)

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
                            self._trained_flag.touch()
                            break

                    if self.step > max_step:
                        log.info("Max steps reached;")
                        break

            if not force_stop and self.step < max_step:
                self.epoch += 1


def accuracy(output, target):
    """Computes accuracy"""
    batch_size = target.size(0)
    _, top_idx = output.max(dim=1)
    correct = top_idx.eq(target).float().sum()
    return 100.0 * correct / batch_size


def parse_args():
    import argparse
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('exp_dir', type=Path, help='Experiment dir path (must have conf.yml in it).')
    return p.parse_args()


def main(**args):
    args = args or vars(parse_args())
    exp_dir: Path = args['exp_dir']
    trainer = Trainer(exp_dir)
    train_args: Dict = copy(trainer.conf['train'])
    train_args.pop('data', None)
    trainer.train(**train_args)

    if trainer.conf.get('tests'):
        log.info('Running tests')
        from .eval import main as eval_main
        for name, test_dir in trainer.conf['tests'].items():
            step_num = trainer._state.get("best_step", trainer._state["step"])
            out_file = exp_dir / f'result.step{step_num}.{name}.txt'
            if out_file.exists() and out_file.stat().st_size > 0:
                log.info(f"{out_file} exists; skipping {test_dir}")
                continue
            with out_file.open('w') as out:
                eval_main(exp_dir=exp_dir, test_dir=test_dir, out=out)


if __name__ == '__main__':
    main()
