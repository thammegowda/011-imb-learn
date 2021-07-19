#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 7/18/21

import logging as log
from collections import Counter
from copy import copy
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from nlcodec import learn_vocab, load_scheme, Reseved
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from imblearn import log, yaml, registry, LOSS
from imblearn.common.exp import BaseTrainer, BaseExperiment
from imblearn.common.util import read_parallel_recs, collate_batch
from imblearn.common.metric import ClsMetric, accuracy


class InMemDataset(Dataset):

    def __init__(self, src_path, tgt_path, src_to_idx=None, tgt_to_idx=None,
                 max_src_len=None, max_tgt_len=None):
        """

        :param path:  dataset path having text and labels in each row
        :param delim: delimiter separating text and label (default \\t)
        :param src_to_idx: function to map text to sequence of integer (embedding index)
        :param tgt_to_idx: function to map label to integer
        :param max_src_len: truncate source text at this length (optional)
        :param max_tgt_len: truncate target text at this length (optional)
        """
        self.texts, self.labels = [], []
        recs = read_parallel_recs(src_path, tgt_path, src_prep=src_to_idx, tgt_prep=tgt_to_idx,
                                  max_src_len=max_src_len, max_tgt_len=max_tgt_len)
        for txt, lbl in recs:
            self.texts.append(txt)
            self.labels.append(txt)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        return self.texts[item], self.labels[item]


class NLPExperiment(BaseExperiment):

    def __init__(self, work_dir, *args, **kwargs):
        super().__init__(work_dir)
        self.data_dir = data_dir = self.work_dir / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)

        self.src_vocab_file = data_dir / 'nlcodec.src.tsv'
        self.tgt_vocab_file = data_dir / 'nlcodec.tgt.tsv'
        assert self.src_vocab_file.exists() == self.tgt_vocab_file.exists(), \
            f'Both or none; {self.src_vocab_file} {self.tgt_vocab_file}'
        if not self.src_vocab_file.exists():
            self._make_vocabs()
        self.src_vocab = load_scheme(self.src_vocab_file)
        self.tgt_vocab = load_scheme(self.tgt_vocab_file)

        self.pad_idx = Reseved.PAD_IDX

    def _make_vocabs(self):
        args = self.conf['prep']
        log.info(f"preprocessing args: {args}")

        src_path = self.conf['train']['src_path']
        tgt_path = self.conf['train']['tgt_path']
        data = read_parallel_recs(src_path=src_path, tgt_path=tgt_path,
                                  src_prep=None, tgt_prep=None)
        srcs, tgts = zip(*data)  # separate iterable of tuples into two collections
        # note: srcs and tgts are are  tuples
        assert 'shared' in args or ('src' in args and 'tgt' in args), \
            'requires either prep.shared or both prep.src and prep.tgt vocabulary settings'

        if args.get('shared'):
            log.info("creating shared vocabulary")
            learn_vocab(inp=srcs + tgts, model=self.src_vocab_file, **args['shared'])
            self.tgt_vocab_file.link_to(self.src_vocab_file)
        else:
            log.info("creating source vocabulary")
            learn_vocab(inp=srcs, model=self.src_vocab_file, **args['src'])
            log.info("creating target vocabulary")
            learn_vocab(inp=tgts, level='class', model=self.tgt_vocab_file, vocab_size=-1)

    def _get_inmem_data(self, name: str):
        cache = self.data_dir / f'{name}.pkl'
        if not cache.exists():
            src_path = self.conf['train']['src_path']
            tgt_path = self.conf['train']['tgt_path']
            data = read_parallel_recs(src_path=src_path, tgt_path=tgt_path,
                                      src_prep=self.src_vocab.encode,
                                      tgt_prep=self.tgt_vocab.encode)
            data = list(data)
            torch.save(data, cache)
        else:
            data = torch.load(cache)
        return data

    def get_train_data(self):
        return self._get_inmem_data(name='train')

    def get_val_data(self):
        return self._get_inmem_data(name='validation')

    def get_src_mask(self, xs):
        return (xs == self.pad_idx)


class Trainer(BaseTrainer, NLPExperiment):

    def __init__(self, work_dir, *args, **kwargs):
        self.work_dir = work_dir

        super().__init__(*args, work_dir=work_dir, **kwargs)
        self.sanity_check_conf(self.conf)

        # we don't want to use fancy loss function here
        # by fancy, i mean, functions with label smoothing, weighted etc ...
        self.val_loss_func = registry[LOSS]['cross_entropy'](exp=self)  # this is vanilla CE no args
        assert all(idx == typ.idx for idx, typ in enumerate(self.tgt_vocab.table))
        self.classes = [t.name for t in self.tgt_vocab.table]

        self.n_classes = self.conf['model']['args'].get('n_classes')
        assert len(self.classes) == self.n_classes, \
            f'Dataset has {len(self.classes)}, but conf.yml:model.args.n_classes' \
            f' has model.n_classes={self.n_classes}'
        self.cls_freqs = self.get_train_freqs()

    def get_train_freqs(self):
        class_stats = self.work_dir / 'classes.csv'
        if not class_stats.exists():
            data = self.get_train_data()
            cls_freqs = Counter(lbl[0] for txt, lbl in iter(data))
            lines = []
            for idx, name in enumerate(self.classes):
                freq = cls_freqs.get(idx)
                lines.append(f"{idx},{name},{freq}")
            class_stats.write_text("\n".join(lines))
        # third column has frequency
        return [int(line.strip().split(',')[2])
                for line in class_stats.read_text().splitlines()]

    @classmethod
    def sanity_check_conf(cls, conf):
        def _safe_get(key):
            root = conf
            for p in key.split('.'):
                assert root.get(p), f'{key} is required but not found in conf'
                root = root[p]
            return root

        dirs = ['train.src_path', 'train.tgt_path', 'validation.src_path', 'validation.tgt_path']
        pwd = Path(".").absolute()
        for d in dirs:
            path = _safe_get(d)
            assert Path(path).exists(), f'{d}={path} is invalid or doesnt exist; PWD={pwd}'

        if conf.get('tests'):
            for name, (src_path, tgt_path) in conf['tests'].items():
                assert Path(src_path).exists(), f'test {name} src {src_path} not valid. PWD={pwd}'
                assert Path(tgt_path).exists(), f'test {name} tgt {tgt_path} not valid. PWD={pwd}'

    @staticmethod
    def _checkpt_name(step, train_loss, val_loss):
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
            xs, ys = xs.to(self.device), ys.to(self.device).type(torch.long)
            xs_mask = self.get_src_mask(xs)  # Batch x Length
            if len(ys.shape) > 1:
                ys = ys.squeeze(1)  # get rid off second dimension; it has to be just a vector
            output = self.model(xs, xs_mask, score=self.val_loss_func.input_type)
            loss = self.val_loss_func(output, ys)
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
              num_threads=0, checkpoint=1000, min_step=0):
        """
        :param max_step: maximum steps to train
        :param max_epoch: maximum epochs to train
        :param batch_size: batch size
        :param num_threads: num threads for data loader
        :param checkpoint: checkpoint and validate after every these steps
        :param keep_in_mem: keep datasets in memory
        :param train_parent_after: start training parent model after these many steps
        :param min_step: minimum steps to train
        :return:
        """
        train_loader = DataLoader(self.get_train_data(), batch_size=batch_size, shuffle=True,
                                  num_workers=num_threads, pin_memory=True,
                                  collate_fn=collate_batch)
        val_batch_size = self.conf.get('validation', {}).get('batch_size', batch_size)
        val_loader = DataLoader(self.get_val_data(), batch_size=val_batch_size, shuffle=False,
                                num_workers=num_threads, pin_memory=True,
                                collate_fn=collate_batch)

        assert min_step >= 0
        assert min_step < max_step
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
        while self.step < min_step or \
                not force_stop and self.step <= max_step and self.epoch <= max_epoch:
            train_losses = []
            train_accs = []
            log.info(f"Epoch={self.epoch}")
            with tqdm(train_loader, initial=self.step, total=max_step, unit='batch',
                      dynamic_ncols=True, desc=f'Ep:{self.epoch}') as pbar:
                for xs, ys in pbar:
                    self.step += 1
                    if len(ys.shape) > 1:
                        ys = ys.squeeze(1)  # get rid off second dimension; ys has to be a vector
                    xs, ys = xs.to(self.device), ys.to(self.device).type(torch.long)
                    xs_mask = self.get_src_mask(xs)  # Batch x Length
                    output = self.model(xs, xs_mask, score=self.loss_function.input_type)

                    loss = self.loss_function(output, ys)
                    if torch.isnan(loss):
                        log.warning("Loss is NaN; batch skipped")
                        continue

                    train_losses.append(loss.item())
                    train_accs.append(accuracy(output.data, ys).item())
                    p_msg = f'Loss={train_losses[-1]:g} Acc:{train_accs[-1]:.2f}%'

                    # compute gradients
                    self.optimizer.zero_grad()
                    loss.backward()

                    if self.scheduler:
                        lr = self.learning_rate_adjust()
                        self.tbd.add_scalar('lr', lr, global_step=self.step - 1)
                        p_msg = f'Lr={lr:g} ' + p_msg
                    # take SGD step
                    self.optimizer.step()
                    pbar.set_postfix_str(p_msg, refresh=False)

                    if self.step % checkpoint == 0:
                        metrics = dict(loss=np.mean(train_losses), accuracy=np.mean(train_accs))
                        force_stop = self._checkpoint(metrics, val_loader)
                        train_losses.clear()
                        train_accs.clear()
                        if force_stop:
                            if self.step < min_step:
                                log.warning("Early stop has reached,"
                                            f" but min_step={min_step} is keeping me going...")
                            else:
                                log.info("Force early stopping the training")
                                self._trained_flag.touch()
                                break

                    if self.step > max_step:
                        log.info("Max steps reached;")
                        break

            if not force_stop and self.step < max_step:
                self.epoch += 1

    def pipeline(self):
        train_args: Dict = copy(self.conf['train'])
        # ignore args
        train_args.pop('src_path')
        train_args.pop('tgt_path')
        self.train(**train_args)

        if self.conf.get('tests'):
            from .eval import main as eval_main
            step_num = self._state.get("best_step", self._state["step"])
            test_dir = self.work_dir / f'tests_step{step_num}_best1'
            test_dir.mkdir(exist_ok=True, parents=True)
            for name, (src_path, lbl_path) in self.conf['tests'].items():
                out_file = test_dir / f'{name}.preds.tsv'
                score_file = test_dir / f'{name}.score.tsv'
                if out_file.exists() and out_file.stat().st_size > 0:
                    log.info(f"{out_file} exists; skipping {test_dir}")
                    continue
                eval_main(exp_dir=self.work_dir, inp=src_path, out=out_file, labels=lbl_path,
                          result=score_file)
