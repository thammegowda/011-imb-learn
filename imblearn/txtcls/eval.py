#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 4/27/21
import collections
import sys
from pathlib import Path
from typing import List, Union

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from imblearn import device, log, ClsMetric
from imblearn.common.util import read_lines, collate_batch
from imblearn.txtcls.train import NLPExperiment

DEF_MAX_LEN = 1024
DEF_BATCH_SIZE = 10

class Sequences(Dataset):
    def __init__(self, data:Union[List[str], str, Path], prep_fn=None, max_len=DEF_MAX_LEN):
        if isinstance(data, (str, Path)):
            self.seqs = list(read_lines(data))
        elif isinstance(data, list):
            self.seqs = data
        elif isinstance(data, collections.Iterable):
            self.seqs = [x.strip() for x in data]
        self.prep_fn = prep_fn
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, item):
        seq = self.seqs[item].strip()
        if not seq:
            log.warning(f"empty sequence at index {item}; inserted a dot (.) to make it not empty")
            seq = '.'
        if self.prep_fn:
            seq = self.prep_fn(seq)
        if self.max_len:
            seq = seq[:self.max_len]
        return seq, item

class Evaluator:

    def __init__(self, work_dir: Path):
        self.exp = exp = NLPExperiment(work_dir=work_dir, device=device)
        self.device = device
        assert exp.best_checkpt.exists()
        assert exp.classes
        self.model = exp._get_model().to(device).eval()
        chkpt = torch.load(exp.best_checkpt, map_location=device)
        log.info(f'Restoring model state from checkpoint {exp.best_checkpt}; step={chkpt["step"]}')
        self.model.load_state_dict(chkpt['model_state'])
        self.checkpt_step = chkpt["step"]

    def predict(self, data: Union[List[str], str, Path], num_threads=0,
                        batch_size=DEF_BATCH_SIZE, max_len=DEF_MAX_LEN):
        seqs = Sequences(data=data, prep_fn=self.exp.src_vocab.encode, max_len=max_len)
        loader = DataLoader(seqs, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_batch, num_workers=num_threads, pin_memory=True)
        for batch, idxs in tqdm(loader):
            batch = batch.to(self.device)
            probs = self.model(batch, src_mask=self.exp.get_src_mask(xs=batch), score='softmax')
            #probs = torch.softmax(output, dim=1)
            top_prob, top_idx = probs.detach().max(dim=1)
            for i in range(len(batch)):
                _name, _prob  = self.exp.classes[top_idx[i]], top_prob[i]
                yield (seqs[idxs[i]], _name, _prob)

    def evaluate(self, pred_labels: List[str], labels_file:(str, Path)) -> (ClsMetric):
        gold_labels_name = list(read_lines(labels_file))
        gold_labels_idx = [self.exp.tgt_vocab.str_to_idx[name] for name in gold_labels_name]
        pred_labels_idx = [self.exp.tgt_vocab.str_to_idx[name] for name in pred_labels]
        return ClsMetric(prediction=pred_labels_idx, truth=gold_labels_idx, clsmap=self.exp.classes)

def main(**args):
    args = args or vars(parse_args())
    evaluator = Evaluator(args['exp_dir'])
    conf = evaluator.exp.conf
    val_batch_size = conf['validation'].get('batch_size', conf['train']['batch_size'])
    batch_size = args.get('batch_size', val_batch_size)

    log.info(f"Reading sequences {args['inp']}")
    src_file = args['inp']
    pred_file: Path = args['out']
    labels_file: Path = args.get('labels')
    result_file: Path = args.get('result')
    max_len = args.get('max_len', DEF_MAX_LEN)

    preds = evaluator.predict(src_file, batch_size=batch_size, max_len=max_len)
    __seqs, top1_names, top1_probs = zip(*preds)
    pred_file.write_text('\n'.join(f'{n}\t{p:g}' for n, p in zip(top1_names, top1_probs)))
    if labels_file:
        assert Path(labels_file).exists()
        metrics = evaluator.evaluate(pred_labels=top1_names, labels_file=labels_file)
        result_file.write_text(metrics.format(delim=','))

def parse_args():
    import argparse
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('exp_dir', type=Path, help='Experiment dir path (must have conf.yml in it).')
    inp_p = p.add_mutually_exclusive_group()
    inp_p.add_argument('-i', '--inp', type=Path, required=True,
                   help='Input: text file having one test image path per line.')
    p.add_argument('-o', '--out', type=Path, required=True,
                   help='Output file path to store predictions')

    inp_p.add_argument('-l', '--labels', type=Path, help='Ground truth labels for evaluating --inp seqs')
    inp_p.add_argument('-r', '--result', type=Path, default=sys.stdout,
                       help='File to write the evaluation results; valid when --labels is valid')

    p.add_argument('-b', '--batch-size', type=int, default=5, help='num sequences in a mini batch')
    p.add_argument('-m', '--max-len', type=int, default=DEF_MAX_LEN, help='max source length')
    return p.parse_args()


if __name__ == '__main__':
    main()
