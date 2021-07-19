#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 7/19/21
from typing import List, Tuple
import torch
from itertools import zip_longest
try:
    from collections.abc import Iterable, Sequence  # noqa
except ImportError:
    from collections import Iterable, Sequence

from imblearn import log


def read_lines(path):
    with open(path, 'r', encoding='utf8', errors='ignore') as lines:
        for line in lines:
            yield line.strip()


def read_tsv(path, n_cols: int = -1, delim='\t', pick_cols: List[int] = None):
    """
    read TSV
    :param path:  path to TSV file
    :param n_cols:  (optional) validate that each rec has this many cols
    :param delim:  (optional) delimiter for splitting columns; default \\t
    :param pick_cols: (optional) pick columns with these indices
    :return: iterator of columns
    """
    with open(path, 'r', encoding='utf8', errors='ignore') as lines:
        for line_num, line in enumerate(lines, start=1):
            cols = line.split(delim)
            if n_cols > 0:
                assert len(cols) == n_cols, f'line {line_num} from {path}::' \
                                            f' expected {n_cols} but found {len(cols)} columns'
            if pick_cols:
                cols = [cols[idx] for idx in pick_cols]
            cols = tuple(x.strip() for x in cols)
            if any(not x for x in cols):
                log.info(f"skipping line {line_num} because one of columns is emtpy")
                continue
            yield cols


def read_parallel_recs(src_path, tgt_path, src_prep=None, tgt_prep=None,
                       max_src_len=None, max_tgt_len=None):
    args = dict(mode='r', encoding='utf8', errors='ignore')
    with open(src_path, **args) as srcs, open(tgt_path, **args) as tgts:
        for line_num, (src, tgt) in enumerate(zip_longest(srcs, tgts)):
            if src_prep:
                src = src_prep(src)
            if tgt_prep:
                tgt = tgt_prep(tgt)
            if max_src_len:
                src = src[:max_src_len]
            if max_tgt_len:
                tgt = tgt[:max_tgt_len]
            yield src, tgt

def pad_seqs(batch: (Tuple[Sequence], List[Sequence]), pad_val=0):
    max_len = max(len(seq) for seq in batch)
    padded = torch.full((len(batch), max_len), fill_value=pad_val, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, :len(seq)] = torch.tensor(seq)
    return padded

def collate_batch(batch: List[Tuple]):
    """ Collate function with both x and y are sequences"""
    cols = zip(*batch)
    res = []
    for col in cols:
        if isinstance(col[0], Sequence):
            collated = pad_seqs(col)
        else:
            # assume its a scalar, also assume its an integer
            collated = torch.tensor(col, dtype=torch.long)
        res.append(collated)
    if len(res) == 1:
        return res[0]
    return tuple(res)