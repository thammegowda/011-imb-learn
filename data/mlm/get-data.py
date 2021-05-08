#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 5/7/21

""""""
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import logging as log
from html import unescape
import numpy as np

from sacremoses import MosesTokenizer

log.basicConfig(level=log.INFO)
cache_dir = Path(__file__).absolute().parent / 'cache'
tokr = MosesTokenizer()

protected_patterns = [
        r'((https?|ftp|rsync)://|www\.)[^ ]*',   # URLs
        r'[\w\-\_\.]+\@([\w\-\_]+\.)+[a-zA-Z]{2,}', # Emails user@host.domain
#        r'[@#][a-zA-Z0-9_]+', # @handler such as twitter/github ID #hashtag
    ]


def flatten(dataset):
    for rec in tqdm(dataset):
        if rec.get('title'):
            yield from rec['title'].splitlines()
        if rec.get('text'):
            yield from rec['text'].splitlines()

def tokenize(text):
    text = unescape(text)
    return tokr.tokenize(text, aggressive_dash_splits=True, return_str=True, escape=False,
                  protected_patterns=protected_patterns)

def dedupe(texts):
    mem = set()
    dupes = 0
    for text in texts:
        # fast hash --> 8byte int via numpy ; python's int is 28byte
        n = np.int64(hash(text))
        if n in mem:
            dupes += 1
            continue
        # not a dupe
        mem.add(n)
        yield text

if __name__ == '__main__':
    names = ['cc_news']
    for name in names:
        split = 'train'
        file = Path(f'{name}-{split}.txt')
        if not file.exists():
            log.info(f"Downloading {name} {split} -> {file}")
            dataset = load_dataset(name, split=split, cache_dir=cache_dir)
            with file.open('w', encoding='utf-8', errors='ignore') as out:
                for line in flatten(dataset):
                    line = line.strip()
                    out.write(line)
                    out.write('\n')
        tok_file = file.with_suffix(".dedup.tok")
        if not tok_file.exists():
            log.info(f"Tokenizing {file} -> {tok_file}")
            with file.open() as lines, tok_file.open('w') as out:
                for line in tqdm(dedupe(lines)):
                    out.write(tokenize(line))
                    out.write('\n')
        log.info("Done")