#!/usr/bin/env python
#
# this is a file for coarse discourse preparation: flattening, splitting to train/dev/test
# Author: Thamme Gowda [tg (at) isi (dot) edu]
# Created: 7/13/21


import logging as log
from pathlib import Path
import json
from collections import defaultdict

log.basicConfig(level=log.INFO)
import random

random.seed(1234)


def partition(data, val_per=0.06, test_per=0.1):
    assert 0 <= val_per < 1
    assert 0 <= test_per < 1
    subreddits = defaultdict(set)
    thread_id = 'url'
    threads = {}
    for rec in data:
        assert rec[thread_id] not in threads
        threads[rec[thread_id]] = len(rec['posts'])
        subreddits[rec['subreddit']].add(thread_id)
    log.info(f"Found {len(threads)} threads from {len(subreddits)} subreddits")
    n = len(threads)
    threads = list(threads.items())
    random.shuffle(threads)
    test_n, val_n = int(n * test_per), int(n * val_per)
    test = data[:test_n]
    valid = data[test_n:test_n + val_n]
    train = data[test_n + val_n:]
    log.info(f"Threads: Train: {len(train)}; valid: {len(valid)}; test: {len(test)}")
    allocation = {}
    stats = {}
    for split, name in [(train, 'train'), (valid, 'valid'), (test, 'test')]:
        thread_ids = set()
        stats[name] = stat = dict(posts_total=0,
                                  posts_empty=0,
                                  posts_valid=0,
                                  labels=defaultdict(int))
        for rec in split:
            thread_ids.add(rec[thread_id])
            stat['posts_total'] += len(rec['posts'])
            for post in rec['posts']:
                """
                Missing label: there was no majority annotation; it was ambiguous
                Missing body: Thread was valid, but post/comment was deleted or banned during my retrieval
                """
                label = post.get('majority_type')
                body = post.get('body')
                stat['labels'][label] += 1
                if not body:
                    stat['posts_empty'] += 1
                if  label and body:
                    stat['posts_valid'] += 1

        stat['threads'] = len(split)
        stat['posts_nolabel'] = stat['labels'].get(None, 0)
        allocation[name] = list(thread_ids)
    log.info(f"Stats:\n{json.dumps(stats, indent=2)}")
    return dict(stats=stats, partition=allocation)


def get_split(data, thread_ids):
    if not isinstance(thread_ids, set):
        thread_ids_set = set(thread_ids)
        assert len(thread_ids_set) == len(thread_ids), 'thread ids are not unique'
        thread_ids = thread_ids_set
    return [rec for rec in data if rec['url'] in thread_ids]

def flatten_posts(threads):
    count = [0, 0, 0]
    for thread in threads:
        thread_cols = [thread['url']]
        count[0] += 1
        for post in thread['posts']:
            body = ' '.join(
                post.get('body', '').split()).strip()  # remove unnecessary white spaces
            post_cols = thread_cols + [
                post['id'],
                post.get('in_reply_to', 'null'),
                post.get('majority_type', 'null'),
                body
            ]
            if not body:
                log.debug(f"Skipping empty post: {post_cols}")
                count[2] += 1
                continue
            yield post_cols
            count[1] += 1

    log.info(f'Found {count[1]} posts from {count[0]} threads')
    if count[2]:
        log.warning(f"Skipped {count[2]} posts because they had no body")

def write_tsv(recs, path, delim='\t', header=None):
    n_cols = -1
    count = 0
    with open(path, 'w', encoding='utf-8', errors='ignore') as out:
        if header:
            out.write(delim.join(header) + '\n')
            n_cols = len(header)
        for rec in recs:
            if n_cols > 0:
                assert len(rec) == n_cols
            else:
                assert len(rec) > 0
                n_cols = len(rec)
            out.write(delim.join(rec))
            out.write('\n')
            count += 1
    log.info(f"Wrote {count} recs to {path}")

def write_plain(lines, path):
    count = 0
    with open(path, 'w', encoding='utf8', errors='ignore') as out:
        for line in lines:
            out.write(line.strip())
            out.write("\n")
            count += 1
    log.info(f"Wrote {count} lines to {path}")

def main(**args):
    args = args or vars(parse_args())
    inp: Path = args['inp']
    out_dir: Path = args['out']
    split_spec = args.get('splits')
    out_dir.mkdir(parents=True, exist_ok=True)
    assert inp.exists() and inp.is_file()
    def_splits_file_name = 'splits.json'


    with inp.open(encoding='utf8', errors='ignore') as reader:
        data = [json.loads(line) for line in reader]
        log.info(f"Found {len(data)} records in {inp}")

    if not split_spec and (out_dir / def_splits_file_name).exists():
        split_spec = (out_dir / def_splits_file_name)
        log.info(f"Detected previous splits from {split_spec}")

    if split_spec:
        log.info(f"Going to recreate splits as per {split_spec}")
        assert split_spec.is_file()
        split_spec = json.loads(split_spec.read_text())
    else:
        log.info("Creating new partition")
        split_spec = partition(data, val_per=args["val_percent"], test_per=args["test_percent"])
        split_file = out_dir / 'splits.json'
        log.info(f"Writing partition spec at {split_file}")
        with split_file.open('w', encoding='utf-8') as wrtr:
            json.dump(split_spec, wrtr, indent=2)

    log.info("Creating partitions from spec")
    header = ['url', 'id', 'reply_to', 'label', 'body']
    for name in ['train', 'valid', 'test']:
        threads = get_split(data, split_spec['partition'][name])
        posts = list(flatten_posts(threads))
        write_tsv(posts, out_dir / f'{name}.tsv', header=header)
        labels = [p[3] for p in posts]
        texts = [p[4] for p in posts]
        write_plain(labels, path=out_dir / f'{name}.lbl')
        write_plain(texts, path=out_dir / f'{name}.txt')

        #exlcude nulls
        posts_notnull = [p for p in posts if p[3] not in("null", None)]
        write_tsv(posts_notnull, out_dir / f'{name}.nonull.tsv', header=header)

        labels = [p[3] for p in posts_notnull]
        texts = [p[4] for p in posts_notnull]
        write_plain(labels, path=out_dir / f'{name}.nonull.lbl')
        write_plain(texts, path=out_dir / f'{name}.nonull.txt')


def parse_args():
    import argparse
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('-i', '--inp', type=Path, required=True,
                   help='Input file path, having coarse-discourse dump')
    p.add_argument('-o', '--out', type=Path, required=True,
                   help='Output dir path where coarse discourse splits needs to be saved')
    p.add_argument('-s', '--splits', type=Path,
                   help='Split file specification (for recreating based on prior splits)')
    p.add_argument('-vp', '--val_percent', type=float, default=0.075, help="validation percent")
    p.add_argument('-tp', '--test_percent', type=float, default=0.1, help="test percent")
    return p.parse_args()


if __name__ == '__main__':
    main()
