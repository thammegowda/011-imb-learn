#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 4/27/21
from pathlib import Path
from typing import List
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader, ImageFolder
import sys

from .train import BaseExperiment, device, log
from .metric import ClsMetric


class Images(Dataset):
    def __init__(self, paths, transform=None, loader=default_loader):
        self.loader = loader
        self.samples = paths
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        img = self.loader(self.samples[item])
        if self.transform:
            img = self.transform(img)
        return img, item


class Evaluator(BaseExperiment):

    def __init__(self, work_dir: Path):
        super(Evaluator, self).__init__(work_dir=work_dir, device=device)
        assert self.best_checkpt.exists()
        assert self.classes
        self.model = self._get_model().to(device).eval()
        chkpt = torch.load(self.best_checkpt, map_location=device)
        log.info(f'Restoring model state from checkpoint {self.best_checkpt}; step={chkpt["step"]}')
        self.model.load_state_dict(chkpt['model_state'])
        self.checkpt_step = chkpt["step"]



    def predict_files(self, paths: List[str], num_threads=0, batch_size=10):
        images = Images(paths=paths, transform=self.eval_transform)
        loader = DataLoader(images, batch_size=batch_size, shuffle=False,
                                num_workers=num_threads, pin_memory=True)
        for batch, idxs in tqdm(loader):
            batch = batch.to(self.device)
            output = self.model(batch)
            probs = torch.softmax(output, dim=1)
            top_prob, top_idx = probs.detach().max(dim=1)
            for i in range(len(batch)):
                yield (paths[idxs[i]], self.classes[top_idx[i]], top_prob[i])

    def predict_dir(self, test_dir: str, num_threads=0, batch_size=10) -> ClsMetric:
        images = ImageFolder(test_dir, transform=self.eval_transform)
        loader = DataLoader(images, batch_size=batch_size, shuffle=False,
                            num_workers=num_threads, pin_memory=True)
        preds = []
        truth = []
        for batch, target in tqdm(loader):
            batch = batch.to(self.device)
            output = self.model(batch)
            probs = torch.softmax(output, dim=1)
            top_prob, top_idx = probs.detach().max(dim=1)
            preds.append(top_idx)
            truth.append(target)
        return ClsMetric(prediction=torch.cat(preds), truth=torch.cat(truth), clsmap=self.classes)

def main(**args):
    args = args or vars(parse_args())
    evaluator = Evaluator(args['exp_dir'])
    conf = evaluator.conf
    img_paths = []
    skips = 0
    batch_size = args.get('batch_size', conf['validation'].get('batch_size',
                                                               conf['train']['batch_size']))
    if args.get('test_dir'):
        log.info(f"Reading images and their labels from {args['test_dir']}")
        with torch.no_grad():
            result = evaluator.predict_dir(args['test_dir'], batch_size=batch_size)
        args['out'].write(result.format(confusion=True, delim=','))
    else:
        log.info(f"Reading image paths from {args['inp']}")
        for path in args['inp']:
            path = path.strip()
            if not os.path.exists(path):
                log.warning(f"{path} not found")
                skips += 1
                continue
            img_paths.append(path)
        log.info(f"Total images: {len(img_paths)}; skipped={skips}")
        with torch.no_grad():
            result = evaluator.predict_files(img_paths, batch_size=batch_size)
        for path, label, prob in result:
            args['out'].write(f'{path}\t{label}\t{prob:g}\n')
    log.info("Done")


def parse_args():
    import argparse
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('exp_dir', type=Path, help='Experiment dir path (must have conf.yml in it).')
    inp_p = p.add_mutually_exclusive_group()
    inp_p.add_argument('-i', '--inp', type=argparse.FileType('r'), default=sys.stdin,
                   help='Input: text file having one test image path per line.')
    inp_p.add_argument('-t', '--test-dir',  help='Test dir where class labels are subdir')

    p.add_argument('-o', '--out', type=argparse.FileType('w'), default=sys.stdout,
                   help='Output file path to store results')
    p.add_argument('-b', '--batch-size', type=int, default=5, help='Help')
    return p.parse_args()


if __name__ == '__main__':
    main()
