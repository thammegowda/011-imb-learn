#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 7/19/21
from pathlib import Path
from imblearn import yaml, registry, MODEL, device

def parse_args():
    import argparse
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('exp_dir', type=Path, help='Experiment dir path (must have conf.yml in it).')
    return p.parse_args()


def main(**args):
    args = args or vars(parse_args())
    exp_dir: Path = args['exp_dir']
    conf_file = exp_dir / 'conf.yml'
    assert conf_file.exists(), f'{conf_file} expected, but it doesnt exist'
    conf = yaml.load(conf_file)
    assert 'model' in conf and 'name' in conf['model']
    model_name = conf['model']['name']
    assert model_name in registry[MODEL]
    model_cls = registry[MODEL][model_name]
    trainer = model_cls.Trainer(exp_dir, device=device)
    trainer.pipeline()

if __name__ == '__main__':
    main()
