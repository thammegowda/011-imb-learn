#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 5/7/21


from pathlib import Path
from copy import copy
from typing import List, Dict, Union
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

from datasets import load_dataset
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import RobertaTokenizerFast
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer as HFTrainer, TrainingArguments as HFTrainerArgs

from imblearn.common.exp import BaseTrainer
from imblearn import device, log

DEF_special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>", "<cls>"]


class MLMTrainer(BaseTrainer):

    def __init__(self, work_dir, *args, **kwargs):
        super(MLMTrainer, self).__init__(*args, work_dir=work_dir, **kwargs)
        self.data_dir = self.work_dir / 'data'
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._prepared_flag = self.data_dir / '_PREPARED'  # fully
        self._train_file = self.data_dir / 'train.pkl'
        self._val_file = self.data_dir / 'val.pkl'

        self.tokenizer = self.get_tokenizer()

    def get_tokenizer(self, prefix='bpevocab'):
        args = self.conf['prep']
        vocab_dir = self.data_dir / prefix
        vocab_dir.mkdir(exist_ok=True)
        vocab_file = vocab_dir / 'vocab.json'
        merge_file = vocab_dir / 'merges.txt'
        if not vocab_file.exists() or not merge_file.exists():
            train_path = self.conf['train']['data']
            log.info(f"Creating vocabulary from train_data {train_path}")
            assert Path(train_path).exists(), f'train.data: {train_path} not found'

            bpe_args = dict(files=[train_path],
                            vocab_size=args['vocab_size'],
                            min_frequency=args['min_frequency'],
                            special_tokens=args.get('special_tokens', DEF_special_tokens))

            tokenizer = ByteLevelBPETokenizer()
            tokenizer.train(**bpe_args)
            tokenizer.save_model(str(vocab_dir))
        assert vocab_file.exists()
        max_length = args.get('max_length', 512)
        tokr = RobertaTokenizerFast.from_pretrained(vocab_dir, max_length=max_length)
        """tokr = ByteLevelBPETokenizer(vocab=str(vocab_file), merges=str(merge_file))
        tokr._tokenizer.post_processor = BertProcessing(
            ("</s>", tokr.token_to_id("</s>")),
            ("<s>", tokr.token_to_id("<s>")))
        
        tokr.enable_truncation(max_length=max_length)
        """
        log.info(f"vocab loaded from {vocab_dir}")
        return tokr

    def prepare(self):
        if self._prepared_flag.exists():
            log.info("Experiment is prepared")
            return
        if not self._train_file.exists():
            txt_file = Path(self.conf['train']['data'])
            self._prepare_file(txt_file, self._train_file)
        if not self._val_file.exists():
            txt_file = Path(self.conf['validation']['data'])
            self._prepare_file(txt_file, self._val_file)

    def _prepare_file(self, txt_file: Path, pkl_file: Path, block_size=128):
        log.info(f"Binarizing ... {txt_file}")
        data = LineByLineTextDataset(tokenizer=self.tokenizer, file_path=str(txt_file),
                                     block_size=block_size)
        log.info(f"Saving to {pkl_file}")
        torch.save(data, pkl_file)

    def train(self, max_epoch: int, batch_size: int, checkpoint: int):
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15)
        args = HFTrainerArgs(
            output_dir=str(self.models_dir),
            overwrite_output_dir=True,
            num_train_epochs=max_epoch,
            per_device_train_batch_size=batch_size,
            save_steps=checkpoint,
            save_total_limit=2,
            prediction_loss_only=True,
        )
        train_data = torch.load(self._train_file)
        _trainer = HFTrainer(model=self.model, args=args, data_collator=data_collator,
                             train_dataset=train_data)
        _trainer.train()  #resume_from_checkpoint=True


class TextDataset(Dataset):

    def __init__(self, tokenizer, path: Path):
        self.tokenizer = tokenizer
        if not isinstance(path, Path):
            path = Path(path)
        assert path.exists()
        lines = path.read_text(encoding="utf-8", errors='ignore').splitlines()
        self.examples = [x.ids for x in tokenizer.encode_batch(lines)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i])


def parse_args():
    import argparse
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('exp_dir', type=Path, help='Experiment dir path (must have conf.yml in it).')
    return p.parse_args()


def main(**args):
    args = args or vars(parse_args())
    exp_dir: Path = args['exp_dir']
    trainer = MLMTrainer(exp_dir, device=device)
    trainer.prepare()

    train_args: Dict = copy(trainer.conf['train'])
    train_args.pop('data', None)
    trainer.train(**train_args)

    if trainer.conf.get('tests'):
        log.info('Running tests not implemented yet')


if __name__ == '__main__':
    main()
