#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 5/7/21


from pathlib import Path
from datasets import load_dataset
from imblearn.common.exp import BaseTrainer
from tokenizers.implementations import ByteLevelBPETokenizer

paths = [str(x) for x in Path(".").glob("**/*.txt")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])


class MLMTrainer(BaseTrainer):

    def __init__(self, work_dir, *args, **kwargs):
        super(MLMTrainer, self).__init__(*args, work_dir=work_dir, **kwargs)
        self.data_dir = self.work_dir / 'data'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._prepared_flag = self.data_dir / '_PREPARED'  # fully

    def prepare(self):
        if self._prepared_flag.exists():
            return
        raise Exception('Not implemented yet')

