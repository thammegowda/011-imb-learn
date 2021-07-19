#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 7/18/21

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, MultiheadAttention
from imblearn.registry import register, MODEL

@register(MODEL)
class TransformerSeqClassifier(nn.Module):

    def __init__(self, vocab_size, n_classes, model_dim, n_heads, ff_dim, n_layers, dropout=0.1):
        super().__init__()
        self.model_type = 'Transformer'
        self.encoder = nn.Embedding(vocab_size, model_dim)
        self.model_dim = model_dim
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        encoder_layers = TransformerEncoderLayer(model_dim, n_heads, ff_dim, dropout,
                                                 batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        attn = MultiheadAttention(model_dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.compressor = SentenceCompressor(model_dim=model_dim, attn=attn)
        self.classifier = Classifier(model_dim=model_dim, n_classes=n_classes)
        self.init_params()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def init_params(self, scheme='xavier'):
        assert scheme == 'xavier'  # only supported scheme as of now
        # Initialize parameters with xavier uniform
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask, score='logits'):
        src = self.encoder(src) * math.sqrt(self.model_dim)
        src = self.pos_encoder(src)

        tok_repr = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        sent_repr = self.compressor(tok_repr, src_mask)
        output = self.classifier(sent_repr, score=score)
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, model_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class SentenceCompressor(nn.Module):
    """
    Compresses token representation into a single vector
    """

    def __init__(self, model_dim: int, attn: MultiheadAttention):
        super().__init__()
        self.cls_repr = nn.Parameter(torch.zeros(model_dim))
        self.d_model = model_dim
        self.attn = attn

    def forward(self, src, src_mask):
        B, T, D = src.size()  # [Batch, Time, Dim]
        assert D == self.d_model
        query = self.cls_repr.view(1, 1, D).repeat(B, 1, 1)
        # Args: Query, Key, Value, Mask
        cls_repr = self.attn(query, src, src, src_mask)[0]
        cls_repr = cls_repr.view(B, D)  # [B, D]
        return cls_repr


class Classifier(nn.Module):
    scores = {
        'logits': lambda x, dim=None: x,
        'softmax': F.softmax,
        'log_softmax': F.log_softmax,
        'sigmoid': lambda x, dim=None: x.sigmoid(),
    }

    def __init__(self, model_dim: int, n_classes: int):
        super().__init__()
        self.d_model = model_dim
        self.n_classes = n_classes
        self.proj = nn.Linear(model_dim, n_classes)

    def forward(self, repr, score='logits'):
        B, D = repr.shape  # [Batch, Dim]
        assert D == self.d_model
        assert score in self.scores, f'"score", Given={score}, known={list(self.scores.keys())}'
        cls_repr = self.proj(repr)
        return self.scores[score](cls_repr, dim=-1)
