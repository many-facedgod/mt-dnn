# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import os
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from module.dropout_wrapper import DropoutWrapper
from pytorch_pretrained_bert.modeling import BertConfig, BertEncoder, BertLayerNorm, BertModel
from module.san import SANClassifier, Classifier
import math


def gelu(input):
    return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

classifier_ids = {'mnli': 2, 'snli': 2, 'qqp': 1, 'qnli': 1, 'wnli': 1, 'rte': 1, 'mrpc': 1, 'sst': 1, 'stsb': 0, 'cola': 1}

cluster_ids = {'mrpc': 0, 'stsb': 0, 'rte': 0, 'cola': 1, 'sst': 2}

individual_ids = {'mrpc': 0, 'stsb': 1, 'rte': 2, 'cola': 3, 'sst': 4}

class SANBertNetwork(nn.Module):
    def __init__(self, opt, bert_config=None):
        super(SANBertNetwork, self).__init__()
        self.dropout_list = []
        self.bert_config = BertConfig.from_dict(opt)
        self.bert = BertModel(self.bert_config)
        if opt['update_bert_opt'] > 0:
            for p in self.bert.parameters():
                p.requires_grad = False
        mem_size = self.bert_config.hidden_size
        self.decoder_opt = opt['answer_opt']
        self.scoring_list = nn.ModuleList()
        labels = [int(ls) for ls in opt['label_size'].split(',')]
        task_dropout_p = opt['tasks_dropout_p']
        self.bert_pooler = None

        self.cluster_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.individual_layers = nn.ModuleList()
        for i in range(3):
            layer = nn.Linear(self.embedding_size, self.embedding_size)
            layer.weight.data.normal_(0.0, 0.02)
            layer.bias.data.zero_()
            self.dropout_layers.append(nn.Dropout(0.1))
            self.cluster_layers.append(layer)
        for i in range(5):
            if i == 1:
                layer = nn.Linear(self.embedding_size, 1)
            else:
                layer = nn.Linear(self.embedding_size, 2)
            layer.weight.data.normal_(0.0, 0.02)
            layer.bias.data.zero_()
            self.individual_layers.append(layer)

        self.opt = opt
        self._my_init()
        self.set_embed(opt)

    def _my_init(self):
        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=self.bert_config.initializer_range * self.opt['init_ratio'])
            elif isinstance(module, BertLayerNorm):
                # Slightly different from the BERT pytorch version, which should be a bug.
                # Note that it only affects on training from scratch. For detailed discussions, please contact xiaodl@.
                # Layer normalization (https://arxiv.org/abs/1607.06450)
                # support both old/latest version
                if 'beta' in dir(module) and 'gamma' in dir(module):
                    module.beta.data.zero_()
                    module.gamma.data.fill_(1.0)
                else:
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def nbert_layer(self):
        return len(self.bert.encoder.layer)

    def freeze_layers(self, max_n):
        assert max_n < self.nbert_layer()
        for i in range(0, max_n):
            self.freeze_layer(i)

    def freeze_layer(self, n):
        assert n < self.nbert_layer()
        layer = self.bert.encoder.layer[n]
        for p in layer.parameters():
            p.requires_grad = False

    def set_embed(self, opt):
        bert_embeddings = self.bert.embeddings
        emb_opt = opt['embedding_opt']
        if emb_opt == 1:
            for p in bert_embeddings.word_embeddings.parameters():
                p.requires_grad = False
        elif emb_opt == 2:
            for p in bert_embeddings.position_embeddings.parameters():
                p.requires_grad = False
        elif emb_opt == 3:
            for p in bert_embeddings.token_type_embeddings.parameters():
                p.requires_grad = False
        elif emb_opt == 4:
            for p in bert_embeddings.token_type_embeddings.parameters():
                p.requires_grad = False
            for p in bert_embeddings.position_embeddings.parameters():
                p.requires_grad = False

    def forward(self, input_ids, token_type_ids, attention_mask, premise_mask=None, hyp_mask=None, task_id=0, prefix=None):
        all_encoder_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = all_encoder_layers[-1]
        if self.bert_pooler is not None:
            pooled_output = self.bert_pooler(sequence_output)
        cluster_id = cluster_ids[prefix]
        individual_id = individual_ids[prefix]
        h = gelu(self.cluster_layers[cluster_id](self.dropout_layers[cluster_id](pooled_output)))
        return self.individual_layers[individual_id](h)
