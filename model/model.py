# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for wrapping BertModel."""

import torch

from olfmlm.model.modeling import BertConfig
from olfmlm.model.modeling import BertLayerNorm
from olfmlm.model.new_models import BertHeadTransform

from olfmlm.model.new_models import Bert

def get_params_for_weight_decay_optimization(module):

    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0}
    for module_ in module.modules():
        if isinstance(module_, (BertLayerNorm, torch.nn.LayerNorm)):
            no_weight_decay_params['params'].extend(
                [p for p in list(module_._parameters.values())
                 if p is not None])
        else:
            weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n != 'bias'])
            no_weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n == 'bias'])

    return weight_decay_params, no_weight_decay_params


# def get_params_for_weight_decay_optimization(module):
#
#     weight_decay_params = {'params': []}
#     no_weight_decay_params = {'params': [], 'weight_decay': 0}
#     for module_ in module.modules():
#         if isinstance(module_, (BertLayerNorm, torch.nn.LayerNorm)):
#             no_weight_decay_params['params'].extend(
#                 [p for p in list(module_._parameters.values())
#                  if p is not None])
#         else:
#             weight_decay_params['params'].extend(
#                 [p for n, p in list(module_._parameters.items())
#                  if p is not None and n != 'bias'])
#             no_weight_decay_params['params'].extend(
#                 [p for n, p in list(module_._parameters.items())
#                  if p is not None and n == 'bias'])
#     return weight_decay_params, no_weight_decay_params
#

# def get_params_for_dict_format(module_dict):
#     '''
#     Although the above function also can return the same result, just create another one in case of other issues.
#     '''
#     weight_decay_params = {'params': []}
#     no_weight_decay_params = {'params': [], 'weight_decay': 0}
#     for module_obj in module_dict:
#         if isinstance(module_obj, BertHeadTransform):
#             continue
#         for module_ in module_obj.modules():
#             if isinstance(module_, (BertLayerNorm, torch.nn.LayerNorm)):
#                 no_weight_decay_params['params'].extend(
#                     [p for p in list(module_._parameters.values())
#                      if p is not None])
#             else:
#                 weight_decay_params['params'].extend(
#                     [p for n, p in list(module_._parameters.items())
#                      if p is not None and n != 'bias'])
#                 no_weight_decay_params['params'].extend(
#                     [p for n, p in list(module_._parameters.items())
#                  if p is not None and n == 'bias'])
#     return weight_decay_params, no_weight_decay_params


class BertModel(torch.nn.Module):

    def __init__(self, tokenizer, args):

        super(BertModel, self).__init__()
        # if args.pretrained_bert:
        #     self.model = BertForPreTraining.from_pretrained(
        #         args.tokenizer_model_type,
        #         cache_dir=args.cache_dir,
        #         fp32_layernorm=args.fp32_layernorm,
        #         fp32_embedding=args.fp32_embedding,
        #         layernorm_epsilon=args.layernorm_epsilon)
        # else:
        if args.bert_config_file is None:
            raise ValueError("If not using a pretrained_bert, please specify a bert config file")
        self.config = BertConfig(args.bert_config_file)
        model_args = [self.config]
        # if self.model_type == "referential_game":
        #     self.small_config = BertConfig(args.bert_small_config_file)
        #     model_args.append(self.small_config)
        
        #diversify_hidden_layer = 6
        #diversify_hidden_layer = [8]
        #diversify_hidden_layer = [4,8]
        #diversify_hidden_layer = list(range(12))
        #diversify_hidden_layer = -1
        if len(args.diversify_hidden_layer) > 0:
            diversify_hidden_layer = sorted([int(x) for x in args.diversify_hidden_layer.split(',')])
        else:
            diversify_hidden_layer = []
        self.model = Bert(*model_args, modes=args.modes.split(','), extra_token=args.extra_token,
                          agg_function=args.agg_function, unnorm_facet=args.unnorm_facet,
                          unnorm_token=args.unnorm_token, facet2facet=args.facet2facet, facet2facet_mr=args.facet2facet_mr,
                          use_dropout=args.use_dropout, autoenc_reg_const=args.autoenc_reg_const, num_facets=args.num_facets, use_proj_bias=args.use_proj_bias, use_mulitasks=args.use_multitasks, diversify_hidden_layer = diversify_hidden_layer, diversify_mode =args.diversify_mode, bi_head=args.bi_head, same_out=args.same_out, loss_mode=args.loss_mode, use_hard_neg=args.use_hard_neg, skip_act=args.skip_act,
                          facet_mode=args.facet_mode, minus_avg_div=args.minus_avg_div, testing_agg=args.testing_agg, so_sharing=args.so_sharing, add_testing_agg=args.add_testing_agg, agg_weight=args.agg_weight,
                          post_normalization=args.post_normalization, facet_mask=args.facet_mask)
        if args.pretrained_bert:
            print('use pretrained weight')
            #self.model.bert=self.model.bert.from_pretrained('bert-base-uncased',cache_dir=args.cache_dir,config_file_path=args.bert_config_file)
            self.model.bert=self.model.bert.from_pretrained(args.tokenizer_model_type,cache_dir=args.cache_dir,config_file_path=args.bert_config_file)
            #print(self.model.bert.embeddings.word_embeddings.weight.data.shape)
            #for i in range(1,args.num_facets+1):
            #    tok_id = tokenizer.get_command('s_'+str(i)).Id
            #    #print(tok_id, self.model.bert.embeddings.word_embeddings.weight.data[tok_id,:])
            #    self.model.bert.embeddings.word_embeddings.weight.data[tok_id, :].normal_(mean=0.0, std=self.config.initializer_range)
            if 'mf' in args.modes:
                with torch.no_grad():
                    self.model.lm.decoder.weight = self.model.bert.embeddings.word_embeddings.weight 
                    if args.same_weight:
                        if args.same_out:
                            init_num = 1
                        else:
                            init_num = args.num_facets
                        for i in range(1,init_num+1):
                            self.model.sent.mf['v_{}'.format(i)].dense.weight.data = self.model.bert.pooler.dense.weight.data
                            self.model.sent.mf['s_{}'.format(i)].dense.weight.data = self.model.bert.pooler.dense.weight.data
                            if self.model.bi_head:
                                self.model.sent.mf['r_{}'.format(i)].dense.weight.data = self.model.bert.pooler.dense.weight.data
                            if args.use_proj_bias:
                                self.model.sent.mf['v_{}'.format(i)].dense.bias.data = self.model.bert.pooler.dense.bias.data
                                self.model.sent.mf['s_{}'.format(i)].dense.bias.data = self.model.bert.pooler.dense.bias.data
                                if self.model.bi_head:
                                    self.model.sent.mf['r_{}'.format(i)].dense.bias.data = self.model.bert.pooler.dense.bias.data
                            #self.model.sent.mf['v_{}'.format(i)].dense.weight = torch.nn.Parameter(self.model.bert.pooler.dense.weight.data)
                            #self.model.sent.mf['s_{}'.format(i)].dense.weight = torch.nn.Parameter(self.model.bert.pooler.dense.weight.data)
                    if args.same_weight_internal:
                        for i in range(len(diversify_hidden_layer)):
                            for j in range(args.num_facets - 1):
                                self.model.facet_lin_emb[i][j+1].weight.data = self.model.facet_lin_emb[i][0].weight.data


            #self.model.bert=self.model.bert.from_pretrained('bert-base-uncased',cache_dir=args.cache_dir,config_file_path=args.bert_config_file)
            
            

    def forward(self, modes, input_tokens, token_type_ids=None, task_ids=None, attention_mask=None, checkpoint_activations=False, first_pass=False, output_attentions=False):
        return self.model(modes, input_tokens, token_type_ids, task_ids, attention_mask, checkpoint_activations=checkpoint_activations, output_attentions=output_attentions)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.model.state_dict(destination=destination, prefix=prefix,
                                     keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)

    # def get_params(self):
    #     param_groups = []
    #     param_groups += list(get_params_for_weight_decay_optimization(self.model.bert.encoder.layer))
    #     param_groups += list(get_params_for_weight_decay_optimization(self.model.bert.pooler))
    #     param_groups += list(get_params_for_weight_decay_optimization(self.model.bert.embeddings))
    #     for classifier in self.model.sent.values():
    #         if isinstance(classifier, torch.nn.ModuleDict):
    #             #handle the mf dict type
    #             classifier = classifier.values()
    #             param_groups += list(get_params_for_dict_format(classifier))
    #         else:
    #             param_groups += list(get_params_for_weight_decay_optimization(classifier))
    #     for k, classifier in self.model.tok.items():
    #         if k == "sbo":
    #             param_groups += list(get_params_for_weight_decay_optimization(classifier.transform))
    #             param_groups[1]['params'].append(classifier.bias)
    #         else:
    #             param_groups += list(get_params_for_weight_decay_optimization(classifier))
    #     param_groups += list(get_params_for_weight_decay_optimization(self.model.lm.transform))
    #     param_groups[1]['params'].append(self.model.lm.bias)
    #
    #     return param_groups
    def get_params(self):
        param_groups = []
        param_groups += list(get_params_for_weight_decay_optimization(self.model.bert.encoder.layer))
        param_groups += list(get_params_for_weight_decay_optimization(self.model.bert.pooler))
        param_groups += list(get_params_for_weight_decay_optimization(self.model.bert.embeddings))
        for classifier in self.model.sent.values():
            param_groups += list(get_params_for_weight_decay_optimization(classifier))
        for k, classifier in self.model.tok.items():
            if k == "sbo":
                param_groups += list(get_params_for_weight_decay_optimization(classifier.transform))
                param_groups[1]['params'].append(classifier.bias)
            else:
                param_groups += list(get_params_for_weight_decay_optimization(classifier))
        param_groups += list(get_params_for_weight_decay_optimization(self.model.lm.transform))
        param_groups[1]['params'].append(self.model.lm.bias)

        return param_groups
