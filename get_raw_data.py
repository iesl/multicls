import os
import random
import numpy as np
import psutil
import torch
import pickle
import sys

from olfmlm.configure_data import configure_data
from olfmlm.model import BertModel
from olfmlm.optim import Adam
from olfmlm.utils import save_checkpoint
from olfmlm.utils import load_checkpoint
from argparse import Namespace

from olfmlm.paths import pretrained_path

from tqdm import tqdm

save_path = sys.argv[1]

save_dir = os.path.join(save_path, 'raw_data')

'''
Initialize the model, which ara the same codes from pretrain_bert.py.
The difference is I fix the arguments and thus we don't need to type many things in the command line.
'''
args = Namespace(alternating=False, always_mlm=True, attention_dropout=0.1, batch_size=16,
                 bert_config_file='bert_config.json', cache_dir='cache_dir', checkpoint_activations=False,
                 clip_grad=1.0, continual_learning=False, cuda=True, delim=',', distributed_backend='nccl',
                 dynamic_loss_scale=True, epochs=32, eval_batch_size=None, eval_iters=2000, eval_max_preds_per_seq=None,
                 eval_seq_length=None, eval_text_key=None, eval_tokens=1000000, fp32_embedding=False,
                 fp32_layernorm=False, fp32_tokentypes=False, hidden_dropout=0.0, hidden_size=1024, incremental=False,
                 intermediate_size=None, layernorm_epsilon=1e-12, lazy_loader=True, load=None, load_all_rng=False,
                 load_optim=True, load_rng=True, local_rank=None, log_interval=1000000, loose_json=False, lr=0.0001,
                 lr_decay_iters=None, lr_decay_style='linear', max_dataset_size=None, max_position_embeddings=512,
                 max_preds_per_seq=80, model_type='rg+mlm', modes='mlm,rg', no_aux=False, num_attention_heads=16,
                 num_layers=24, num_workers=22, presplit_sentences=True, pretrained_bert=False, rank=0,
                 resume_dataloader=False, save='pretrained_berts/rg+mlm', save_all_rng=False, save_iters=None,
                 save_optim=True, save_rng=True, seed=1234, seq_length=128, shuffle=True, split='1000,1,1',
                 test_data=None, text_key='text', tokenizer_model_type='bert-base-uncased',
                 tokenizer_path='tokenizer.model', tokenizer_type='BertWordPieceTokenizer', track_results=True,
                 train_data=['bert_corpus'], train_iters=1000000, train_tokens=1000000000, use_tfrecords=False,
                 valid_data=None, vocab_size=30522, warmup=0.01, weight_decay=0.02, world_size=1)

args.num_facets=3
'''
Loading dataset
'''

data_config = configure_data()
data_config.set_defaults(data_set_type='BERT', transpose=True)
(train_data, val_data, test_data), tokenizer = data_config.apply(args)
args.data_size = tokenizer.num_tokens



def truncate_sequence(tokens):
    """
    Truncate sequence pair
    """
    max_num_tokens = val_data.dataset.max_seq_len - 1 - 3
    while True:
        if len(tokens) <= max_num_tokens:
            break
        idx = 0 if random.random() < 0.5 else len(tokens) - 1
        tokens.pop(idx)


from collections import defaultdict, Counter
word_freq = Counter()

print('Calculating token frequency...')
with tqdm(total=len(train_data.dataset.ds)) as pbar:  # the total number of training doc
    for doc in train_data.dataset.ds:
        for sent in val_data.dataset.sentence_split(doc):
            token = tokenizer.EncodeAsIds(sent).tokenization
            truncate_sequence(token)
            for i in token:
                word_freq[i]+=1
        pbar.update(1)

print('Extracting validation raw text...')
val_raw_text = []
with tqdm(total=159973) as pbar:  # the total number of validation sentences
    for doc in val_data.dataset.ds:
        for sent in val_data.dataset.sentence_split(doc):
            '''
            save the truncated and tokenized format, in order to match the gradient
            '''
            val_raw_text.append(sent)
            pbar.update(1)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

breakpoint()
freq_save_file = os.path.join(save_dir, 'freq_counter.pkl')
with open(freq_save_file, 'wb') as handle:
    pickle.dump(word_freq, handle, protocol=pickle.HIGHEST_PROTOCOL)


val_save_file = os.path.join(save_dir, 'raw_val.pkl')
with open(save_file, 'wb') as handle:
    pickle.dump(val_raw_text, handle, protocol=pickle.HIGHEST_PROTOCOL)








