Code for "On Losses for Modern Language Models" (ACL: https://www.aclweb.org/anthology/2020.emnlp-main.403/, arxiv: https://arxiv.org/abs/2010.01694)

This repository is primarily for reproducibility and posterity. It is not maintained.

Thank you to NVIDIA and NYU's jiant group for their code which helped create the base of this repo. Specifically
https://github.com/NVIDIA/Megatron-LM/commits/master (commit 0399d32c75b4719c89b91c18a173d05936112036)  
and  
https://github.com/nyu-mll/jiant/commits/master (commit 14d9e3d294b6cb4a29b70325b2b993d5926fe668)  
were used.

# Setup
Only tested on python3.6.

```
python -m pip install virtualenv
virtualenv bert_env
source bert_env/bin/activate
pip install -r requirements.txt
```

# To do
+olfmlm/evaluate/generate_random_number.py
If I use bert-large-uncased tokenizer during the testing time, I will get assert failure on this line: assert (ids > 1).all()  in the file olfmlm/evaluate/bert_embedder.py. However, it seems to be fine if I use bert-base-uncased tokenizer for the bert-large-uncased model. I believe the tokenizer of bert-large-uncased and bert-base-uncased should do the same thing because their vocab file should be the same, but it seems that allennlp handle the bert-base-uncased and bert-large-uncased differently.
It might be a bug in allennlp 0.8.4
Before releasing the code, remember to change the num-workers to be None as default


# Usage
The code is built on the source code of [On Losses for Modern Language Models](https://github.com/StephAO/olfmlm) with several enhancements and modifications.
In addition to previous proposed pre-training tasks ("mlm", "rg (QT) in the paper", "tf", "tf_idf", "so"...etc), we provide a new training mechanism for transformers which enjoys the benefits of ensembling
without sacrificing efficiency. To train our Multi-CLS Bert, simply specify `--model-type mf` (MCQT in paper) with number of facets `K` you want via `--num-facets K`.

Currently `mf` type can be combined with any of the following methods:
- Using Hard Negative `--use_hard_neg 1`
- Architecture-based Diversification `--diversify_mode`
- On which layer of BERT to insert additional linear layer `--diversify_hidden_layer`
- Enable cross facets loss`--facet2facet`
- λ in our MCQT loss: `agg_weight`
- Always using MLM loss (--always-mlm True). The loss of pre-training task will be "mf" + "mlm".
- Initialize with pretrained bert's weight `--pretrained`

When pre-training with multi-tasks, the loss function can calculated using any of the following methods:
- Summing all losses (default, incompatible between a small subset of tasks, see paper for more detail)
- Continuous Multi-Task Learning, based on ERNIE 2.0 (--continual-learning True)
- Alternating between losses (--alternating True)

To view all usable parameters that shares by all different pretrain tasks, you may find them in `arguments.py`.

Note that our code still supports those comparing tasks listed in our paper, you may just change the model type to reproduce the result (ex: using `--model-type rg+so+tf_idf` to perform MTL method )

Before training, you should 
* Set paths to read/save/load from in paths.py
* To create datasets, see data_utils/make_dataset.py
* For tf_idf prediction, you need to first calculate the idf score for your dataset. See idf.py for a script to do this.
* If you want to change the transformer size, check out bert_config.json.
* If you want to train bert-large, you may use `bert_large_config.json` with `--tokenizer-model-type bert-large-uncased`.

The following command is the best setting that we used our paper for Multi-Bert
```angular2html
python -m olfmlm.pretrain_bert --model-type mf,tf_idf,so --pretrained-bert --save-iters 200000 --lr 2e-5 --agg-function max --warmup 0.001 --facet2facet  --epochs 2 --num-facets 5 --diversify_hidden_layer 4,8 --loss_mode log  --use_hard_neg 1 --batch-size 30 --seed 1 --diversify_mode lin_new --add_testing_agg --agg_weight 0.1 --save_suffix _add_testing_agg_max01_n5_no_pooling_no_bias_h48_lin_no_bias_hard_neg_tf_idf_so_bsz_30_e2_norm_facet_warmup0001_s1
```

## Fine-tuning
Before running fine-tuning task, change `output_path` in `evaluate/generate_random_number.py` as well as `random_file_path` in `olfmlm/evaluate/config/test_bert.conf` to your local path. Run the python file to generate random number, which is to ensure the random seeds for training data sampling remain same while fine-tuning.

To run fine-tuning task:
You will need to convert the saved state dict of the required model using the `convert_state_dict.py` file.
Then run:
`python3 -m olfmlm.evaluate.main --exp_name [experiment name] --overrides parameters_to_Overide`
Where experiment name is the same as the model type above. If using a saved checkpoint instead of the best model, use the --checkpoint argument.
You may change the data you want to use in `olfmlm/paths.py`, can be glue or super glue. As for the `--overrides`, this parameter accepts command like strings to override the default values in fine-tuning config (`olfmlm/evaluate/config/test_bert.conf`). You may specify learning rate, model_suffix or few shot setting there.

In Multi-Bert, we provide different ways to aggregate all the CLS embeddings. To specify the aggregation function, change the value of `pool_type` in `olfmlm/evaluate/config/test_bert.conf`
- Re-parameterization `pool_type=proj_avg_train`
- Sum Aggregation `pool_type=first_init`

The following command is an example to run fine-tuning task on Glue dataset with few shot sample size =100. Use run name with suffix to reload the model weight you saved from pretraining.

```angular2html
common_para="warmup_ratio = 0.1, max_grad_norm = 1.0, pool_type=proj_avg_train, "
common_name="warmup01_clip1_proj_avg_train_correct"

python -m olfmlm.evaluate.main 
--exp_name $exp_name 
--overrides "run_name = ${model_name}_1, 
$common_para pretrain_tasks = glue}, 
target_tasks = glue, 
lr=1e-5, batch_size=4, few_shot = 32, max_epochs = 20, 
pooler_dropout = 0, random_seed = 1, 
run_name_suffix = adam_${common_name}_e20_bsz4:s1:lr"
```



