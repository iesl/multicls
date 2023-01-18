import numpy as np
import math
import os
# import scipy
import scipy.stats

folder_list = ['finetuned_berts/mf']

care_model_list = ['max___log_n1_no_pooling_no_bias_h48_pos_no_bias_hard_neg_tf_idf_so_bsz_48_large_warmup0001',
                   'max___log_n5_no_pooling_no_bias_h48_pos_no_bias_hard_neg_tf_idf_so_bsz_48_large_warmup0001',
                   'max___add_testing_agg_max01_n5_no_pooling_no_bias_h48_pos_no_bias_hard_neg_tf_idf_so_bsz_48_large_warmup0001',
                   'max___add_testing_agg_n5_no_pooling_no_bias_h48_pos_no_bias_hard_neg_tf_idf_so_bsz_48_large_warmup0001',
                   'max___testing_agg_n5_no_pooling_no_bias_h48_pos_no_bias_hard_neg_tf_idf_so_bsz_48_large_warmup0001']
care_model_list += [
    'max___add_testing_agg_max01_n5_no_pooling_no_bias_h48_lin_no_bias_hard_neg_tf_idf_so_bsz_30_e2_norm_facet_warmup0001',
    'max___add_testing_agg_n5_no_pooling_no_bias_h48_lin_no_bias_hard_neg_tf_idf_so_bsz_30_e2_norm_facet_warmup0001',
    'max___testing_agg_n5_no_pooling_no_bias_h48_lin_no_bias_hard_neg_tf_idf_so_bsz_30_e2_norm_facet_warmup0001']
care_model_list += [
    'max___add_testing_agg_max01_n3_no_pooling_no_bias_h48_lin_no_bias_hard_neg_tf_idf_so_bsz_30_e2_norm_facet_warmup0001',
    'max___add_testing_agg_max01_n10_no_pooling_no_bias_h48_lin_no_bias_hard_neg_tf_idf_so_bsz_30_e2_norm_facet_warmup0001',
    'max___add_testing_agg_max01_n5_no_pooling_no_bias_h48_lin_no_bias_tf_idf_so_bsz_30_e2_norm_facet_warmup0001',
    'max___log_n1_no_pooling_no_bias_h48_pos_no_bias_tf_idf_so_bsz_48_large_warmup0001',
    'max___add_testing_agg_max01_n5_no_pooling_no_bias_h48_pos_no_bias_tf_idf_so_bsz_48_large_warmup0001',
    'max___add_testing_agg_max01_n5_no_pooling_no_bias_no_bias_hard_neg_tf_idf_so_bsz_30_e2_norm_facet_warmup0001']

# care_model_list = ['bert_base_org', 'mlm_bsz30_warmup0001', 'mlm_so_tf_idf_bsz30_warmup0001', 'rg_mlm_so_tf_idf_bsz30_warmup0001', 'rg_mlm_so_tf_idf_bsz30_warmup0001_continual', 'rg_mlm_so_tf_idf_large_e1_bsz48_warmup0001']
# care_model_list = []

# care_model_set = set(care_model_list)

care_ft_list = ['warmup02_clip1_l2_e-6_proj_avg_train_ep20', 'adam_warmup01_clip1_proj_avg_train_correct_e20',
                'adam_warmup01_clip1_first_init']
# care_ft_list = ['adam_warmup01_clip1_e20', 'adam_warmup01_clip1_proj_init_e20', '_adam_warmup02_clip1_l2_e-6_e20']

# input_path_arr = ["+mlm_too_large/results.tsv", "+mlm/results.tsv"]
# input_path_arr = ["+mlm_few/results.tsv"]
input_path_arr = ["+mlm_few_100_ry/results.tsv"]
# input_path_arr = ["+mlm_super_glue/results.tsv"]
# input_path_arr = ["+mlm_super_glue_few/results.tsv"]
# input_path_arr = ["+mlm_super_glue_few_100_ry/results.tsv"]

# input_path_arr = [ "+mlm_super_glue_too_large/results.tsv", "+mlm_super_glue/results.tsv", "+mlm_super_glue_few/results.tsv"]
# input_path_arr = ["+mlm_super_glue_few/results.tsv", "", "+mlm_super_glue/results.tsv"]

# input_path_arr = ["+mlm_few_noise/results.tsv"]
# input_path_arr = ["+mlm_few_32/results.tsv"]
# input_path_arr = ["+mlm/results.tsv"]
# input_path_arr = ["+mlm_super_glue/results.tsv", "+mlm_super_glue_few/results.tsv"]

get_task_list = [[], [], ['commitbank', 'copa']]

# input_path_arr = ["+mlm/results.tsv"]
# input_path_arr = [ "+mlm_super_glue/results.tsv", "+mlm_super_glue_too_large/results.tsv", "+mlm_super_glue_few/results.tsv"]
# input_path_arr = [ "+mlm/results.tsv", "+mlm_too_large/results.tsv" ]

# input_path = "+mlm/results.tsv"
# input_path = "+mlm_few/results.tsv"
# input_path = "+mlm_few_100_ry/results.tsv"
# input_path = "+mlm_few_32/results.tsv"
# input_path = "+mlm_super_glue/results.tsv"
# input_path = "+mlm_super_glue_few/results.tsv"
# input_path = "+mlm_super_glue_few_100_ry/results.tsv"
# input_path = "+mlm_few_noise/results.tsv"


task_d2_metric = {'mnli': 'accuracy', 'qqp': 'f1', 'qnli': 'accuracy', 'sst': 'accuracy', 'cola': 'mcc',
                  'sts-b': 'spearmanr', 'mrpc': 'f1', 'rte': 'accuracy',
                  'boolq': 'accuracy', 'commitbank': 'accuracy;f1', 'wic': 'accuracy', 'copa': 'accuracy',
                  'multirc': 'ans_f1;em', 'rte-superglue': 'accuracy', 'winograd-coreference': 'acc', 'record': 'f1;em'}
task_order = ['cola_mcc', 'sst_accuracy', 'mrpc_f1', 'sts-b_spearmanr', 'qqp_f1', 'mnli_accuracy', 'qnli_accuracy',
              'rte_accuracy',
              'boolq_accuracy', 'commitbank_accuracy', 'commitbank_f1', 'copa_accuracy', 'multirc_ans_f1', "multirc_em",
              'rte-superglue_accuracy', 'wic_accuracy', 'winograd-coreference_acc']
# BoolQ      CB      COPA    MultiRC ReCoRD  RTE     WiC     WSC]
# exclude_task_set = set([])
exclude_task_set = set(['record'])
# exclude_task_set = set(['mnli', 'qnli', 'qqp', 'sst'])
merge_method_runs = True
# merge_method_runs = False
remove_duplication = True
use_only_too_large = True
# use_only_too_large = False
# remove_duplication = False
exclude_seed = set([])
# exclude_seed = set(['s4'])
# exclude_seed = set(['s3'])
# exclude_seed = set(['s1'])
# exclude_seed = set(['s2', 's3', 's4'])
# exclude_seed = set(['s9', 's10', 's11','s12', 's13', 's14','s15', 's16'])
# exclude_training_seed = ['_s2', '_s3', '_s4']
exclude_training_seed = []
# exclude_training_seed = ['_s3', '_s4']
# exclude_training_seed = ['_s2','_s3', '_s4']
# exclude_training_seed = ['_s1']
# exclude_training_seed = ['_s2']
method_d2_task_d2_lr_d2_scores = {}

max_train_seed = 4
max_test_seed = 4

all_seed_set = set()
for i in range(max_train_seed):
    for j in range(max_test_seed):
        all_seed_set.add('t' + str(i + 1) + '_s' + str(j + 1))

for folder_name in folder_list:
    # print(folder_name)
    if use_only_too_large:
        iter_range = range(len(input_path_arr))
    else:
        iter_range = range(len(input_path_arr) - 1, -1, -1)
    for i in iter_range:
        # for i in range(len(input_path_arr)-1,-1,-1):
        # print(i)
        input_path = input_path_arr[i]
        get_task = get_task_list[i]
        # print(folder_name + input_path)
        if not os.path.exists(folder_name + input_path):
            continue
        with open(folder_name + input_path) as f_in:
            for line in f_in:
                # print(line)
                fields = line.rstrip().split('\t')
                if len(fields) != 2:
                    print('skip', line)
                    continue
                method_name, scores_str = fields
                # method_name, scores_str = line.rstrip().split('\t')
                if len(get_task) > 0:
                    get_run = False
                    for task in get_task:
                        if task in scores_str:
                            get_run = True
                            break
                    # print(scores_str, get_run, get_task)
                    if not get_run:
                        continue

                method_name = method_name.replace('token:', 'token;')
                if ':' in method_name:
                    model_name, random_seed_str, lr_str = method_name.split(':')
                else:
                    lr_str = method_name[-2:]
                    method_name_arr = []
                    for info in method_name[:-2].split('_'):
                        if 's1k' in info:
                            random_seed_str = info
                        else:
                            method_name_arr.append(info)
                    model_name = '_'.join(method_name_arr)
                if random_seed_str in exclude_seed:
                    continue
                model_name_raw = model_name
                seed_index_start = model_name.rfind('_s')
                if seed_index_start > 0:
                    training_seed_num = model_name[seed_index_start + 2:seed_index_start + 4]
                    if not training_seed_num.isnumeric():
                        training_seed_num = model_name[seed_index_start + 2]
                    if not training_seed_num.isnumeric():
                        continue
                    if '_s' + training_seed_num in exclude_training_seed:
                        continue
                if len(care_model_list) > 0:
                    skip_run = True
                    for care_model in care_model_list:
                        if care_model in model_name_raw:
                            skip_run = False
                            break
                    if skip_run:
                        continue
                if len(care_ft_list) > 0:
                    skip_run_2 = True
                    for care_ft in care_ft_list:
                        if care_ft in model_name_raw:
                            skip_run_2 = False
                            break
                    if skip_run_2:
                        continue
                # skip_run = False
                # for training_seed in exclude_training_seed:
                #    if training_seed in model_name_raw:
                #        skip_run = True
                #        break
                # if skip_run:
                #    continue
                if merge_method_runs:
                    # model_name = model_name.replace('_v2','').replace('_v3','').replace('_warmup01','')
                    # model_name = model_name.replace('_v2','').replace('_v3','')
                    model_name = model_name.replace('_s1', '').replace('_s2', '').replace('_s3', '').replace('_s4', '')
                    if len(input_path_arr) > 1:
                        model_name = model_name.replace('_bsz4', '').replace('_bsz8', '')
                    # model_name = model_name.replace('_v2','').replace('_v3','').replace('_bsz4','')
                scores_fields = scores_str.split(',')

                first_special_metric_fields = scores_fields[2].split('_')
                task_name = first_special_metric_fields[0].strip()
                if task_name not in task_d2_metric or task_name in exclude_task_set:
                    continue
                target_metric_arr = task_d2_metric[task_name].split(';')
                score_metric = []
                # target_metric = task_d2_metric[task_name]
                for field in scores_fields[2:]:
                    metric_name, score_str = field.split(':')
                    metric_name = metric_name.strip()
                    for target_metric in target_metric_arr:
                        if metric_name == task_name + '_' + target_metric:
                            score_metric.append([score_str, metric_name])
                for score_str, metric_name in score_metric:
                    score = float(score_str)

                    macro_str, score_str = scores_fields[1].split(':')
                    macro_score = float(score_str)

                    task_d2_lr_d2_scores = method_d2_task_d2_lr_d2_scores.get(model_name, {})
                    lr_d2_scores = task_d2_lr_d2_scores.get(metric_name, {})
                    scores = lr_d2_scores.get(lr_str, [[], [], 1, set(), []])
                    # if use_only_too_large:
                    #    duplication_name = model_name_raw + random_seed_str
                    # else:
                    #    duplication_name = model_name_raw + random_seed_str + str(i)
                    duplication_name = model_name_raw + random_seed_str + str(i)

                    if remove_duplication and duplication_name in scores[3]:
                        continue
                    scores[0].append(macro_score)
                    scores[1].append(score)
                    scores[2] = 1 / float(len(score_metric))
                    if use_only_too_large and "too_large" in input_path:
                        for i in iter_range:
                            scores[3].add(model_name_raw + random_seed_str + str(i))
                    else:
                        scores[3].add(duplication_name)
                    scores[4].append('t' + training_seed_num + '_' + random_seed_str)
                    lr_d2_scores[lr_str] = scores
                    task_d2_lr_d2_scores[metric_name] = lr_d2_scores
                    method_d2_task_d2_lr_d2_scores[model_name] = task_d2_lr_d2_scores

for method_name in method_d2_task_d2_lr_d2_scores:
    method_score_arr = []
    method_macro_score_arr = []
    method_max_score_arr = []
    method_max_macro_score_arr = []
    lr_d2_task_score = {}
    lr_d2_task_macro_score = {}
    lr_d2_weight_arr = {}
    weight_arr = []
    task_d2_max_macro_var = {}
    task_d2_max_mean_var_num = {}

    var_weighted_sum = 0
    num_weighted_sum = 0
    for task in method_d2_task_d2_lr_d2_scores[method_name]:
        mean_score_arr = []
        var_score_arr = []
        mean_macro_score_arr = []
        var_macro_score_arr = []
        num_arr = []
        max_score_arr = []
        max_macro_score_arr = []
        for lr in method_d2_task_d2_lr_d2_scores[method_name][task]:
            # if lr == 'lr_5':
            #    continue
            macro_scores, scores, weight, method_random, random_seeds = \
            method_d2_task_d2_lr_d2_scores[method_name][task][lr]
            mean_score = np.mean(scores)
            mean_macro_score = np.mean(macro_scores)
            var_score = np.var(scores)
            var_macro_score = np.var(macro_scores)
            mean_score_arr.append(mean_score)
            mean_macro_score_arr.append(mean_macro_score)
            var_score_arr.append(var_score)
            var_macro_score_arr.append(var_macro_score)
            num_arr.append(len(scores))
            max_score_arr.append(np.max(scores))
            max_macro_score_arr.append(np.max(macro_scores))
            if lr not in lr_d2_task_score:
                lr_d2_task_score[lr] = []
                lr_d2_task_macro_score[lr] = []
                lr_d2_weight_arr[lr] = []
            lr_d2_task_score[lr].append(mean_score)
            lr_d2_task_macro_score[lr].append(mean_macro_score)
            lr_d2_weight_arr[lr].append(weight)
            print(method_name, task, lr, macro_scores, scores, random_seeds, all_seed_set - set(random_seeds),
                  "{:.3f} {:.3f}".format(mean_score, mean_macro_score))
        max_macro_score = np.max(mean_macro_score_arr)
        max_macro_score_idx = np.argmax(mean_macro_score_arr)
        # max_score= np.max(mean_score_arr)
        max_score = mean_score_arr[max_macro_score_idx]
        max_max_macro_score = np.max(max_macro_score_arr)
        max_max_score = np.max(max_score_arr)
        print(method_name, task,
              "{:.3f} {:.3f} {:.3f} {:.3f}".format(max_macro_score, max_score, max_max_macro_score, max_max_score))
        task_d2_max_macro_var[task] = [max_macro_score, var_macro_score_arr[max_macro_score_idx]]
        task_d2_max_mean_var_num[task] = [max_score, var_score_arr[max_macro_score_idx], num_arr[max_macro_score_idx]]

        var_weighted_sum += var_macro_score_arr[max_macro_score_idx] * weight * num_arr[max_macro_score_idx]
        num_weighted_sum += weight * num_arr[max_macro_score_idx]
        method_score_arr.append(max_score)
        method_macro_score_arr.append(max_macro_score)
        method_max_score_arr.append(max_max_score)
        method_max_macro_score_arr.append(max_max_macro_score)
        weight_arr.append(weight)
    # print(method_name, "{:.3f} {:.3f}".format(np.mean(method_score_arr), np.mean(method_macro_score_arr)))
    OTL_macro_score = np.average(method_macro_score_arr, weights=weight_arr)
    OTL_score = np.average(method_score_arr, weights=weight_arr)
    OTL_max_macro_score = np.average(method_max_macro_score_arr, weights=weight_arr)
    OTL_max_score = np.average(method_max_score_arr, weights=weight_arr)
    # OTL_gmean = scipy.stats.gmean(method_macro_score_arr, weights=weight_arr)
    OTL_gmean = np.exp(np.average(np.log(method_macro_score_arr), weights=weight_arr))
    # OTL_hmean = scipy.stats.hmean(method_macro_score_arr)
    # print(method_name, "{:.3f} {:.3f}".format(OTL_score, OTL_macro_score))
    max_macro = -1
    max_avg = -1
    max_lr = ''
    max_num_task = -1
    for lr in lr_d2_task_score:
        # score = np.mean(lr_d2_task_macro_score[lr])
        macro_score = np.average(lr_d2_task_macro_score[lr], weights=lr_d2_weight_arr[lr])
        score = np.average(lr_d2_task_score[lr], weights=lr_d2_weight_arr[lr])
        print(method_name, lr, "{:.3f} {:.3f}".format(score, macro_score))
        num_task = len(lr_d2_task_macro_score[lr])
        if num_task >= max_num_task and macro_score > max_macro:
            max_macro = macro_score
            max_lr = lr
            max_avg = score
            max_num_task = num_task
    for task in task_order:
        if task not in method_d2_task_d2_lr_d2_scores[method_name]:
            continue
        # for task in method_d2_task_d2_lr_d2_scores[method_name]:
        if max_lr in method_d2_task_d2_lr_d2_scores[method_name][task]:
            macro_scores, scores, weight, method_random, random_seeds = \
            method_d2_task_d2_lr_d2_scores[method_name][task][max_lr]
            print(task, "{:.3f}".format(np.mean(scores)), weight)
    print('-----------------------')
    for task in task_order:
        if task not in task_d2_max_macro_var:
            continue
        # print(task, "{:.3f}".format(task_d2_max_macro_var[task][0]))
        print(task, "{:.3f} {:.3f} {:d}".format(task_d2_max_mean_var_num[task][0], math.sqrt(
            task_d2_max_mean_var_num[task][1] / task_d2_max_mean_var_num[task][2]), task_d2_max_mean_var_num[task][2]))

    print('macro_avg', max_macro)
    print('simple_avg', max_avg)
    print('max_lr', max_lr)
    print('OTL_macro_score', OTL_macro_score,
          '+-{}'.format(math.sqrt(var_weighted_sum / num_weighted_sum / num_weighted_sum)), num_weighted_sum)
    print('OTL_score', OTL_score)
    print('OTL_max_macro_score', OTL_max_macro_score)
    print('OTL_max_score', OTL_max_score)
    print('OTL_gmean_score', OTL_gmean)
    # print('OTL_hmean_score', OTL_hmean)
