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
#

"""Pretrain BERT"""

from comet_ml import Experiment
from memory_profiler import profile

#from apex import amp
import os
import random
import numpy as np
import psutil
import torch

from olfmlm.arguments import get_args
from olfmlm.configure_data import configure_data
from olfmlm.learning_rates import AnnealingLR
from olfmlm.model import BertModel
from olfmlm.model import get_params_for_weight_decay_optimization
from olfmlm.model import DistributedDataParallel as DDP

from olfmlm.optim import Adam
from olfmlm.utils import Timers
from olfmlm.utils import save_checkpoint
from olfmlm.utils import load_checkpoint
#from olfmlm.find_neighbors import Vocab_finder

#finder =Vocab_finder()
def check_vocab(model,tokenizer,num_facet):
    '''
    Check the nearest neighbors during training
    '''
    model.eval()
    word_embed=model.model.bert.embeddings.word_embeddings.weight.cpu().detach().numpy()
    finder.build_faiss(word_embed)
    for _ in range(num_facet):
        choose_id = random.randint(0, finder.total_examples - 1)
        view_list = finder.get_facet(model, tokenizer,choose_id)
        finder.query(view_list,choose_id,tokenizer)
        print('\n')

    model.train()


global_weight=None
batch_step = 0



def get_model(tokenizer, args):
    """Build the model."""

    print('building BERT model ...')
    model = BertModel(tokenizer, args)
    print(' > number of parameters: {}'.format(
        sum([p.nelement() for p in model.parameters()])), flush=True)

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Wrap model for distributed training.
    if args.world_size > 1:
        model = DDP(model)

    return model

def get_optimizer(model, args):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, DDP):
        model = model.module

    param_groups = model.get_params()

    # Use Adam.
    optimizer = Adam(param_groups,
                     lr=args.lr, weight_decay=args.weight_decay)

    return optimizer





def get_learning_rate_scheduler(optimizer, args):
    """Build the learning rate scheduler."""

    # Add linear learning rate scheduler.
    if args.lr_decay_iters is not None:
        num_iters = args.lr_decay_iters
    else:
        num_iters = args.train_tokens * args.epochs
    init_step = -1
    warmup_iter = args.warmup * num_iters
    lr_scheduler = AnnealingLR(optimizer,
                               start_lr=args.lr,
                               warmup_iter=warmup_iter,
                               num_iters=num_iters,
                               decay_style=args.lr_decay_style,
                               last_iter=init_step)

    return lr_scheduler


def setup_model_and_optimizer(args, tokenizer):
    """Setup model and optimizer."""

    model = get_model(tokenizer, args)
    optimizer = get_optimizer(model, args)
    lr_scheduler = get_learning_rate_scheduler(optimizer, args)
    
    criterion_cls = torch.nn.CrossEntropyLoss(reduce=False, ignore_index=-1)
    criterion_reg = torch.nn.MSELoss(reduce=False)
    criterion_nll = torch.nn.NLLLoss(reduce=False, ignore_index=-1)

    criterion = (criterion_cls, criterion_reg, criterion_nll)

    if args.load is not None:
        args.epoch = load_checkpoint(model, optimizer, lr_scheduler, args)
        args.resume_dataloader = True
    return model, optimizer, lr_scheduler, criterion


def get_batch(data,args):
    """ Get a batch of data from the data loader, which automatically batches the individual examples
        Concatenates necessary data (lm_labels, loss_mask, tgs_mask), which is required for FS/QT variant tasks
        Puts data into tensors, and places them onto CUDA
    """
    # TODO Add trigram mask
    aux_labels = {}
    for mode, label in data['aux_labels'].items():
        if args.use_hard_neg:
            assert label.shape[1] == 3
            label = torch.cat([label[:, 0], label[:, 1], label[:, 2]])
        elif label.shape[1] == 2:
            label = torch.cat([label[:, 0], label[:, 1]])
        else:
            label = label.squeeze()
        aux_labels[mode] = torch.autograd.Variable(label.long()).cuda()
    num_sentences = data['n']
    num_tokens = torch.tensor(sum(data['num_tokens']).item()).long().cuda()
    tokens = []
    types = []
    tasks = []
    loss_mask = []
    tgs_mask = []
    lm_labels = []
    att_mask = []
    for i in range(min(num_sentences)):
        suffix = "_" + str(i)
        tokens.append(torch.autograd.Variable(data['text' + suffix].long()).cuda())
        types.append(torch.autograd.Variable(data['types' + suffix].long()).cuda())
        tasks.append(torch.autograd.Variable(data['task' + suffix].long()).cuda())
        att_mask.append(1 - torch.autograd.Variable(data['pad_mask' + suffix].byte()).cuda())
        lm_labels.append((data['mask_labels' + suffix]).long())
        loss_mask.append((data['mask' + suffix]).float())
        tgs_mask.append((data['tgs_mask' + suffix]).float())

    lm_labels = torch.autograd.Variable(torch.cat(lm_labels, dim=0).long()).cuda()
    loss_mask = torch.autograd.Variable(torch.cat(loss_mask, dim=0).float()).cuda()
    tgs_mask = torch.autograd.Variable(torch.cat(tgs_mask, dim=0).float()).cuda()

    return (tokens, types, tasks, aux_labels, loss_mask, tgs_mask, lm_labels, att_mask, num_tokens)


def forward_step(data, model, criterion, modes, args):
    """Forward step."""
    criterion_cls, criterion_reg, criterion_nll = criterion
    # Get the batch.
    batch = get_batch(data, args)
    tokens, types, tasks, aux_labels, loss_mask, tgs_mask, lm_labels, att_mask, num_tokens = batch
    # Create self-supervised labels which required batch size
    
    if "rg" in modes:
        aux_labels['rg'] = torch.autograd.Variable(torch.arange(tokens[0].shape[0]).long()).cuda()
    if "mf" in modes:
        aux_labels['mf']=torch.autograd.Variable(torch.arange(tokens[0].shape[0]).long()).cuda()
    if "fs" in modes:
        aux_labels['fs'] = torch.autograd.Variable(torch.ones(tokens[0].shape[0] * 2 * args.seq_length).long()).cuda()
    # Forward model.
    #print("tfidf", aux_labels["tf_idf"])
    #print("tfidf", aux_labels["tf_idf"].size())
    #print("so", aux_labels["so"].size())
    #print("input", tokens)
    scores = model(modes, tokens, types, tasks, att_mask, checkpoint_activations=args.checkpoint_activations)
    assert sorted(list(scores.keys())) == sorted(modes)

    # Calculate losses based on required criterion
    losses = {}
    for mode, score in scores.items():
        if mode in ["mlm", "sbo"]:
            mlm_loss = criterion_cls(score.view(-1, args.data_size).contiguous().float(),
                                     lm_labels.view(-1).contiguous())
            loss_mask = loss_mask.view(-1).contiguous()
            losses[mode] = torch.sum(mlm_loss * loss_mask.view(-1).float()) / loss_mask.sum()
        elif mode == "tgs":
            tgs_loss = criterion_cls(score.view(-1, 6).contiguous().float(),
                                     aux_labels[mode].view(-1).contiguous())
            tgs_loss = tgs_loss.view(-1).contiguous()
            losses[mode] = torch.sum(tgs_loss * tgs_mask.view(-1).float() / tgs_mask.sum())
        elif mode in ["fs", "wlen", "tf", "tf_idf"]: # use regression
            #if args.use_hard_neg:
            #    losses[mode] = criterion_reg(score.view(-1).contiguous().float(),
            #                             aux_labels[mode].transpose(0,1).reshape(-1).contiguous().float()).mean() #hacky, I really cannot find where the axis got switched if we do not use hard negative
            #else:
            losses[mode] = criterion_reg(score.view(-1).contiguous().float(),
                                         aux_labels[mode].view(-1).contiguous().float()).mean()

        elif mode=="mf":
            '''
            loss function related to mf task
            '''
            loss_left, loss_right, score_all, loss_autoenc = score
            losses[mode] = (loss_left + loss_right)/2

            if args.facet2facet or args.facet2facet_mr:
                loss_facet = 0
                for i in range(len(score_all)):
                    score_tmp = score_all[i]#*model.model.neg_w2
                    loss_facet += criterion_cls(score_tmp.contiguous().float(),
                                           aux_labels[mode].view(-1).contiguous()).mean()
                #reduce the weight of f2f to balance two variances?
                #losses[mode] += 0.5 * loss_facet
                #losses[mode] += 1.5 * loss_facet
                losses[mode] += loss_facet

            if args.autoenc_reg_const>0:
                losses[mode] += loss_autoenc

            #print(losses[mode])

        else:
            score = score.view(-1, 2) if mode in ["tc", "cap"] else score
            #if args.use_hard_neg:
            #    losses[mode] = criterion_cls(score.contiguous().float(),
            #                             aux_labels[mode].transpose(0,1).reshape(-1).contiguous()).mean()
            #else:
            losses[mode] = criterion_cls(score.contiguous().float(),
                                         aux_labels[mode].view(-1).contiguous()).mean()
    return losses, num_tokens


def backward_step(optimizer, model, losses, num_tokens, args):
    """Backward step."""
    # Backward pass.
    optimizer.zero_grad()
    # For testing purposes, should always be False
    if args.no_aux:
        total_loss = losses['mlm']
    else:
        total_loss = sum(losses.values())
    #total_loss = total_loss/2
    total_loss.backward()


    # Reduce across processes.
    losses_reduced = losses
    if args.world_size > 1:
        losses_reduced = [[k,v] for k,v in losses_reduced.items()]
        reduced_losses = torch.cat([x[1].view(1) for x in losses_reduced])
        torch.distributed.all_reduce(reduced_losses.data)
        torch.distributed.all_reduce(num_tokens)
        reduced_losses.data = reduced_losses.data / args.world_size
        model.allreduce_params(reduce_after=False,
                               fp32_allreduce=False)#args.fp32_allreduce)
        losses_reduced = {losses_reduced[i][0]: reduced_losses[i] for i in range(len(losses_reduced))}
    # Clipping gradients helps prevent the exploding gradient.
    if args.clip_grad > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

    return losses_reduced, num_tokens


def train_step(input_data, model, criterion, optimizer, lr_scheduler, modes, args):
    """Single training step."""
    # Forward model for one step.
    losses, num_tokens = forward_step(input_data, model, criterion, modes, args)
    # Calculate gradients, reduce across processes, and clip.
    losses_reduced, num_tokens = backward_step(optimizer, model, losses, num_tokens, args)
    '''
    global batch_step
    batch_step = batch_step+1
    if batch_step % 2 ==0:
        optimizer.step()
        optimizer.zero_grad()
    '''
    # Update parameters.
    optimizer.step()
    return losses_reduced, num_tokens

def get_stage_info(total_tokens, num_tasks):
    """
    Get number of tokens for each task during each stage. Based on ERNIE 2.0's continual multi-task learning
    Number of stages is equal to the number of tasks (each stage is larger than the previous one)
    :param total_tokens: total number of tokens to train on
    :param num_tasks: number of tasks
    :return: Number of tokens for each task at each stage
    """
    tokens_per_task = total_tokens / num_tasks
    tokens_subunit = tokens_per_task / (num_tasks + 1)
    tokens_per_task_per_stage = []
    for i in range(num_tasks):
        stage_tokens = []
        for j in range(num_tasks):
            if i < j:
                stage_tokens.append(0)
            elif i > j:
                stage_tokens.append(tokens_subunit)
            else:
                stage_tokens.append(tokens_subunit * (i + 2))
        tokens_per_task_per_stage.append(stage_tokens)

    return tokens_per_task_per_stage

def set_up_stages(args):
    """
    Set up stage information and functions to use for ERNIE 2.0's continual multi-task learning
    Closure that returns a function that will return next stages token requirements as requested
    :param args: arguments
    :return: a function that will return next stages token requirements as requested
    """
    assert not args.incremental
    total_tokens = args.epochs * args.train_tokens
    modes = args.modes.split(',')
    if args.always_mlm:
        modes = modes[1:]
    stage_splits = get_stage_info(total_tokens, len(modes))
    stage_idx = 0

    def next_stage():
        nonlocal stage_idx
        if stage_idx >= len(stage_splits):
            print("Finished all training, shouldn't reach this unless it's the very final iteration")
            return {k: float(total_tokens) for k in modes}
        assert len(modes) == len(stage_splits[stage_idx])
        current_stage = {k: v for k, v in zip(modes, stage_splits[stage_idx])}
        print("Starting stage {} of {}, with task distribution: ".format(stage_idx, len(stage_splits)))
        print(current_stage)
        stage_idx += 1
        return current_stage

    return next_stage

def get_mode_from_stage(current_stage, args):
    """
    Get the mode to use given the current stage
    :param current_stage: number of tokens left for each task for this stage
    :param args: arguments
    :return: selected mode
    """
    modes = args.modes.split(',')
    if args.always_mlm:
        modes = modes[1:]
    p = np.array([current_stage[m] for m in modes])
    p /= np.sum(p)
    return [np.random.choice(modes, p=p)]

def train_epoch(epoch, model, optimizer, train_data, lr_scheduler, criterion, timers, experiment, metrics, args,
                current_stage=None, next_stage=None, val_data=None, tz=None):
    """Train one full epoch."""
    print("Starting training of epoch {}".format(epoch), flush=True)
    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_losses = {}

    # Iterations.
    max_tokens = args.train_tokens
    log_tokens = 0
    tot_tokens = args.tot_tokens
    iteration = 0
    tot_iteration = 0

    threshold=0
    # Data iterator.
    modes = args.modes.split(',')
    # Incrementally add tasks after each epoch
    if args.incremental:
        modes = modes[:epoch]

    train_data.dataset.set_args(modes)
    sent_tasks = [m for m in modes if m in train_data.dataset.sentence_tasks]
    tok_tasks = [m for m in modes if m not in train_data.dataset.sentence_tasks + ["mlm"]]

    data_iters = iter(train_data)

    timers('interval time').start()
    timers('checkpoint time').start()
    while tot_tokens < max_tokens:
        # ERNIE 2.0's continual multi task learning
        if args.continual_learning:
            # Continual learn through all tasks
            modes_ = get_mode_from_stage(current_stage, args)
            if args.always_mlm:
                # Continual learn through auxiliary tasks, always learning MLM
                modes_ = ['mlm'] + modes_
        # Alternating between tasks
        elif args.alternating:
            if args.always_mlm:
                # Alternate between all tasks
                modes_ = ['mlm']
                if len(modes[1:]) > 0:
                    # Alternate between auxiliary tasks, always learning MLM
                    modes_ += [modes[(iteration % (len(modes) - 1)) + 1]]
            else:
                modes_ = [modes[iteration % len(modes)]]
        # Summing all tasks
        else:
            sent_task = [] if len(sent_tasks) == 0 else [sent_tasks[iteration % len(sent_tasks)]]
            modes_ = ['mlm'] + sent_task + tok_tasks

        while True:
            try:
                losses, num_tokens = train_step(next(data_iters),
                              model,
                              criterion,
                              optimizer,
                              lr_scheduler,
                              modes_,
                              args)
                break
            except StopIteration:
                data_iters = iter(train_data)

        log_tokens += num_tokens.item()
        tot_tokens += num_tokens.item()
        if args.continual_learning:
            for m in modes_:
                if args.always_mlm and m == "mlm":
                    continue
                current_stage[m] = max(0, current_stage[m] - num_tokens.item())

            if sum(current_stage.values()) == 0:
                ns = next_stage()
                for m in ns:
                    current_stage[m] = ns[m]

        # Update learning rate.
        lr_scheduler.step(step_num=(epoch-1) * max_tokens + tot_tokens)
        iteration += 1
        # Update losses.
        for mode, loss in losses.items():
            total_losses[mode] = total_losses.get(mode, 0.0) + loss.data.detach().float()
        #print(total_losses)
        
        if args.save_iters and tot_iteration and tot_iteration%args.save_iters==0:
            ck_path = 'ck/model_{}_{}.pt'.format(epoch,tot_iteration)
            print('saving ck model to:',os.path.join(args.save, ck_path))
            save_checkpoint(ck_path, epoch+1, model, optimizer, lr_scheduler, args)
            #elapsed_time = timers('checkpoint time').elapsed()
            val_loss = evaluate(epoch, val_data, model, criterion, timers('checkpoint time').elapsed() , args)
            global best_val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if args.save:
                    best_path = 'best/model.pt'
                    print('saving best model to:',os.path.join(args.save, best_path))
                    save_checkpoint(best_path, epoch+1, model, optimizer, lr_scheduler, args)
           
        # Logging.
        if log_tokens > args.log_interval:
            # print(model.model.bert.pooler.dense.weight)
            # print(model.model.sent.mf.v_1.dense.weight)
            # print(model.model.sent.mf.v_2.dense.weight)
            # print(model.model.sent.mf.v_3.dense.weight)
            # print(model.model.extra_token)
            threshold +=1
            if threshold==args.alter_point:
                model.model.extra_token += '-mr'
                print("switch to " + model.model.extra_token)
                
            # if args.extra_token in ['vocab','vocab-mr']:
            #     check_vocab(model, tz)
            log_tokens = 0
            learning_rate = optimizer.param_groups[0]['lr']
            avg_loss = {}
            for mode, v in total_losses.items():
                avg_loss[mode] = v.item() / iteration

            elapsed_time = timers('interval time').elapsed()
            log_string = ' epoch{:2d} |'.format(epoch)
            log_string += ' tokens {:8d}/{:8d} |'.format(tot_tokens, max_tokens)
            log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
                elapsed_time * 1000.0 / iteration)
            log_string += ' learning rate {:.3E} |'.format(learning_rate)
            for mode, v in avg_loss.items():
                log_string += ' {} loss {:.3E} |'.format(mode, v)

            print(log_string, flush=True)
            #print(iteration)
            total_losses = {}
            experiment.set_step((epoch - 1) * max_tokens + tot_tokens)
            metrics['learning_rate'] = learning_rate
            for mode, v in avg_loss.items():
                metrics[mode] = v
            if "mf" in modes:
                model.model.normalize_facet_agg_stats()
                print('Facet stats:', model.model.facet_agg_stats)
                #metrics['var_before_pool']=model.model.embedding_var_before_pool
                #metrics['var_after_pool']=model.model.embedding_var_after_pool
                metrics['var_before_trans']=model.model.emedding_var_before_trans
                metrics['var_after_trans']=model.model.emedding_var_after_trans
                metrics['emedding_var_across'] = model.model.emedding_var_across

                cor_index=0
                for i in range(1,args.num_facets+1):
                    for j in range(i+1,args.num_facets+1):
                        metrics['cor_{}_{}'.format(i,j)] = model.model.corr_list[cor_index]
                        cor_index += 1
                def sketch_agg_stats(stats_name):
                    if stats_name in model.model.facet_agg_stats:
                        metrics[stats_name+'_max'] = torch.amax(model.model.facet_agg_stats[stats_name])
                        metrics[stats_name+'_min'] = torch.amin(model.model.facet_agg_stats[stats_name])
                    else:
                        metrics[stats_name+'_max'] = 1.1
                        metrics[stats_name+'_min'] = -0.1
                sketch_agg_stats('vocab')
                sketch_agg_stats('vocab-self')
                sketch_agg_stats('token')
                sketch_agg_stats('f2f')
                #experiment.log_curve(name='softmax_weight:', x=[1,2,3,4,5,6],y=global_weight.detach().cpu().numpy().tolist(), overwrite=True)
                model.model.reset_facet_agg_stats()
            experiment.log_metrics(metrics)
            #tot_iteration += iteration
            iteration = 0
        tot_iteration=tot_iteration+1

    print("Learnt using {} tokens over {} iterations this epoch".format(tot_tokens, tot_iteration))

def evaluate(epoch, data_source, model, criterion, elapsed_time, args, test=False):
    """Evaluation."""
    print("Entering evaluation", flush=True)
    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_losses = {}
    max_tokens = args.eval_tokens
    tokens = 0
    modes = args.modes.split(',')
    data_source.dataset.set_args(modes)
    data_iters = iter(data_source)
    with torch.no_grad():
        iteration = 0
        while tokens < max_tokens:
            # Forward evaluation.
            
            while True:
                try:
                    losses, num_tokens = forward_step(next(data_iters), model, criterion, modes, args)
                    break
                except (TypeError, RuntimeError) as e:
                    print("Ooops, caught: '{}', continuing".format(e))
                except StopIteration:
                    data_iters = iter(data_source)

            # Reduce across processes.
            if isinstance(model, DDP):
                losses_reduced = [[k, v] for k, v in losses.items()]
                reduced_losses = torch.cat([x[1].view(1) for x in losses_reduced])
                torch.distributed.all_reduce(reduced_losses.data)
                reduced_losses.data = reduced_losses.data / args.world_size
                torch.distributed.all_reduce(num_tokens)
                losses = {losses_reduced[i][0]: reduced_losses[i] for i in range(len(losses_reduced))}
            
            assert sorted(list(losses.keys())) == sorted(modes)
            for mode, loss in losses.items():
                total_losses[mode] = total_losses.get(mode, 0.0) + loss.data.detach().float().item()
            iteration += 1
            tokens += num_tokens.item()

    print("Evaluated using {} tokens over {} iterations.".format(tokens, iteration), flush=True)

    # Move model back to the train mode.
    model.train()

    avg_loss = {}
    for mode, v in total_losses.items():
        avg_loss[mode] = v / args.eval_iters

    tot_loss = sum(avg_loss.values())
    sep_char = '=' if test else '-'
    print(sep_char * 100)
    log_string = '| End of training | '.format(epoch) if test else '| End of epoch {:3d} | '
    log_string += 'time: {:5.2f}s | valid loss {:.4E} | '.format(epoch, elapsed_time, tot_loss)
    for mode, v in avg_loss.items():
        log_string += ' {} loss {:.3E} |'.format(mode, v)
    print(log_string, flush=True)
    print(sep_char * 100, flush=True)

    return tot_loss


def initialize_distributed(args):
    """Initialize torch.distributed."""

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)
    # Call the init process
    if args.world_size > 1:
        init_method = 'tcp://'
        master_ip = os.getenv('MASTER_ADDR', 'localhost')
        master_port = os.getenv('MASTER_PORT', '6000')
        init_method += master_ip + ':' + master_port
        torch.distributed.init_process_group(
            backend=args.distributed_backend,
            world_size=args.world_size, rank=args.rank,
            init_method=init_method)


def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

def save_extra_config(args):
    dir_name = os.path.dirname(args.save)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    with open(dir_name+'/training.conf','w') as f_out:
        f_out.write("num_facet = {}\n".format(args.num_facets))
        f_out.write('diversify_hidden_layer = "{}"\n'.format(args.diversify_hidden_layer))
        f_out.write('diversify_mode = "{}"\n'.format(args.diversify_mode))
        f_out.write('facet_mode = "{}"\n'.format(args.facet_mode))
        f_out.write('unnorm_facet = "{}"\n'.format(args.unnorm_facet))
        f_out.write('//slurm id = {}\n'.format(os.environ['SLURM_JOB_ID']))
        f_out.write('//slurm node = {}\n'.format(os.environ['SLURM_JOB_NODELIST']))


def main():
    """Main training program."""

    print('Pretrain BERT model')
    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Timer.
    timers = Timers()

    # Arguments.
    args = get_args()

    experiment=None


    experiment = Experiment(api_key='yourkey',
                             project_name='multi-facet-sent', workspace="youraccount",
                             auto_param_logging=False, auto_metric_logging=False,
                             disabled=(not args.track_results))
    experiment.log_parameters(vars(args))

    metrics = {}

    # Pytorch distributed.
    initialize_distributed(args)
    save_extra_config(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    # Data stuff.
    data_config = configure_data()
    data_config.set_defaults(data_set_type='BERT', transpose=False)
    (train_data, val_data, test_data), tokenizer = data_config.apply(args)
    args.data_size = tokenizer.num_tokens


    # Model, optimizer, and learning rate.
    model, optimizer, lr_scheduler, criterion = setup_model_and_optimizer(
        args, tokenizer)

    #model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    timers("total time").start()
    epoch = 0
    
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        start_epoch = 1
        best_val_loss = float('inf')
        # Resume data loader if necessary.
        #if args.resume_dataloader: #Not sure why, but the stored epoch number is not correct, so I comment this in order to continue training a failed model
        #    start_epoch = args.epoch 
        next_stage = None
        current_stage = None
        if args.continual_learning:
            next_stage = set_up_stages(args)
            current_stage = next_stage()
            if args.resume_dataloader:
                num_tokens = args.epoch * args.train_tokens
                # Get to the right stage
                while num_tokens > sum(current_stage.values()):
                    num_tokens -= sum(current_stage.values())
                    ns = next_stage()
                    for m in ns:
                        current_stage[m] = ns[m]
                # Get to right part of stage
                stage_tokens = sum(current_stage.values())
                stage_ratios = {k: v / float(stage_tokens) for k, v in current_stage.items()}
                for k in current_stage:
                    current_stage[k] -= num_tokens * stage_ratios[k]

        # Train for required epochs
        #print(start_epoch)
        #print(args.epochs+1)
        for epoch in range(start_epoch, args.epochs+1):
            if args.shuffle:
                train_data.batch_sampler.sampler.set_epoch(epoch+args.seed)
            timers('epoch time').start()

            # Train
            train_epoch(epoch, model, optimizer, train_data, lr_scheduler, criterion, timers, experiment, metrics, args,
                        current_stage=current_stage, next_stage=next_stage, val_data=val_data, tz=tokenizer)
            
            elapsed_time = timers('epoch time').elapsed()
             
            if args.save:
                ck_path = 'ck/model_{}.pt'.format(epoch)
                print('saving ck model to:',
                       os.path.join(args.save, ck_path))
                save_checkpoint(ck_path, epoch+1, model, optimizer, lr_scheduler, args)

            # Validate
            
            val_loss = evaluate(epoch, val_data, model, criterion, elapsed_time, args)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if args.save:
                    best_path = 'best/model.pt'
                    print('saving best model to:',
                           os.path.join(args.save, best_path))
                    save_checkpoint(best_path, epoch+1, model, optimizer, lr_scheduler, args)
            

    except KeyboardInterrupt:
        print('-' * 100)
        print('Exiting from training early')
        exit()

    if test_data is not None:
        # Run on test data.
        print('entering test')
        elapsed_time = timers("total time").elapsed()
        evaluate(epoch, test_data, model, criterion, elapsed_time, args, test=True)

import tracemalloc
tracemalloc.start(10)
import signal
import sys
import traceback
#from guppy import hpy
#import objgraph

def term_handler(signum, frame):
    print("TERM signal handler called.  Exiting.")
    first_size, first_peak = tracemalloc.get_traced_memory()
    print(f"{first_size=}, {first_peak=}")
    objgraph.show_most_common_types(limit=20)
    h = hpy()
    print(h.heap())
    #exc_type, exc_obj, exc_tb = sys.exc_info()
    #print(exc_tb)
    #print(exc_tb.tb_lineno)
    print(traceback.format_exc())
    snapshot = tracemalloc.take_snapshot()    
    #tracemalloc.top_n(25, snapshot, trace_type='filename')
    top_stats = snapshot.statistics('lineno')

    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)
    sys.stdout.flush()    
        

if __name__ == "__main__":
    best_val_loss = float('inf')
    #signal.signal(signal.SIGUSR1, term_handler)
    #signal.signal(signal.SIGCONT, term_handler)
    #signal.signal(signal.SIGTERM, term_handler)
    #signal.signal(signal.SIGINT, term_handler)
    #main()
    try:
        main()
    except:
        print("exception is called.  Exiting.") 
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(exc_tb)
        #print(exc_tb.tb_lineno)
        print(traceback.format_exc())
    #    snapshot = tracemalloc.take_snapshot()    
    #    top_stats = snapshot.statistics('lineno')

    #    print("[ Top 10 ]")
    #    for stat in top_stats[:10]:
    #        print(stat)
    #    sys.stdout.flush()    
        #snapshot = tracemalloc.take_snapshot()    
        #top_n(25, snapshot, trace_type='filename')
