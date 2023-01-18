import torch
import torch.nn as nn
from ...model.new_models import BertPoolerforview, BertHeadTransform
from ...model.modeling import BertConfig
import sys
import logging as log
import random

class NullPhraseLayer(nn.Module):
    """ Dummy phrase layer that does nothing. Exists solely for API compatibility. """

    def __init__(self, input_dim: int):
        super(NullPhraseLayer, self).__init__()
        self.input_dim = input_dim

    def get_input_dim(self):
        return self.input_dim

    def get_output_dim(self):
        return 0

    def forward(self, embs, mask):
        return None

def get_num_facet(previous_state):
    target_prefix = 'sent_encoder._text_field_embedder.model.mf.s_'
    target_suffix = '.dense.weight'
    facet_list = []
    for param_name in previous_state.keys():
        if target_prefix in param_name and target_suffix in param_name:
            facet_idx = int(param_name[len(target_prefix):-len(target_suffix)])
            facet_list.append(facet_idx)

    assert len(facet_list) > 0
    return max(facet_list)

extracted_grads_word=[]
def extract_grad_hook(module, grad_in, grad_out):
    #log.info(grad_out)
    extracted_grads_word.append(grad_out[0])

class Pooler(nn.Module):
    """ Do pooling, possibly with a projection beforehand """

    def print_facet_norm_avg(self):
        if not self.vis_facet_norm:
            return
        if self.pass_count > 0:
            facet_norm_mean = self.facet_norm_sum / self.pass_count
            facet_div_mean = self.facet_div_sum / self.pass_count
            facet_var_batch_mean = self.facet_var_batch_sum / self.pass_count
            log.info("facet norm, facet div, facet var batch: ")
            log.info(facet_norm_mean)
            log.info(facet_div_mean)
            log.info(facet_var_batch_mean)
        self.init_facet_norm()
        self.facet_norm_act = False
        if self.record_grad:
            self.handle.remove()
        self.record_grad = False

    def init_facet_norm(self, record_grad = False):
        if self.vis_facet_norm:
            self.facet_norm_sum = torch.zeros(self.num_facet, device = self.trans.weight.data.device)
            self.pass_count = 0
            self.facet_div_sum = 0
            self.facet_var_batch_sum = torch.zeros(self.num_facet, device = self.trans.weight.data.device)
        self.facet_norm_act = True
        self.record_grad = record_grad
        if record_grad:
            self.handle = self.embedder_for_grad.model.embeddings.word_embeddings.register_backward_hook(extract_grad_hook)
            #self.embedder_for_grad.model.embeddings.word_embeddings.register_full_backward_hook(extract_grad_hook)
            

    def __init__(self, num_facet, project=True, d_inp=512, d_proj=512, pool_type="max", previous_state_path=None, unnorm_facet=None, analyze_grad = False, embedder_for_grad = None, pooler_dropout = 0, pooler_scalar_dropout = 0):
        super(Pooler, self).__init__()
        self.project = nn.Linear(d_inp, d_proj) if project else lambda x: x
        if pool_type == "single_first":
            self.trans = nn.Linear(d_inp, d_inp)
        else:
            self.trans = nn.Linear(d_inp*num_facet, d_inp)
            for i in range(num_facet):
                self.trans.weight.data[:,i*d_inp:(i+1)*d_inp] = 1/float(num_facet) * torch.eye(d_inp)
        
        self.d_inp = d_inp

        self.facet_emb_arr = None
        self.vis_facet_norm = True
        self.facet_norm_act = False
        self.analyze_grad = analyze_grad
        self.pooler_dropout = pooler_dropout
        if pooler_dropout > 0:
            assert pool_type == 'first' or pool_type == 'first_init' or pool_type == 'first_init_1' or pool_type == 'first_init_avg'
        #self.drop = nn.Dropout(0.3)
        self.drop = nn.Dropout(pooler_scalar_dropout)
        
        self.trans.bias.data[:] = 0
        self.weight_global = nn.Parameter( torch.ones(num_facet) / float(num_facet) )
        #self.weight_global = nn.Parameter( torch.rand(3) )
        self.num_facet = num_facet
        self.pool_type = pool_type
        self.embedder_for_grad = embedder_for_grad
        if previous_state_path is not None:
        #if False:
            #print(previous_state_path)
            previous_state = torch.load(previous_state_path, map_location='cpu')
            #print(previous_state)
            #sys.stdout.flush()
            #previous_state = previous_state['sd']
            #previous_state = torch.load(previous_state_path, map_location='cpu')['sd']
            self.unnorm_facet = unnorm_facet
            self.sent = torch.nn.ModuleDict()
            self.num_out_head = get_num_facet(previous_state)
            #for i in range(num_facet):
            if 'sent_encoder._text_field_embedder.model.mf.s_1.dense.bias' in previous_state:
                use_proj_bias = True
            else:
                use_proj_bias = False
            for i in range(self.num_out_head):
                if self.pool_type[:5] == "head_":
                    config = BertConfig('olfmlm/bert_config.json')
                    self.sent['s_'+str(i+1)] = BertHeadTransform(config, use_perturbation=False, use_bias=use_proj_bias)
                    #self.sent['v_'+str(i+1)] = BertPoolerforview(config)
                    self.sent['s_'+str(i+1)].dense.weight.data = previous_state['sent_encoder._text_field_embedder.model.mf.s_'+str(i+1)+'.dense.weight']
                    if use_proj_bias:
                        self.sent['s_'+str(i+1)].dense.bias.data = previous_state['sent_encoder._text_field_embedder.model.mf.s_'+str(i+1)+'.dense.bias']
                    self.sent['s_'+str(i+1)].LayerNorm.weight.data = previous_state['sent_encoder._text_field_embedder.model.mf.s_'+str(i+1)+'.LayerNorm.weight']
                    self.sent['s_'+str(i+1)].LayerNorm.bias.data = previous_state['sent_encoder._text_field_embedder.model.mf.s_'+str(i+1)+'.LayerNorm.bias']
                elif self.pool_type[:9] == 'proj_init' or self.pool_type == 'proj_avg_train' or self.pool_type == 'proj_avg_train_skip':
                    self.sent['s_'+str(i+1)] = nn.Linear(d_inp, d_inp)
                    self.sent['s_'+str(i+1)].weight.data = previous_state['sent_encoder._text_field_embedder.model.mf.s_'+str(i+1)+'.dense.weight']
                    if use_proj_bias:
                        self.sent['s_'+str(i+1)].bias.data = previous_state['sent_encoder._text_field_embedder.model.mf.s_'+str(i+1)+'.dense.bias']
                    else:
                        self.sent['s_'+str(i+1)].bias.data[:] = 0
            if self.pool_type == 'first_init' or (self.pool_type == 'first_init_avg' and num_facet == 1):
                for i in range(num_facet):
                    self.trans.weight.data[:,i*d_inp:(i+1)*d_inp] = previous_state['sent_encoder._text_field_embedder.model.mf.s_'+str(min(i+1,self.num_out_head))+'.dense.weight']
            elif self.pool_type == 'first_init_1':
                for i in range(num_facet):
                    self.trans.weight.data[:,i*d_inp:(i+1)*d_inp] = previous_state['sent_encoder._text_field_embedder.model.mf.s_'+str(min(1,self.num_out_head))+'.dense.weight']
            elif self.pool_type == 'first_init_avg':
                weight_avg = 0
                for i in range(num_facet):
                    self.trans.weight.data[:,i*d_inp:(i+1)*d_inp] = previous_state['sent_encoder._text_field_embedder.model.mf.s_'+str(min(i+1,self.num_out_head))+'.dense.weight']
                    weight_avg += self.trans.weight.data[:,i*d_inp:(i+1)*d_inp]
                weight_avg = weight_avg / num_facet
                for i in range(num_facet):
                    self.trans.weight.data[:,i*d_inp:(i+1)*d_inp] = self.trans.weight.data[:,i*d_inp:(i+1)*d_inp] - weight_avg
            elif self.pool_type == "single_first":
                self.trans.weight.data = previous_state['sent_encoder._text_field_embedder.model.mf.s_'+str(min(1,self.num_out_head))+'.dense.weight']
                #self.trans.weight.data = self.trans.weight.data
                #self.sent['v_'+str(i+1)].dense.weight.data = previous_state['sent_encoder._text_field_embedder.model.mf.v_'+str(i+1)+'.dense.weight']
                #self.sent['v_'+str(i+1)].dense.bias.data = previous_state['sent_encoder._text_field_embedder.model.mf.v_'+str(i+1)+'.dense.bias']
                #for param in self.sent['s_'+str(i+1)].parameters():
                #     param.requires_grad = False
                #for param in self.sent['v_'+str(i+1)].parameters():
                #     param.requires_grad = False
    
    def var_of_embedding(self, inputs, dim):
        #(batch, facet, embedding)
        with torch.no_grad():
            inputs_norm = inputs/inputs.norm(dim=-1, keepdim=True)
            pred_mean = inputs_norm.mean(dim=dim, keepdim=True)
            loss_set_div = (inputs_norm - pred_mean).norm(dim=2)
        return loss_set_div

    def compute_facet_norm(self, proj_seq):
        facet_norm = torch.zeros(self.num_facet, device = self.trans.weight.data.device)
        facet_arr = []
        if self.pool_type == "proj_avg_train" or self.pool_type == "proj_avg_train_skip":
            avg_w = 0
            if self.num_facet > 1:
                for i in range(1,(self.num_facet+1)):
                    avg_w = avg_w + self.sent['s_'+str(i)].weight.data
                avg_w = avg_w / self.num_facet
                
        for i in range(self.num_facet):
            if self.pool_type == "facet_mean" or self.pool_type == "lin":
                facet_emb = proj_seq[:, 1+i]
            elif self.pool_type == "proj_avg_train":
                facet_emb= nn.functional.linear(proj_seq[:, 1+i], self.sent['s_'+str(i+1)].weight.data - avg_w, self.sent['s_'+str(i+1)].bias.data)
            elif self.pool_type == "proj_avg_train_skip":
                if i == 0:
                    facet_emb= nn.functional.linear(proj_seq[:, 1+i], self.sent['s_'+str(i+1)].weight.data, self.sent['s_'+str(i+1)].bias.data)
                else:
                    facet_emb= nn.functional.linear(proj_seq[:, 1+i], self.sent['s_'+str(i+1)].weight.data - avg_w, self.sent['s_'+str(i+1)].bias.data)

            elif self.pool_type == "single_first":
                facet_emb = nn.functional.linear(proj_seq[:, 1+i], self.trans.weight.data)
            else:
                facet_emb = nn.functional.linear(proj_seq[:, 1+i], self.trans.weight.data[:,i*self.d_inp:(i+1)*self.d_inp])
            facet_arr.append(facet_emb)
            facet_norm[i] = facet_emb.norm(dim = -1).mean()
        facet_tensor = torch.stack(facet_arr, dim=1)
        facet_div = torch.mean( self.var_of_embedding(facet_tensor, dim = 1) )
        facet_var_batch = self.var_of_embedding(facet_tensor, dim = 0).mean(dim = 0)
        return facet_norm, facet_div, facet_var_batch, facet_tensor, facet_arr

    def forward(self, sequence, mask):
        global extracted_grads_word
        if len(mask.size()) < 3:
            mask = mask.unsqueeze(dim=-1)
        pad_mask = mask == 0
        proj_seq = sequence
        
        self.importance_arr = None
        if self.vis_facet_norm and self.facet_norm_act: #checking the model is in evaluation mode?
            facet_norm, facet_div, facet_var_batch, facet_tensor, facet_arr = self.compute_facet_norm(proj_seq) 
            bsz, token_num, emb_size = sequence.size()
            if self.analyze_grad and self.record_grad:
                importance_arr = torch.empty( (bsz, self.num_facet, token_num - ( 1 + self.num_facet)), device = self.trans.weight.data.device)
                for j in range(bsz):
                    facet_arr_norm = [facet[j,:].norm(dim = -1) for facet in facet_arr]
                    for i in range(self.num_facet):
                        #if i < self.num_facet - 1:
                        #    facet_arr[i][j].backward(retain_graph=True)
                        #else:
                        #    facet_arr[i][j].backward()
                        if i == self.num_facet - 1 and j == bsz - 1:
                            facet_arr_norm[i].backward()
                        else:
                            facet_arr_norm[i].backward(retain_graph=True)

                        grads = extracted_grads_word[0][j][1+self.num_facet:]
                        extracted_grads_word=[]
                        #grads = embedder_for_grad.model.embeddings.word_embeddings.grad[1+self.num_facet:]
                        #embedder_for_grad.model.embeddings.word_embeddings[token[1+self.num_facet:]]
                        dot=(grads).norm(dim=-1)
                        dot=dot/dot.sum()
                        dot-=dot.min()
                        dot/=dot.max()
                        #self.importance_arr.append(dot.tolist())
                        importance_arr[j, i, :] = dot
                        #self.embedder_for_grad.zero_grad()
                        #self.zero_grad()
                self.importance_arr = importance_arr.tolist()
                #log.info(facet_arr_norm)
                #log.info(grads)
                #log.info(self.importance_arr)
            self.facet_emb_arr = facet_tensor.tolist()

            self.facet_norm_sum += facet_norm
            self.facet_div_sum += facet_div
            self.facet_var_batch_sum += facet_var_batch
            self.pass_count += 1

        #proj_seq = self.project(sequence)  # linear project each hid state
        if self.pool_type == "max":
            proj_seq = proj_seq.masked_fill(pad_mask, -float("inf"))
            seq_emb = proj_seq.max(dim=1)[0]
        elif self.pool_type == "mean":
            proj_seq = proj_seq.masked_fill(pad_mask, 0)
            seq_emb = proj_seq.sum(dim=1) / mask.sum(dim=1)
        elif self.pool_type == "final":
            idxs = mask.expand_as(proj_seq).sum(dim=1, keepdim=True).long() - 1
            seq_emb = proj_seq.gather(dim=1, index=idxs).squeeze(dim=1)
        elif self.pool_type == "first" or self.pool_type == "first_init" or self.pool_type == "first_init_1" or self.pool_type == "first_init_avg":
            #seq_emb = proj_seq[:, 0]
            #seq_emb = proj_seq[:, 1]
            if self.pooler_dropout > 0 and self.num_facet > 1:
                #if self.training:
                #    bsz = proj_seq.size(0)
                #    chunck_size = int(bsz / self.num_facet)
                #    all_idx = list(range(bsz))
                #    random.shuffle(all_idx)
                #    f_idx = list(range(self.num_facet))
                #    random.shuffle(f_idx)
                #    proj_seq_new = torch.zeros_like(proj_seq)
                #    for i in range(1, 1+self.num_facet):
                #        chosen_facet = f_idx[i-1]
                #        if i == self.num_facet:
                #            proj_seq_new[ all_idx[ (i-1)*chunck_size: ], chosen_facet ] = proj_seq[all_idx[ (i-1)*chunck_size: ], chosen_facet ]
                #        else:
                #            proj_seq_new[ all_idx[ (i-1)*chunck_size: i*chunck_size ], chosen_facet ] = proj_seq[all_idx[ (i-1)*chunck_size: i*chunck_size ], chosen_facet ]
                #    proj_seq[:, 1:(self.num_facet+1)] = proj_seq_new[:, 1:(self.num_facet+1)]
                #else:
                #    proj_seq = proj_seq / self.num_facet
                
                bsz = proj_seq.size(0)
                sample_num = min(1, int(self.pooler_dropout * bsz) )
                if self.training:
                    chosen_idx = random.randint(1,self.num_facet)
                    all_idx = list(range(bsz))
                    random.shuffle(all_idx)
                    selected_idx = all_idx[:sample_num]
                    proj_seq[selected_idx, chosen_idx] = 0
                else:
                   reduce_ratio = sample_num / float(bsz) / self.num_facet
                   proj_seq = proj_seq / ( 1 - reduce_ratio)
                #if random.random() <= self.pooler_dropout:
                #    chosen_idx = random.randint(1,self.num_facet)
                #    proj_seq[:, chosen_idx] = 0
            seq_emb = self.drop( proj_seq[:, 1:(self.num_facet+1)].reshape(-1,self.num_facet*proj_seq.shape[-1]) )
            seq_emb = self.trans(seq_emb)
        elif self.pool_type == "single_first":
            seq_emb = proj_seq[:, 1]
            seq_emb = self.trans(seq_emb)
        elif self.pool_type == "lin":
            seq_emb =  proj_seq[:, 1:(self.num_facet+1)] * (self.weight_global.unsqueeze(dim=0).unsqueeze(dim=-1))
            seq_emb = seq_emb.sum(dim=1)
        elif self.pool_type == "facet_mean":
            seq_emb =  proj_seq[:, 1:(self.num_facet+1)].mean(dim=1) 
        elif self.pool_type == "proj_avg_train" or self.pool_type == "proj_avg_train_skip":
            if self.num_facet == 1:
                pooled_facet = proj_seq[:,1]
                seq_emb = self.sent['s_1'](pooled_facet)
            else:
                avg_w = 0
                for i in range(1,(self.num_facet+1)):
                    avg_w = avg_w + self.sent['s_'+str(i)].weight
                avg_w = avg_w / self.num_facet
                seq_emb = 0
                for i in range(1,(self.num_facet+1)):
                    if self.pool_type == "proj_avg_train_skip" and i == 1:
                        seq_emb += nn.functional.linear(proj_seq[:, i], self.sent['s_'+str(i)].weight, self.sent['s_'+str(i)].bias)
                    else:
                        seq_emb += nn.functional.linear(proj_seq[:, i], self.sent['s_'+str(i)].weight - avg_w, self.sent['s_'+str(i)].bias)
        elif self.pool_type[:9] == "proj_init":
            emb_list=[]
            for i in range(1,(self.num_facet+1)):
                pooled_facet = proj_seq[:,i]
                emb_list.append(self.sent['s_'+str( min(i,self.num_out_head) )](pooled_facet))
            haed_emb = torch.stack(emb_list, dim=1)
            if self.pool_type == "proj_init_first":
                seq_emb = self.trans( haed_emb.reshape(-1,self.num_facet*proj_seq.shape[-1]) )
            elif self.pool_type == "proj_init_lin":
                seq_emb =  haed_emb * (self.weight_global.unsqueeze(dim=0).unsqueeze(dim=-1))
                seq_emb = seq_emb.sum(dim=1)
            elif self.pool_type == "proj_init_mean":
                seq_emb = haed_emb.mean(dim=1)
            

        elif self.pool_type[:5] == "head_":
            emb_list=[]
            for i in range(1,(self.num_facet+1)):
                #pooled_facet = self.sent['v_'+str(i)](proj_seq[:,i])
                pooled_facet = proj_seq[:,i]
                #emb_list.append(pooled_facet)
                emb_list.append(self.sent['s_'+str( min(i,self.num_out_head) )](pooled_facet, not_norm=self.unnorm_facet))
            haed_emb = torch.stack(emb_list, dim=1)
            if self.pool_type == "head_first":
                seq_emb = self.trans( haed_emb.reshape(-1,self.num_facet*proj_seq.shape[-1]) )
            elif self.pool_type == "head_lin":
                seq_emb =  haed_emb * (self.weight_global.unsqueeze(dim=0).unsqueeze(dim=-1))
                seq_emb = seq_emb.sum(dim=1)
            
        return seq_emb


class Classifier(nn.Module):
    """ Logistic regression or MLP classifier """

    # NOTE: Expects dropout to have already been applied to its input.

    def __init__(self, d_inp, n_classes, cls_type="mlp", dropout=0.2, d_hid=512):
        super(Classifier, self).__init__()
        if cls_type == "log_reg":
            classifier = nn.Linear(d_inp, n_classes)
        elif cls_type == "mlp":
            classifier = nn.Sequential(
                nn.Linear(d_inp, d_hid),
                nn.Tanh(),
                nn.LayerNorm(d_hid),
                nn.Dropout(dropout),
                nn.Linear(d_hid, n_classes),
            )
        elif cls_type == "fancy_mlp":  # What they did in Infersent.
            classifier = nn.Sequential(
                nn.Linear(d_inp, d_hid),
                nn.Tanh(),
                nn.LayerNorm(d_hid),
                nn.Dropout(dropout),
                nn.Linear(d_hid, d_hid),
                nn.Tanh(),
                nn.LayerNorm(d_hid),
                nn.Dropout(p=dropout),
                nn.Linear(d_hid, n_classes),
            )
        else:
            raise ValueError("Classifier type %s not found" % type)
        self.classifier = classifier

    def forward(self, seq_emb):
        logits = self.classifier(seq_emb)
        return logits

    @classmethod
    def from_params(cls, d_inp, n_classes, params):
        return cls(
            d_inp,
            n_classes,
            cls_type=params["cls_type"],
            dropout=params["dropout"],
            d_hid=params["d_hid"],
        )


class SingleClassifier(nn.Module):
    """ Thin wrapper around a set of modules. For single-sentence classification. """

    def __init__(self, pooler, classifier):
        super(SingleClassifier, self).__init__()
        self.pooler = pooler
        self.classifier = classifier

    def forward(self, sent, mask, idxs=[]):
        """
        This class applies some type of pooling to get a fixed-size vector,
            possibly extracts some specific representations from the input sequence
            and concatenates those reps to the overall representations,
            then passes the whole thing through a classifier.

        args:
            - sent (FloatTensor): sequence of hidden states representing a sentence
            Assumes batch_size x seq_len x d_emb.
            - mask (FloatTensor): binary masking denoting which elements of sent are not padding
            - idxs (List[LongTensor]): list of indices of to extract from sent and
                concatenate to the post-pooling representation.
                For each element in idxs, we extract all the non-pad (0) representations, pool,
                and concatenate the resulting fixed size vector to the overall representation.

        returns:
            - logits (FloatTensor): logits for classes
        """

        emb = self.pooler(sent, mask)

        # append any specific token representations, e.g. for WiC task
        ctx_embs = []
        for idx in idxs:
            if len(idx.shape) == 1:
                idx = idx.unsqueeze(-1)
            if len(idx.shape) == 2:
                idx = idx.unsqueeze(-1)
            if len(idx.shape) == 3:
                assert idx.size(-1) == 1 or idx.size(-1) == sent.size(
                    -1
                ), "Invalid index dimension!"
                idx = idx.expand([-1, -1, sent.size(-1)]).long()
            else:
                raise ValueError("Invalid dimensions of index tensor!")

            ctx_mask = (idx != 0).float()
            # the first element of the mask should never be zero
            ctx_mask[:, 0] = 1
            ctx_emb = sent.gather(dim=1, index=idx) * ctx_mask
            ctx_emb = ctx_emb.sum(dim=1) / ctx_mask.sum(dim=1)
            ctx_embs.append(ctx_emb)

        final_emb = torch.cat([emb] + ctx_embs, dim=-1)
        logits = self.classifier(final_emb)
        return logits


class PairClassifier(nn.Module):
    """ Thin wrapper around a set of modules.
    For sentence pair classification.
    Pooler specifies how to aggregate inputted sequence of vectors.
    Also allows for use of specific token representations to be addded to the overall
    representation
    """

    def __init__(self, pooler, classifier, attn=None):
        super(PairClassifier, self).__init__()
        self.pooler = pooler
        self.classifier = classifier
        self.attn = attn

    def forward(self, s1, s2, mask1, mask2, idx1=[], idx2=[]):
        """
        This class applies some type of pooling to each of two inputs to get two fixed-size vectors,
            possibly extracts some specific representations from the input sequence
            and concatenates those reps to the overall representations,
            then passes the whole thing through a classifier.

        args:
            - s1/s2 (FloatTensor): sequence of hidden states representing a sentence
                Assumes batch_size x seq_len x d_emb.
            - mask1/mask2 (FloatTensor): binary masking denoting which elements of sent are not padding
            - idx{1,2} (List[LongTensor]): list of indices of to extract from sent and
                concatenate to the post-pooling representation.
                For each element in idxs, we extract all the non-pad (0) representations, pool,
                and concatenate the resulting fixed size vector to the overall representation.

        returns:
            - logits (FloatTensor): logits for classes
        """

        mask1 = mask1.squeeze(-1) if len(mask1.size()) > 2 else mask1
        mask2 = mask2.squeeze(-1) if len(mask2.size()) > 2 else mask2
        if self.attn is not None:
            s1, s2 = self.attn(s1, s2, mask1, mask2)
        emb1 = self.pooler(s1, mask1)
        emb2 = self.pooler(s2, mask2)

        s1_ctx_embs = []
        for idx in idx1:
            if len(idx.shape) == 1:
                idx = idx.unsqueeze(-1)
            if len(idx.shape) == 2:
                idx = idx.unsqueeze(-1)
            if len(idx.shape) == 3:
                assert idx.size(-1) == 1 or idx.size(-1) == s1.size(-1), "Invalid index dimension!"
                idx = idx.expand([-1, -1, s1.size(-1)]).long()
            else:
                raise ValueError("Invalid dimensions of index tensor!")

            s1_ctx_mask = (idx != 0).float()
            # the first element of the mask should never be zero
            s1_ctx_mask[:, 0] = 1
            s1_ctx_emb = s1.gather(dim=1, index=idx) * s1_ctx_mask
            s1_ctx_emb = s1_ctx_emb.sum(dim=1) / s1_ctx_mask.sum(dim=1)
            s1_ctx_embs.append(s1_ctx_emb)
        emb1 = torch.cat([emb1] + s1_ctx_embs, dim=-1)

        s2_ctx_embs = []
        for idx in idx2:
            if len(idx.shape) == 1:
                idx = idx.unsqueeze(-1)
            if len(idx.shape) == 2:
                idx = idx.unsqueeze(-1)
            if len(idx.shape) == 3:
                assert idx.size(-1) == 1 or idx.size(-1) == s2.size(-1), "Invalid index dimension!"
                idx = idx.expand([-1, -1, s2.size(-1)]).long()
            else:
                raise ValueError("Invalid dimensions of index tensor!")

            s2_ctx_mask = (idx != 0).float()
            # the first element of the mask should never be zero
            s2_ctx_mask[:, 0] = 1
            s2_ctx_emb = s2.gather(dim=1, index=idx) * s2_ctx_mask
            s2_ctx_emb = s2_ctx_emb.sum(dim=1) / s2_ctx_mask.sum(dim=1)
            s2_ctx_embs.append(s2_ctx_emb)
        emb2 = torch.cat([emb2] + s2_ctx_embs, dim=-1)

        pair_emb = torch.cat([emb1, emb2, torch.abs(emb1 - emb2), emb1 * emb2], 1)
        logits = self.classifier(pair_emb)
        return logits
