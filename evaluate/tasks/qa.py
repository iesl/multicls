"""Task definitions for question answering tasks."""
import os
import re
import json
import string
import collections
import math
from typing import Iterable, Sequence, Type
import logging as log

import torch
from allennlp.training.metrics import Average, F1Measure
from allennlp.data.fields import LabelField, MetadataField
from allennlp.data import Instance

from ..utils.data_loaders import process_sentence

from .tasks import sample_training_data
from .tasks import Task
from .tasks import sentence_to_text_field
from .registry import register_task
import numpy as np
import random

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace.
    From official ReCoRD eval script """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """ Compute normalized token level F1
    From official ReCoRD eval script """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    """ Compute normalized exact match
    From official ReCoRD eval script """
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """ Compute max metric between prediction and each ground truth.
    From official ReCoRD eval script """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


@register_task("multirc", rel_path="MultiRC/")
class MultiRCTask(Task):
    """Multi-sentence Reading Comprehension task
    See paper at https://cogcomp.org/multirc/ """

    def __init__(self, path, max_seq_len, name, **kw):
        """ """
        super().__init__(name, **kw)
        self.scorer1 = F1Measure(positive_label=1)
        self.scorer2 = Average()  # to delete
        self.scorer3 = F1Measure(positive_label=1)
        self._score_tracker = collections.defaultdict(list)
        self.val_metric = "%s_avg" % self.name
        self.val_metric_decreases = False
        self.max_seq_len = max_seq_len
        self.files_by_split = {
            "train": os.path.join(path, "train.jsonl"),
            "val": os.path.join(path, "val.jsonl"),
            "test": os.path.join(path, "test.jsonl"),
        }
        #self.sample_idx = -1

    def load_data(self, few_shot, num_facet):
        self.num_facet = num_facet
        self.few_shot = few_shot
        #self.sample_idx = -1
        #print(self.few_shot)
        #print(self.sample_idx)
        # Data is exposed as iterable: no preloading
        pass

    def get_split_text(self, split: str):
        """ Get split text as iterable of records.

        Split should be one of "train", "val", or "test".
        """
        #print(self.few_shot)
        #print(self.sample_idx)
        training_mode = False
        if split.startswith("train"):
            training_mode = True
        return self.load_data_for_path(self.files_by_split[split], training_mode)

    def load_data_for_path(self, path, training_mode):
        """ Load data """

        with open(path, encoding="utf-8") as data_fh:
            examples = []
            num_q = 0
            for example in data_fh:
                ex = json.loads(example)

                assert (
                    "version" in ex and ex["version"] == 1.1
                ), "MultiRC version is invalid! Example indices are likely incorrect. Please re-download the data from super.gluebenchmark.com ."

                # each example has a passage field -> (text, questions)
                # text is the passage, which requires some preprocessing
                # questions is a list of questions, has fields (question, sentences_used, answers)
                ex["passage"]["text"] = process_sentence(
                    self.tokenizer_name, ex["passage"]["text"], self.max_seq_len, insert_facet=True, num_facet=self.num_facet
                )
                for question in ex["passage"]["questions"]:
                    num_q += 1
                    question["question"] = process_sentence(
                        self.tokenizer_name, question["question"], self.max_seq_len, insert_facet=False
                    )
                    for answer in question["answers"]:
                        answer["text"] = process_sentence(
                            self.tokenizer_name, answer["text"], self.max_seq_len, insert_facet=False
                        )
                examples.append(ex)
        if training_mode and self.few_shot[0] != -1:
            #assert len(examples) < self.few_shot[0]
            
            num_samples = 0
            examples_subsample = []
            counter = 0 
            seed = self.few_shot[1]
            for ex in examples:
                q_num_ex = len(ex["passage"]["questions"])
                new_ex = {}
                new_ex["passage"] = {}
                new_ex["passage"]['text'] = ex["passage"]["text"]
                new_ex['idx'] = ex["idx"]
                #sample_q_num = max(1,int( self.few_shot * q_num_ex / num_q  ))
                sample_q_num = math.ceil( self.few_shot[0] * q_num_ex / num_q  )
                sample_idx, seed = sample_training_data(q_num_ex, [sample_q_num, seed,  self.few_shot[2]])
                #permuted_idx = np.random.permutation(q_num_ex)
                #sample_idx = permuted_idx[:sample_q_num]
                num_samples += sample_q_num
                new_ex["passage"]["questions"] = [ex["passage"]["questions"][idx] for idx in sample_idx]
                for question in new_ex["passage"]["questions"]:
                    question["answers"] = [ question["answers"][counter % len(question["answers"])]  ] 
                    counter += 1
                    #question["answers"] = [ random.choice(question["answers"]) ]
                examples_subsample.append(new_ex)
            sample_idx, seed = sample_training_data(num_samples, [self.few_shot[0], seed,  self.few_shot[2]])
            final_num_samples = 0
            keep_sample_set = set(sample_idx)
            sample_now_idx = 0
            for ex in examples_subsample:
                q_num_ex = len(ex["passage"]["questions"])
                #for idx in range(sample_now_idx, sample_now_idx + q_num_ex):
                #del_idx_list = []
                keep_list_idx = []
                for idx in range(q_num_ex):
                    idx_global = sample_now_idx + idx
                    #if idx_global not in keep_sample_set:
                    #    del_idx_list.append(idx_global)
                    if idx_global in keep_sample_set:
                        keep_list_idx.append(idx)
                ex["passage"]["questions"] = [ex["passage"]["questions"][q_idx] for q_idx in keep_list_idx]
                sample_now_idx += q_num_ex
                final_num_samples += len(ex["passage"]["questions"])
            assert sample_now_idx == num_samples
            log.info("number of samples in multirc: {} -> {}".format(num_samples, final_num_samples) )
            return examples_subsample
            
            #if self.sample_idx == -1:
            #    permuted_idx = np.random.permutation(len(examples))
            #    sample_idx1 = permuted_idx[:self.few_shot]
            #    self.sample_idx = sample_idx1
            #else:
            #    sample_idx1 = self.sample_idx
            #return [examples[idx] for idx in sample_idx1]
        else:
            return examples

    def get_sentences(self) -> Iterable[Sequence[str]]:
        """ Yield sentences, used to compute vocabulary. """
        for split in self.files_by_split:
            if split.startswith("test"):
                continue
            path = self.files_by_split[split]
            training_mode = False
            #if split.startswith("train"):
            #    training_mode = True
            # It's alright to let vocab consider more words
            for example in self.load_data_for_path(path, training_mode):
                yield example["passage"]["text"]
                for question in example["passage"]["questions"]:
                    yield question["question"]
                    for answer in question["answers"]:
                        yield answer["text"]

    def process_split(self, split, indexers, num_facet) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """
        is_using_bert = "bert_wpm_pretokenized" in indexers

        def _make_instance(passage, question, answer, label, par_idx, qst_idx, ans_idx):
            """ pq_id: passage-question ID """
            d = {}
            d["psg_str"] = MetadataField(" ".join(passage))
            d["qst_str"] = MetadataField(" ".join(question))
            d["ans_str"] = MetadataField(" ".join(answer))
            d["psg_idx"] = MetadataField(par_idx)
            d["qst_idx"] = MetadataField(qst_idx)
            d["ans_idx"] = MetadataField(ans_idx)
            d["idx"] = MetadataField(ans_idx)  # required by evaluate()
            if is_using_bert:
                inp = para + question[1:-1] + answer[1:]
                d["psg_qst_ans"] = sentence_to_text_field(inp, indexers)
            else:
                d["psg"] = sentence_to_text_field(passage, indexers)
                d["qst"] = sentence_to_text_field(question, indexers)
                d["ans"] = sentence_to_text_field(answer, indexers)
            d["label"] = LabelField(label, label_namespace="labels", skip_indexing=True)

            return Instance(d)

        for example in split:
            par_idx = example["idx"]
            para = example["passage"]["text"]
            for ex in example["passage"]["questions"]:
                qst_idx = ex["idx"]
                question = ex["question"]
                for answer in ex["answers"]:
                    ans_idx = answer["idx"]
                    ans = answer["text"]
                    label = int(answer["label"]) if "label" in answer else 0
                    yield _make_instance(para, question, ans, label, par_idx, qst_idx, ans_idx)

    def count_examples(self):
        """ Compute here b/c we"re streaming the sentences. """
        example_counts = {}
        for split, split_path in self.files_by_split.items():
            if split.startswith("train") and self.few_shot[0] != -1:
                example_counts[split] = self.few_shot[0]
            else:
                example_counts[split] = sum(
                    len(q["answers"])
                    for r in open(split_path, "r", encoding="utf-8")
                    for q in json.loads(r)["passage"]["questions"]
                )

        self.example_counts = example_counts

    def update_metrics(self, logits, labels, idxs, tagmask=None):
        """ A batch of logits, labels, and the passage+questions they go with """
        self.scorer1(logits, labels)
        logits, labels = logits.detach().cpu(), labels.detach().cpu()
        # track progress on each question
        for ex, logit, label in zip(idxs, logits, labels):
            self._score_tracker[ex].append((logit, label))

    def get_metrics(self, reset=False):
        """Get metrics specific to the task"""
        _, _, ans_f1 = self.scorer1.get_metric(reset)

        ems, f1s = [], []
        for logits_and_labels in self._score_tracker.values():
            logits, labels = list(zip(*logits_and_labels))
            logits = torch.stack(logits)
            labels = torch.stack(labels)

            # question F1
            self.scorer3(logits, labels)
            __, _, ex_f1 = self.scorer3.get_metric(reset=True)
            f1s.append(ex_f1)

            # EM
            preds = logits.argmax(dim=-1)
            ex_em = (torch.eq(preds, labels).sum() == preds.nelement()).item()
            ems.append(ex_em)
        em = sum(ems) / len(ems)
        qst_f1 = sum(f1s) / len(f1s)

        if reset:
            self._score_tracker = collections.defaultdict(list)

        return {"ans_f1": ans_f1, "qst_f1": qst_f1, "em": em, "avg": (ans_f1 + em) / 2}


@register_task("record", rel_path="ReCoRD/")
class ReCoRDTask(Task):
    """Reading Comprehension with commonsense Reasoning Dataset
    See paper at https://sheng-z.github.io/ReCoRD-explorer """

    def __init__(self, path, max_seq_len, name, **kw):
        """ """
        super().__init__(name, **kw)
        self.val_metric = "%s_avg" % self.name
        self.val_metric_decreases = False
        self._score_tracker = collections.defaultdict(list)
        self._answers = None
        self.max_seq_len = max_seq_len
        self.files_by_split = {
            "train": os.path.join(path, "train.jsonl"),
            "val": os.path.join(path, "val.jsonl"),
            "test": os.path.join(path, "test.jsonl"),
        }

    def load_data(self, few_shot, num_facet):
        assert few_shot[0] == -1
        self.num_facet = num_facet
        # Data is exposed as iterable: no preloading
        pass

    def get_split_text(self, split: str):
        """ Get split text as iterable of records.

        Split should be one of "train", "val", or "test".
        """
        return self.load_data_for_path(self.files_by_split[split], split)

    def load_data_for_path(self, path, split):
        """ Load data """

        def tokenize_preserve_placeholder(sent):
            """ Tokenize questions while preserving @placeholder token """
            sent_parts = sent.split("@placeholder")
            assert len(sent_parts) == 2
            #sent_parts = [ 
            #    process_sentence(self.tokenizer_name, sent_parts[0], self.max_seq_len, insert_facet=False),
            #    process_sentence(self.tokenizer_name, sent_parts[1], self.max_seq_len, insert_facet=False)
            #]
            sent_parts = [
                process_sentence(self.tokenizer_name, s, self.max_seq_len, insert_facet=False) for s in sent_parts
            ]
            return sent_parts[0][:-1] + ["@placeholder"] + sent_parts[1][1:]

        examples = []
        data = [json.loads(d) for d in open(path, encoding="utf-8")]
        for item in data:
            psg_id = item["idx"]
            psg = process_sentence(self.tokenizer_name, item["passage"]["text"], self.max_seq_len, insert_facet=True, num_facet=self.num_facet)
            ent_idxs = item["passage"]["entities"]
            ents = [item["passage"]["text"][idx["start"] : idx["end"] + 1] for idx in ent_idxs]
            qas = item["qas"]
            for qa in qas:
                qst = tokenize_preserve_placeholder(qa["query"])
                qst_id = qa["idx"]
                if "answers" in qa:
                    anss = [a["text"] for a in qa["answers"]]
                else:
                    anss = []
                ex = {
                    "passage": psg,
                    "ents": ents,
                    "query": qst,
                    "answers": anss,
                    "psg_id": f"{split}-{psg_id}",
                    "qst_id": qst_id,
                }
                examples.append(ex)

        return examples

    def _load_answers(self) -> None:
        """ """
        answers = {}
        for split, split_path in self.files_by_split.items():
            data = [json.loads(d) for d in open(split_path, encoding="utf-8")]
            for item in data:
                psg_id = f"{split}-{item['idx']}"
                for qa in item["qas"]:
                    qst_id = qa["idx"]
                    if "answers" in qa:
                        answers[(psg_id, qst_id)] = [a["text"] for a in qa["answers"]]
                    else:
                        answers[(psg_id, qst_id)] = ["No answer"]
        self._answers = answers

    def get_sentences(self) -> Iterable[Sequence[str]]:
        """ Yield sentences, used to compute vocabulary. """
        for split in self.files_by_split:
            if split.startswith("test"):
                continue
            path = self.files_by_split[split]
            for example in self.load_data_for_path(path, split):
                yield example["passage"]
                yield example["query"]

    def process_split(self, split, indexers, num_facet) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """
        is_using_bert = "bert_wpm_pretokenized" in indexers

        def is_answer(x, ys):
            """ Given a list of answers, determine if x is an answer """
            return x in ys

        def insert_ent(ent, template):
            """ Replace ent into template (query with @placeholder) """
            assert "@placeholder" in template, "No placeholder detected!"
            split_idx = template.index("@placeholder")
            return template[:split_idx] + ent + template[split_idx + 1 :]

        def _make_instance(psg, qst, ans_str, label, psg_idx, qst_idx, ans_idx):
            """ pq_id: passage-question ID """
            d = {}
            d["psg_str"] = MetadataField(" ".join(psg))
            d["qst_str"] = MetadataField(" ".join(qst))
            d["ans_str"] = MetadataField(ans_str)
            d["psg_idx"] = MetadataField(par_idx)
            d["qst_idx"] = MetadataField(qst_idx)
            d["ans_idx"] = MetadataField(ans_idx)
            d["idx"] = MetadataField(ans_idx)  # required by evaluate()
            if is_using_bert:
                inp = psg + qst[1:]
                d["psg_qst_ans"] = sentence_to_text_field(inp, indexers)
            else:
                d["psg"] = sentence_to_text_field(psg, indexers)
                d["qst"] = sentence_to_text_field(qst, indexers)
            d["label"] = LabelField(label, label_namespace="labels", skip_indexing=True)

            return Instance(d)

        for example in split:
            psg = example["passage"]
            qst_template = example["query"]

            ent_strs = example["ents"]
            ents = [
                process_sentence(self._tokenizer_name, ent, self.max_seq_len, insert_facet=False)[1:-1]
                for ent in ent_strs
            ]

            anss = example["answers"]
            par_idx = example["psg_id"]
            qst_idx = example["qst_id"]
            for ent_idx, (ent, ent_str) in enumerate(zip(ents, ent_strs)):
                label = is_answer(ent_str, anss)
                qst = insert_ent(ent, qst_template)
                yield _make_instance(psg, qst, ent_str, label, par_idx, qst_idx, ent_idx)

    def count_examples(self):
        """ Compute here b/c we're streaming the sentences. """
        example_counts = {}
        for split, split_path in self.files_by_split.items():
            data = [json.loads(d) for d in open(split_path, encoding="utf-8")]
            example_counts[split] = sum([len(d["passage"]["entities"]) for d in data])
        self.example_counts = example_counts

    def update_metrics(self, logits, anss, idxs, tagmask=None):
        """ A batch of logits+answer strings and the questions they go with """
        logits = logits.detach().cpu()
        for idx, logit, ans in zip(idxs, logits, anss):
            self._score_tracker[idx].append((logit, ans))

    def get_metrics(self, reset=False):
        """Get metrics specific to the task"""

        # Load asnwers, used for computing metrics
        if self._answers is None:
            self._load_answers()

        ems, f1s = [], []
        for idx, logits_and_anss in self._score_tracker.items():
            golds = self._answers[idx]
            logits_and_anss.sort(key=lambda x: x[1])
            logits, anss = list(zip(*logits_and_anss))
            logits = torch.stack(logits)

            # take the most probable choice as the model prediction
            pred_idx = torch.softmax(logits, dim=-1)[:, -1].argmax().item()
            pred = anss[pred_idx]

            # F1
            f1 = metric_max_over_ground_truths(f1_score, pred, golds)
            f1s.append(f1)

            # EM
            em = metric_max_over_ground_truths(exact_match_score, pred, golds)
            ems.append(em)

        em = sum(ems) / len(ems)
        f1 = sum(f1s) / len(f1s)

        if reset:
            self._score_tracker = collections.defaultdict(list)

        return {"f1": f1, "em": em, "avg": (f1 + em) / 2}
