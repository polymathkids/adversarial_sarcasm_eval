import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy
import os
import json
import torch
#from run import main


import pandas as pd
import numpy as np

headlines = pd.read_json("./dataset/Sarcasm_Headlines_Dataset_v2.json", lines=True)
#shuffle the data inplace
headlines = headlines.sample(frac=1).reset_index(drop=True)
# show first few rows
print(headlines.columns)
print(headlines.head())

reddit = pd.read_csv("./dataset/train-balanced-sarcasm.csv")
reddit = reddit.sample(frac = 1).reset_index(drop = True) #shuffle lines in dataset
reddit.loc[(reddit.label == 1), 'label'] = 2 #change all instances of sarcasm to be the same as a contradiction leaving the rest as label 0 =entailment
reddit.loc[(reddit.label == 0), 'label'] = 1 #change all other instances of sarcasm to be label 1 =neutral in line with RTE datasets

reddit.rename(columns = {'parent_comment':'premise'}, inplace = True) #rename columns
reddit.rename(columns = {'comment':'hypothesis'}, inplace = True)
reddit = reddit.drop(['author', 'subreddit', 'score', 'ups', 'downs',
       'date', 'created_utc'], axis=1) #drop uneeded input columns
reddit = reddit[['premise', 'hypothesis', 'label']] #reorder colunns
# show column names
print(reddit.columns)
print(type(reddit))
print(reddit.head())

reddit['premise'].astype(str)
reddit['hypothesis'].astype(str)
reddit['label'].astype(int)

print(reddit.columns)
print(type(reddit))
print(reddit.head())


train_reddit = reddit.sample(frac=0.8,random_state=0)
test_reddit = reddit.drop(train_reddit.index)
reddit['train'] = eddit.sample(frac=0.8,random_state=0)
sarcasm['eval'] = test_reddit
print("Sarcasm!! ", sarcasm.shape)

print(train_reddit.shape)
print(test_reddit.shape)


reddit.to_json('./dataset/reddit.json', orient = 'records', lines=True) #?? , orient='records'
reddit.to_csv('./dataset/reddit.csv', index=False) #?? header=False, index=False

from datasets import load_dataset

watson = load_dataset("chitra/contradictionNLI")
watson.save_to_disk('./dataset')

#MNLI:
mnli = datasets.load_dataset('glue','mnli')

        mnli_train = mnli['train']

        mnli_eval = mnli['validation_matched']

        snli = datasets.load_dataset('snli')
        snli_train = snli['train']
        snli_eval = snli['validation']

        mnli_train = mnli_train.remove_columns(['idx'])
        mnli_eval = mnli_eval.remove_columns(['idx'])
        #print(mnli_train) #debug
        #print(mnli_eval)
        snli_train = snli_train.cast(mnli_train.features)
        smnli_train = datasets.concatenate_datasets([snli_train, mnli_train])

        snli_eval =  snli_eval.cast(mnli_eval.features)

        smnli_eval = datasets.concatenate_datasets([snli_eval, mnli_eval])

        smnli_train.to_json("./dataset/smnli-train.json")
        smnli_eval.to_json("./dataset/smnli-eval.json")

#Can use SPE sentence embeddign technique to preserve semantics and improve adversarial attack inputs
import spe

input_sentences = input_sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "I am a sentence for which I would like to get its embedding"]

output_vectors = spe.spe(input_sentences)

print(output_vectors)

# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
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

# Lint as: python3
"""SQUAD: The Stanford Question Answering Dataset."""


import json

import datasets
from datasets.tasks import QuestionAnsweringExtractive


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@article{2016arXiv160605250R,
       author = {{Rajpurkar}, Pranav and {Zhang}, Jian and {Lopyrev},
                 Konstantin and {Liang}, Percy},
        title = "{SQuAD: 100,000+ Questions for Machine Comprehension of Text}",
      journal = {arXiv e-prints},
         year = 2016,
          eid = {arXiv:1606.05250},
        pages = {arXiv:1606.05250},
archivePrefix = {arXiv},
       eprint = {1606.05250},
}
"""

_DESCRIPTION = """\
Stanford Question Answering Dataset (SQuAD) is a reading comprehension \
dataset, consisting of questions posed by crowdworkers on a set of Wikipedia \
articles, where the answer to every question is a segment of text, or span, \
from the corresponding reading passage, or the question might be unanswerable.
"""

_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
_URLS = {
    "train": _URL + "train-v1.1.json",
    "dev": _URL + "dev-v1.1.json",
}


