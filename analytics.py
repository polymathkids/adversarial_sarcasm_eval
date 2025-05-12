import datasets
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk, Features, Value, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy
import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def scatter():

    # Gold Label = 0
    students_id = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    students_marks = np.array([95, 98, 83, 75, 67, 58, 67, 78, 53, 32])
    plt.scatter(students_id, students_marks, color='black')

    # Gold label = 1
    students_id = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    students_marks = np.array([58, 90, 67, 78, 53, 32, 95, 98, 83, 67, ])
    plt.scatter(students_id, students_marks, color='violet')

    # Gold label = 2
    students_id = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    students_marks = np.array([58, 90, 67, 78, 53, 32, 95, 98, 83, 67, ])
    plt.scatter(students_id, students_marks, color='violet')

    plt.show()
