import os
from pathlib import Path
import pickle

from argparse import ArgumentParser
import numpy as np
import pandas as pd

import torch
from torch import nn
import time
import ner_datasets
# from models import BERT_CRF, BERT_BiLSTM_CRF
import conlleval
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import f1_score, classification_report
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    BertTokenizer, 
    BertForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
    AdamW,
    get_linear_schedule_with_warmup,
)


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('using device:', device)


def ensure_dir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    return dir_path


parser = ArgumentParser()
parser.add_argument('--label_schema', default="IO", type=str)
parser.add_argument('--model_name', choices=['scibert', 'matscibert', 'bert'], default="matscibert", type=str)
parser.add_argument('--model_save_dir', default="model/", type=str)
parser.add_argument('--preds_save_dir', default="preds/", type=str)
parser.add_argument('--cache_dir', default="../.cache/", type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--weight_decay',default=1e-4,type=int)
parser.add_argument('--shot', default=32, type=int)
parser.add_argument('--lm_lrs', default=1e-4, type=float)
parser.add_argument('--non_lm_lr', default=3e-4, type=float)
parser.add_argument('--architecture', choices=['bert', 'bert-crf', 'bert-bilstm-crf'], default="bert", type=str)
parser.add_argument('--dataset_name', choices=['sofc', 'sofc_slot', 'matscholar'], default="matscholar", type=str)
parser.add_argument('--fold_num', default=None, type=int)
parser.add_argument('--hidden_dim', default=300, type=int)
parser.add_argument('--device', default=0, type=int)
args = parser.parse_args()

if args.model_name == 'scibert':
    model_name = 'allenai/scibert_scivocab_uncased'
    to_normalize = False
elif args.model_name == 'matscibert':
    model_name = '/home/liwentao/learn/DecT_Mat_NER/model'
    to_normalize = True
elif args.model_name == 'bert':
    model_name = 'bert-base-uncased'
    to_normalize = False
else:
    raise NotImplementedError

dataset_name = args.dataset_name
fold_num = args.fold_num
model_revision = 'main'
cache_dir = ensure_dir(args.cache_dir) if args.cache_dir else None
output_dir = ensure_dir(args.model_save_dir)
preds_save_dir = ensure_dir(args.preds_save_dir) if args.preds_save_dir else None
if preds_save_dir:
    preds_save_dir = os.path.join(preds_save_dir, dataset_name)
    if fold_num:
        preds_save_dir = os.path.join(preds_save_dir, f'cv_{fold_num}')
    preds_save_dir = ensure_dir(preds_save_dir)


train_X, train_y, val_X, val_y, test_X, test_y = ner_datasets.get_ner_data(dataset_name,args.shot,args.seed, fold=fold_num, norm=to_normalize)
print(len(train_X), len(val_X), len(test_X))

def transfer_to_IO_schema(labels):
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            if labels[i][j] != "O":
                labels[i][j] = "I-"+labels[i][j][2:]

if args.label_schema=='IO':
    transfer_to_IO_schema(train_y)
    transfer_to_IO_schema(val_y)
    transfer_to_IO_schema(test_y)

unique_labels = set(label for sent in train_y for label in sent)
label_list = sorted(list(unique_labels))
print(label_list)
tag2id = {tag: id for id, tag in enumerate(label_list)}
id2tag = {id: tag for tag, id in tag2id.items()}
if dataset_name == 'sofc_slot':
    id2tag[tag2id['B-experiment_evoking_word']] = 'O'
    id2tag[tag2id['I-experiment_evoking_word']] = 'O'
num_labels = len(label_list)

cnt = dict()
for sent in train_y:
    for label in sent:
        if label[0] in ['I', 'B']: tag = label[2:]
        else: continue
        if tag not in cnt: cnt[tag] = 1
        else: cnt[tag] += 1

eval_labels = sorted([l for l in cnt.keys() if l != 'experiment_evoking_word'])

tokenizer_kwargs = {
    'cache_dir': cache_dir,
    'use_fast': True,
    'revision': model_revision,
    'use_auth_token': None,
    'model_max_length': 512
}
tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)


def remove_zero_len_tokens(X, y):
    new_X, new_y = [], []
    for sent, labels in zip(X, y):
        new_sent, new_labels = [], []
        for token, label in zip(sent, labels):
            if len(tokenizer.tokenize(token)) == 0:
                assert dataset_name == 'matscholar'
                continue
            new_sent.append(token)
            new_labels.append(label)
        new_X.append(new_sent)
        new_y.append(new_labels)
    return new_X, new_y


train_X, train_y = remove_zero_len_tokens(train_X, train_y)
val_X, val_y = remove_zero_len_tokens(val_X, val_y)
test_X, test_y = remove_zero_len_tokens(test_X, test_y)

train_encodings = tokenizer(train_X, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
val_encodings = tokenizer(val_X, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
test_encodings = tokenizer(test_X, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)


def encode_tags(tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)
        doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())
    return encoded_labels


train_labels = encode_tags(train_y, train_encodings)
val_labels = encode_tags(val_y, val_encodings)
test_labels = encode_tags(test_y, test_encodings)

train_encodings.pop('offset_mapping')
val_encodings.pop('offset_mapping')
test_encodings.pop('offset_mapping')


class NER_Dataset(torch.utils.data.Dataset):
    def __init__(self, inp, labels):
        self.inp = inp
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.inp.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = NER_Dataset(train_encodings, train_labels)
val_dataset = NER_Dataset(val_encodings, val_labels)
test_dataset = NER_Dataset(test_encodings, test_labels)

config_kwargs = {
    'num_labels': num_labels,
    'cache_dir': cache_dir,
    'revision': model_revision,
    'use_auth_token': None,
}
config = AutoConfig.from_pretrained(model_name, **config_kwargs)


def compute_metrics(predictions,labels):
    
    true_predictions = [
        [id2tag[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2tag[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    preds, labs = [], []
    for pred, lab in zip(true_predictions, true_labels):
        preds.extend(pred)
        labs.extend(lab)
    assert(len(preds) == len(labs))
    labels_and_predictions = [" ".join([str(i), labs[i], preds[i]]) for i in range(len(labs))]
    counts = conlleval.evaluate(labels_and_predictions)
    scores = conlleval.get_scores(counts)
    results = {}
    macro_f1 = 0
    for k in eval_labels:
        if k in scores:
            results[k] = scores[k][-1]
        else:
            results[k] = 0.0
        macro_f1 += results[k]
    macro_f1 /= len(eval_labels)
    results['macro_f1'] = macro_f1 / 100
    results['micro_f1'] = conlleval.metrics(counts)[0].fscore
    return results


metric_for_best_model = 'macro_f1' if dataset_name[:4] == 'sofc' else 'micro_f1'
other_metric = 'micro_f1' if metric_for_best_model == 'macro_f1' else 'macro_f1'

best_lr = None
best_val = 0
best_val_acc_list = None
best_test_acc_list = None
best_val_oth_list = None
best_test_oth_list = None

if dataset_name == 'sofc':
    num_epochs = 20
elif dataset_name == 'sofc_slot':
    num_epochs = 40
elif dataset_name == 'matscholar':
    num_epochs = 15
else:
    raise NotImplementedError

model = AutoModelForTokenClassification.from_pretrained(
                model_name, from_tf=False, config=config,
                cache_dir=cache_dir, revision=model_revision, use_auth_token=None,
            )

device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
model.to(device)

train_batch_size = 32
test_batch_size = 64

train_dataloader = DataLoader(
            train_dataset,  
            sampler = RandomSampler(train_dataset), 
            batch_size = train_batch_size 
        )

test_dataloader = DataLoader(
            test_dataset, 
            sampler = SequentialSampler(test_dataset), 
            batch_size = test_batch_size 
        )

no_decay = ["bias", "LayerNorm.weight"]
last_layer = model.bert.encoder.layer[10:]
classifier = model.classifier
print(last_layer)

optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    }
]

# optimizer_grouped_parameters = [
#     {
#         "params": [p for n, p in classifier.named_parameters() if not any(nd in n for nd in no_decay)],
#         "weight_decay": args.weight_decay,
#     },
#     {
#         "params": [p for n, p in classifier.named_parameters() if any(nd in n for nd in no_decay)],
#         "weight_decay": 0.0,
#     },
#     {
#         "params": [p for n, p in last_layer.named_parameters() if not any(nd in n for nd in no_decay)],
#         "weight_decay": args.weight_decay,
#     },
#     {
#         "params": [p for n, p in last_layer.named_parameters() if any(nd in n for nd in no_decay)],
#         "weight_decay": 0.0,
#     },
# ]

optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lm_lrs)


total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

def train(model, train_dataloader, optimizer, scheduler):
    model.train()
    start_time = time.time()
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in list(batch.values()))
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[2],
                  'labels':         batch[3],
                 }
        outputs = model(**inputs)
        loss = outputs[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        model.zero_grad()
    end_time = time.time()
    print("training_time:{}".format(end_time-start_time))
    print("loss:{}".format(loss.item()))

def evaluate(model, test_dataloader):
    model.eval()
    predictions = []
    true_labels = []
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in list(batch.values()))
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[2],
                  'labels':         batch[3],
                 }
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs[1].detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)
    results = compute_metrics(predictions,true_labels)
    
    return results

for epoch in range(num_epochs):
    train(model, train_dataloader, optimizer, scheduler)
    result = evaluate(model, test_dataloader)
    
    print(result)

# 需要参考matscibert的computed——metric的实现