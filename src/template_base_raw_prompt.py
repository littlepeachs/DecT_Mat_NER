
import os
import sys
sys.path.append(".")
from sklearn.metrics import accuracy_score
import argparse
import csv
from re import template
from process_data import load_dataset
from dect_verbalizer import DecTVerbalizer
from dect_trainer import DecTRunner
from openprompt.prompts import ManualTemplate
from manual_verbalizer import ManualVerbalizer
from openprompt.pipeline_base import PromptForClassification
from openprompt.utils.reproduciblity import set_seed
from openprompt import PromptDataLoader
from openprompt.data_utils import FewShotSampler
from openprompt.plms import load_plm
from openprompt.data_utils.utils import InputExample
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch

parser = argparse.ArgumentParser("")

parser.add_argument("--model", type=str, default='bert', help="plm name")
parser.add_argument("--model_name_or_path", default='m3rg-iitd/matscibert', help="default load from Huggingface cache")
parser.add_argument("--shot", type=int, default=1, help="number of shots per class")
parser.add_argument("--seed", type=int, default=0, help="data sampling seed")
parser.add_argument("--template_id", type=int, default=0)
parser.add_argument("--dataset", type=str, default='matsciner')
parser.add_argument("--max_epochs", type=int, default=5, help="number of training epochs for DecT")
parser.add_argument("--batch_size", type=int, default=8, help="batch size for train and test")
parser.add_argument("--proto_dim", type=int, default=128, help="hidden dimension for DecT prototypes")
parser.add_argument("--model_logits_weight", type=float, default=1, help="weight factor (\lambda) for model logits")
parser.add_argument("--lr", default=3e-5, type=float, help="learning rate for DecT")
parser.add_argument("--calibration", type=int, default=1, help="whether using calibration,0 not use, 1 use")
parser.add_argument("--device", default=2, type=int, help="cuda device")

args = parser.parse_args()


def build_dataloader(dataset, template, verbalizer, tokenizer, tokenizer_wrapper_class, batch_size):
    dataloader = PromptDataLoader(
        dataset = dataset, 
        template = template, 
        verbalizer = verbalizer, 
        tokenizer = tokenizer, 
        tokenizer_wrapper_class=tokenizer_wrapper_class, 
        batch_size = batch_size,
    )

    return dataloader



def train(model,dataloader,val_dataloader):
    total_loss = 0.0
    best_acc = 0.0
    loss_func = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    for epoch in range(1,args.max_epochs+1):
        print("#### epoch:{} ####".format(epoch))
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            batch = batch.to("cuda:{}".format(args.device)).to_dict()
            outputs = model.prompt_model(batch).logits
            logits = model.extract_at_mask(outputs, batch)
            label_logits = model.verbalizer.process_logits(logits)
            label = batch["label"]
            loss = loss_func(label_logits,label)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        print("epoch loss: {}".format(loss))

        if val_dataloader!=None:
            val_acc = evaluate(model,val_dataloader)

            if val_acc>best_acc:
                torch.save(model,"/home/liwentao/Dec-Tuning-in-Mat/model.pt")

    print("Total epoch: {}. prompt loss: {}".format(epoch, loss))

def evaluate(model,dataloader):
    preds = []
    labels = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = batch.to("cuda:{}".format(args.device)).to_dict()
            outputs = model.prompt_model(batch).logits
            logits = model.extract_at_mask(outputs, batch)
            label_logits = model.verbalizer.process_logits(logits)
            label = batch["label"]
            pred = torch.argmax(label_logits,dim=-1)
            preds.extend(pred.tolist())
            labels.extend(label.tolist())
    score = accuracy_score(labels, preds)
    return score   


def main():
    # set hyperparameter
    data_path = args.dataset.split('-')[0]
    if data_path == "mnli":
        args.model_logits_weight = 1
    elif data_path == "fewnerd":
        args.model_logits_weight = 1/16
    else:
        args.model_logits_weight = 1/args.shot

    # load dataset. The valid_dataset can be None
    train_dataset, valid_dataset, test_dataset, Processor = load_dataset(args.dataset)

    # sample data
    sampler = FewShotSampler(
        num_examples_per_label = args.shot,
        also_sample_dev = True,
        num_examples_per_label_dev = args.shot)

    train_sampled_dataset, valid_sampled_dataset = sampler(
        train_dataset = train_dataset,
        valid_dataset = valid_dataset,
        seed = args.seed
    )

    set_seed(123)

    plm, tokenizer, model_config, plm_wrapper_class = load_plm(args.model, args.model_name_or_path)

    # define template and verbalizer
    # define prompt
    template = ManualTemplate(
        tokenizer=tokenizer).from_file(f"scripts/{data_path}/manual_template.txt", choice=args.template_id)

    verbalizer = ManualVerbalizer(
        tokenizer=tokenizer, 
        classes=Processor.labels,
        multi_token_handler='mean').from_file(f"scripts/{data_path}/manual_verbalizer.json")
    # load prompt’s pipeline model
    prompt_model = PromptForClassification(plm, template, verbalizer, freeze_plm = False).to(args.device)
           
    # process data and get data_loader
    train_dataloader = build_dataloader(train_sampled_dataset, template, verbalizer, tokenizer, plm_wrapper_class, args.batch_size) if train_dataset else None
    valid_dataloader = build_dataloader(valid_sampled_dataset, template, verbalizer, tokenizer, plm_wrapper_class, args.batch_size) if valid_dataset else None
    test_dataloader = build_dataloader(test_dataset, template, verbalizer, tokenizer, plm_wrapper_class, args.batch_size) if test_dataset else None

    train(prompt_model,train_dataloader,valid_dataloader)
    if valid_dataloader!=None:
        prompt_model = torch.load("/home/liwentao/Dec-Tuning-in-Mat/model.pt")
        prompt_model.to(args.device)    
    
    score = evaluate(prompt_model,test_dataloader)
    print("#### TEST ####")
    print("score:{}".format(score))

if __name__ == "__main__":
    main()
    
