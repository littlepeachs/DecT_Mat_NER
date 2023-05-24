import random
import json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--shot', default=4, type=int)
args = parser.parse_args()



def transfer2json(dataset_type):
    data_path="datasets/matsciner/{}.txt".format(dataset_type)
    output_path="datasets/matsciner/{}.json".format(dataset_type)
    data = []
    with open(data_path, 'r') as f:
        lines = f.readlines()
        
        sentence=[]
        sentence_tag=[]
        for line in lines:
            element = {}
            temp_list = line.rstrip('\n').split()
            if len(temp_list)!=2 and len(sentence)==0:
                continue
            if len(temp_list)!=2 and len(sentence)!=0:
                element['text']=sentence
                element['label']=sentence_tag
                data.append(element)
                sentence=[]
                sentence_tag=[]
                continue
            sentence.append(temp_list[0])
            sentence_tag.append(temp_list[1])
    with open(output_path, 'w') as wf:
        for row in data:
            json.dump(row, wf)
            wf.write('\n')

def sample_verbal(data_path,output_path, label_map,k=10):
    with open(data_path, 'r') as f:
        data = f.readlines()
        random.shuffle(data)
        for row in data:
            item = eval(row)
            label = item['label']
            for i in range(len(label)):
                if label[i][0]=='B' and i <len(label)-2 and label[i+1][0]!='I':
                    if len(label_map[label[i][2:]])<=k and item['text'][i] not in label_map[label[i][2:]]:
                        label_map[label[i][2:]].append(item['text'][i])
    result={}
    labels = list(label_map.keys())
    for i in range(len(labels)):
        result["I-"+labels[i]]=label_map[labels[i]]
    
    with open(output_path, 'w') as wf:
        json.dump(result, wf)
        wf.write('\n')

def sample_data_txt(data_path, output_path_1, k=10):
    data = []
    with open(data_path, 'r') as f:
        lines = f.readlines()
        
        sentence=[]
        sentence_tag=[]
        for line in lines:
            element = {}
            temp_list = line.rstrip('\n').split()
            if len(temp_list)!=2 and len(sentence)==0:
                continue
            if len(temp_list)!=2 and len(sentence)!=0:
                element['text']=sentence
                element['label']=sentence_tag
                data.append(element)
                sentence=[]
                sentence_tag=[]
                continue
            sentence.append(temp_list[0])
            sentence_tag.append(temp_list[1])
    few_shot_data = []
    label_cnt_dict = {}
    random.shuffle(data)
    for item in data:
        label = item['label']
        if len(label) <= 10:
            continue
        is_add = True
        temp_cnt_dict = {}
        for l in label:
            if l.startswith('B-'):
                if l not in temp_cnt_dict:
                    temp_cnt_dict[l] = 0
                temp_cnt_dict[l] += 1
                if l not in label_cnt_dict:
                    label_cnt_dict[l] = 0
                if label_cnt_dict[l] + temp_cnt_dict[l] > k:
                    is_add = False
        if len(temp_cnt_dict)==0:
            is_add=False
        if is_add:
            few_shot_data.append(item)
            for key in temp_cnt_dict.keys():
                label_cnt_dict[key] += temp_cnt_dict[key]

    with open(output_path_1, 'w') as wf:
        for row in few_shot_data:
            for i in range(len(row["text"])):
                wf.write(row["text"][i]+" "+row['label'][i]+'\n')
            wf.write('\n')
            wf.write('\n')
    return label_cnt_dict

def sample_data_json(data_path, output_path,output_path_txt, k=10):
    with open(data_path, 'r') as f:
        few_shot_data = []
        label_cnt_dict = {}
        data = f.readlines()
        random.shuffle(data)
        for row in data:
            item = eval(row)
            label = item['label']
            if len(label) <= 10:
                continue
            is_add = True
            temp_cnt_dict = {}
            for l in label:
                if l.startswith('B-'):
                    if l not in temp_cnt_dict:
                        temp_cnt_dict[l] = 0
                    temp_cnt_dict[l] += 1
                    if l not in label_cnt_dict:
                        label_cnt_dict[l] = 0
                    if label_cnt_dict[l] + temp_cnt_dict[l] > k:
                        is_add = False
            if len(temp_cnt_dict)==0:
                is_add=False
            if is_add:
                few_shot_data.append(item)
                for key in temp_cnt_dict.keys():
                    label_cnt_dict[key] += temp_cnt_dict[key]
    with open(output_path, 'w') as wf:
        for row in few_shot_data:
            json.dump(row, wf)
            wf.write('\n')

    with open(output_path_txt, 'w') as wf:
        for row in few_shot_data:
            for i in range(len(row["text"])):
                wf.write(row["text"][i]+" "+row['label'][i]+'\n')
            wf.write('\n')
            wf.write('\n')
    return label_cnt_dict

if __name__ == '__main__':
    import os
    
    # for k in [1,2,4,8,16,32]:
    #     os.system("mkdir datasets/matsciner/{}shot".format(k))
    #     for seed in range(5):
    #         path = f"datasets/matsciner"

    #         sample_data_json(f"datasets/matsciner/train.json", f"{path}/{k}shot/{seed}.json",f"{path}/{k}shot/{seed}.txt", k=k)
    label_map = {
    "MAT": [],
    "SPL": [],
    "DSC": [],
    "PRO": [],
    "APL": [],
    "SMT": [],
    "CMT": []}
    k=100
    sample_verbal(f"datasets/matsciner/train.json",f"scripts/matsciner/selected_verbalizer.json",label_map,k)
