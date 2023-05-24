import random
import json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--shot', default=8, type=int)
args = parser.parse_args()


def sample_data(data_path, output_path_1,output_path_2, k=10):
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
            json.dump(row, wf)
            wf.write('\n')

    with open(output_path_2, 'w') as wf:
        for row in few_shot_data:
            for i in range(len(row["text"])):
                wf.write(row["text"][i]+" "+row['label'][i]+'\n')
            wf.write('\n')
            wf.write('\n')
    return label_cnt_dict



if __name__ == '__main__':
    import os
    
    for seed in [0,1,2,3,4]:
        random.seed(seed)
        os.system("mkdir datasets/matsciner/{}shot".format(args.shot))
        path = f"datasets/matsciner"
        sample_data(f"datasets/matsciner/train.txt", f"{path}/{args.shot}shot/{seed}.json",f"{path}/{args.shot}shot/{seed}.txt", k=args.shot)