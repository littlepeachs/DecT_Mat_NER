import json
def label_frac(data_path, output_path_1):
    data = {}
    with open(data_path, 'r') as f:
        lines = f.readlines()
        
        sentence=[]
        sentence_tag=[]
        for line in lines:
            temp_list = line.rstrip('\n').split()
            if len(temp_list)!=2 and len(sentence)==0:
                continue
            if temp_list[1]!='O' and temp_list[1][2:] not in data:
                data[temp_list[1][2:]] = {}
                data[temp_list[1][2:]][temp_list[0]]=1
            elif temp_list[1]!='O' and temp_list[1][2:] in data:
                if temp_list[0] not in data[temp_list[1][2:]]:
                    data[temp_list[1][2:]][temp_list[0]]=1
                else:
                    data[temp_list[1][2:]][temp_list[0]]+=1
    with open(output_path_1, 'w') as wf:
        json.dump(data, wf)
        wf.write('\n')
            

if __name__ == '__main__':
    import os
    label_frac(f"datasets/matsciner/train.txt", f"datasets/matsciner/label_frac.json")