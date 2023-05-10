import json
from transformers import AutoTokenizer



with open("scripts/matsciner/selected_verbalizer.json", 'r') as f:
    tokenizer = AutoTokenizer.from_pretrained("m3rg-iitd/matscibert")
    item = {}
    token_id_map = {}
    with open("scripts/matsciner/selected_verbalizer.json", 'r') as f:
        few_shot_data = []
        label_cnt_dict = {}
        data = f.readlines()
        
        
        for row in data:
            item = eval(row)
    keys = list(item.keys())
    for key in keys:
        token_id_map[key]=[]
        key_id_lists = tokenizer.convert_tokens_to_ids(item[key])
        for i in range(len(key_id_lists)):
            if key_id_lists[i]>=150 and len(token_id_map[key])<=8:
                token_id_map[key].append(item[key][i])
        # for i in range(len(key_id_lists)):
        #     if len(key_id_lists[i])==3 and key_id_lists[i][1]>=150 and len(token_id_map[key])<=7:
        #         token_id_map[key].append(item[key][i])
    
    
    with open("scripts/matsciner/final_verbalizer.json", 'w') as wf:
        # for row in few_shot_data:
        json.dump(token_id_map, wf)
        wf.write('\n')
