import torch
from transformers import BertTokenizer
 
 
MODELNAME="bert-base-uncased"
 
tokenizer=BertTokenizer.from_pretrained(MODELNAME)
text = "Hello, world!"
tokens = tokenizer(text)
# outputs = tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=True)  #add_special_tokens=True 添加 [cls] [sep]等标志
# token_span=outputs["offset_mapping"]
print(tokens)
 
'''
offset_mapping  记录的是tokenizer后的token与原来的关系
'''