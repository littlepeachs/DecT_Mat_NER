from inspect import Parameter
import json
import time
from os import stat
from transformers.file_utils import ModelOutput
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.dummy_pt_objects import PreTrainedModel
from openprompt.data_utils import InputExample, InputFeatures
import re
from openprompt import Verbalizer
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, Seq2SeqLMOutput, MaskedLMOutput
from sympy.matrices import Matrix, GramSchmidt
import numpy as np

class DecT_NER_Trainer(Verbalizer):
    r"""
    The implementation of the verbalizer in `Prototypical Verbalizer for Prompt-based Few-shot Tuning`

    Args:   
        tokenizer (:obj:`PreTrainedTokenizer`): The tokenizer of the current pre-trained model to point out the vocabulary.
        classes (:obj:`List[Any]`): The classes (or labels) of the current task.
        label_words (:obj:`Union[List[str], List[List[str]], Dict[List[str]]]`, optional): The label words that are projected by the labels.
        prefix (:obj:`str`, optional): The prefix string of the verbalizer (used in PLMs like RoBERTa, which is sensitive to prefix space)
        multi_token_handler (:obj:`str`, optional): The handling strategy for multiple tokens produced by the tokenizer.
        post_log_softmax (:obj:`bool`, optional): Whether to apply log softmax post processing on label_logits. Default to True.
        lr: (:obj:`float`, optional): The learning rate for prototypes.
        hidden_size: (:obj:`int`, optional): The dimension of model hidden states.
        mid_dim: (:obj:`int`, optional): The dimension of prototype embeddings.
        epochs: (:obj:`int`, optional): The training epochs of prototypes.
        model_logits_weight: (:obj:`float`, optional): Weight factor (\lambda) for model logits.
    """
    def __init__(self, 
                 model,
                 verbalizer,
                 calibration,
                 tokenizer: Optional[PreTrainedTokenizer],
                 classes: Optional[List] = None,
                 
                 num_classes: Optional[Sequence[str]] = None,
                 label_words: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
                 prefix: Optional[str] = " ",
                 multi_token_handler: Optional[str] = "first",
                 post_log_softmax: Optional[bool] = True,
                 lr: Optional[float] = 1e-3,
                 hidden_size: Optional[int] = 4096,
                 mid_dim: Optional[int] = 64,
                 epochs: Optional[int] = 5,
                 model_logits_weight: Optional[float] = 1,
                 
                 device: Optional[int]=0,
                ):
        super().__init__(tokenizer=tokenizer, num_classes=num_classes, classes=classes)
        self.prefix = prefix
        self.calibration=calibration
        self.tokenizer = tokenizer
        self.multi_token_handler = multi_token_handler
        self.post_log_softmax = post_log_softmax
        self.lr = lr
        self.mid_dim = mid_dim
        self.epochs = epochs
        self.model_logits_weight = model_logits_weight
        self.device = torch.device("cuda:{}".format(device) if torch.cuda.is_available() else "cpu")
        self.model = model
        self.verbalizer =verbalizer
        self.label_words = list(verbalizer.values())
        self.label_words_id = [tokenizer.encode(self.label_words[i][0])[1] for i in range(len(self.label_words))]
        self.hidden_dims = hidden_size
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.lr = lr
        self.mid_dim = mid_dim
        self.epochs = epochs
        self.model_logits_weight = model_logits_weight

        self.reset_parameter()
        
    def reset_parameter(self):
        self.head = nn.Linear(self.hidden_dims, self.mid_dim, bias=False)
        w = torch.empty((self.num_classes, self.mid_dim)).to(self.device)
        nn.init.xavier_uniform_(w)
        self.proto = nn.Parameter(w, requires_grad=True)
        r = torch.ones(self.num_classes)
        self.proto_r = nn.Parameter(r, requires_grad=True)
        self.optimizer = torch.optim.Adam([p for n, p in self.head.named_parameters()] + [self.proto_r], lr=self.lr)
    
    @property 
    def group_parameters_proto(self,):
        r"""Include the last layer's parameters9
        """
        return [p for n, p in self.head.named_parameters()] + [self.proto_r] # +[self.proto]

    def on_label_words_set(self):
        self.label_words = self.add_prefix(self.label_words, self.prefix)
        self.generate_parameters()
        
    @staticmethod
    def add_prefix(label_words, prefix):
        r"""Add prefix to label words. For example, if a label words is in the middle of a template,
        the prefix should be ``' '``.

        Args:
            label_words (:obj:`Union[Sequence[str], Mapping[str, str]]`, optional): The label words that are projected by the labels.
            prefix (:obj:`str`, optional): The prefix string of the verbalizer.
        
        Returns:
            :obj:`Sequence[str]`: New label words with prefix.
        """
        new_label_words = []
        if isinstance(label_words[0], str):
            label_words = [[w] for w in label_words]  #wrapped it to a list of list of label words.

        for label_words_per_label in label_words:
            new_label_words_per_label = []
            for word in label_words_per_label:
                if word.startswith("<!>"):
                    new_label_words_per_label.append(word.split("<!>")[1])
                else:
                    new_label_words_per_label.append(prefix + word)
            new_label_words.append(new_label_words_per_label)
        return new_label_words

    def generate_parameters(self) -> List:
        r"""In basic manual template, the parameters are generated from label words directly.
        In this implementation, the label_words should not be tokenized into more than one token. 
        """
        all_ids = []
        for words_per_label in self.label_words:
            ids_per_label = []
            for word in words_per_label:
                ids = self.tokenizer.encode(word, add_special_tokens=False)
                ids_per_label.append(ids)
            all_ids.append(ids_per_label)

        max_len  = max([max([len(ids) for ids in ids_per_label]) for ids_per_label in all_ids])
        max_num_label_words = max([len(ids_per_label) for ids_per_label in all_ids])
        words_ids_mask = torch.zeros(max_num_label_words, max_len)
        words_ids_mask = [[[1]*len(ids) + [0]*(max_len-len(ids)) for ids in ids_per_label]
                             + [[0]*max_len]*(max_num_label_words-len(ids_per_label)) 
                             for ids_per_label in all_ids]
        words_ids = [[ids + [0]*(max_len-len(ids)) for ids in ids_per_label]
                             + [[0]*max_len]*(max_num_label_words-len(ids_per_label)) 
                             for ids_per_label in all_ids]
        
        words_ids_tensor = torch.tensor(words_ids)
        words_ids_mask = torch.tensor(words_ids_mask)
        self.label_words_ids = nn.Parameter(words_ids_tensor, requires_grad=False)
        self.words_ids_mask = nn.Parameter(words_ids_mask, requires_grad=False) # A 3-d mask
        self.label_words_mask = nn.Parameter(torch.clamp(words_ids_mask.sum(dim=-1), max=1), requires_grad=False)

    def process_hiddens(self, hiddens: torch.Tensor, model_logits, **kwargs):
        r"""A whole framework to process the original logits over the vocabulary, which contains four steps: 
        """
        proto_logits = self.sim(self.head(hiddens), self.proto, self.proto_r, model_logits, self.model_logits_weight)
        return proto_logits

    def project(self,
                logits: torch.Tensor,
                **kwargs,
                ) -> torch.Tensor:
        r"""
        Project the labels, the return value is the normalized (sum to 1) probs of label words. 
        
        Args:
            logits (:obj:`torch.Tensor`): The orginal logits of label words.
        
        Returns:
            :obj:`torch.Tensor`: The normalized logits of label words
        """

        label_words_logits = logits[:, self.label_words_ids]
        label_words_logits = self.handle_multi_token(label_words_logits, self.words_ids_mask)
        label_words_logits -= 10000*(1-self.label_words_mask)
        label_words_logits = torch.max(label_words_logits, dim=-1, keepdim=True)[0]
        return label_words_logits

    def process_logits(self, logits: torch.Tensor, **kwargs):
        r"""A whole framework to process the original logits over the vocabulary, which contains four steps: 

        (1) Project the logits into logits of label words

        if self.post_log_softmax is True:

            (2) Normalize over all label words

            (3) Calibrate (optional)

        (4) Aggregate (for multiple label words)

        Args:
            logits (:obj:`torch.Tensor`): The orginal logits.
        
        Returns:
            (:obj:`torch.Tensor`): The final processed logits over the labels (classes).
        """
        # project
        label_words_logits = self.project(logits, **kwargs)  #Output: (batch_size, num_classes) or  (batch_size, num_classes, num_label_words_per_label)

        
        if self.post_log_softmax:
            # normalize
            # label_words_probs = self.normalize(label_words_logits)

            # calibrate
            if  hasattr(self, "_calibrate_logits") and self._calibrate_logits is not None:
                label_words_logits = self.calibrate(label_words_probs=label_words_logits)

            # convert to logits
            # label_words_logits = torch.log(label_words_probs+1e-15)

        # aggreate
        label_logits = self.aggregate(label_words_logits)
        return label_logits
    
    def normalize(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Given logits regarding the entire vocabulary, return the probs over the label words set.
        
        Args:
            logits (:obj:`Tensor`): The logits over the entire vocabulary.

        Returns:
            :obj:`Tensor`: The logits over the label words set.
        
        """
        batch_size = logits.shape[0]
        return F.softmax(logits.reshape(batch_size, -1), dim=-1).reshape(*logits.shape)


    def aggregate(self, label_words_logits: torch.Tensor) -> torch.Tensor:
        r"""Use weight to aggregate the logits of label words.

        Args:
            label_words_logits(:obj:`torch.Tensor`): The logits of the label words.
        
        Returns:
            :obj:`torch.Tensor`: The aggregated logits from the label words. 
        """
        label_words_logits = (label_words_logits * self.label_words_mask).sum(-1)/self.label_words_mask.sum(-1)
        return label_words_logits

    def calibrate(self, label_words_probs: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        
        Args:
            label_words_probs (:obj:`torch.Tensor`): The probability distribution of the label words with the shape of [``batch_size``, ``num_classes``, ``num_label_words_per_class``]
        
        Returns:
            :obj:`torch.Tensor`: The calibrated probability of label words.
        """
        shape = label_words_probs.shape
        calibrate_label_words_probs = self._calibrate_logits
        assert calibrate_label_words_probs.shape[1:] == label_words_probs.shape[1:] \
             and calibrate_label_words_probs.shape[0]==1, "shape not match"
        label_words_probs /= (calibrate_label_words_probs+1e-15)
        
        return label_words_probs

    @staticmethod
    def sim(x, y, r=0, model_logits=0, model_logits_weight=1):
        x = torch.unsqueeze(x, -2)
        dist = torch.norm((x - y), dim=-1) - model_logits * model_logits_weight - r
        return -dist
    
    def loss_func(self, x, model_logits, labels):
        sim_mat = torch.exp(self.sim(x, self.proto, self.proto_r, model_logits, self.model_logits_weight))
        sum_prob = sim_mat.sum(-1)
        loss = 0.0 
        for i in range(self.num_classes):
            mask_matrix = torch.zeros_like(labels)
            mask_matrix[labels == i] = 1
            label_prob_i = sim_mat[:,:,i]
            log_prob_i = torch.log(label_prob_i / sum_prob)*mask_matrix
            nonzero_indices = torch.nonzero(log_prob_i)
            nonzero_count = nonzero_indices.size(0)
            loss+= -log_prob_i.sum()/nonzero_count
        return loss
    
    # 得到对应verbalizer的logits
    def extract_logits(self, logits):
        label_words_logits =logits[:,:, self.label_words_id]
        label_words_logits = F.normalize(label_words_logits, p=1, dim=2)
        # label_words_logits = [F.normalize(torch.tensor([l[i] for i in self.label_words_id]), p=1, dim=0) ]
        return label_words_logits

    def inference(self, dataloader):
        logits, hiddens,labels = [], [], []
        for step, batch in enumerate(dataloader):
            ner_label = batch.pop('ori_labels', 'not found ner_labels')
            outputs = self.model(**batch,output_hidden_states=True)
            batch_logits, batch_hiddens,batch_label = outputs.logits, outputs.hidden_states[-1], ner_label
            logits.extend(F.softmax(self.extract_logits(batch_logits),dim=-1))
            # hiddens.extend([b for b in F.normalize(batch_hiddens,dim=-1)])
            hiddens.extend([b for b in batch_hiddens])
            labels.extend([b for b in batch_label])
        # print(logits)
        # self.model.cpu()
        return logits, hiddens,labels
    
    # 用于测试，取-3是指去最后一个token的输出，不包括句子结束符(这是因为最后两位是结束符，倒数第三位是最后一个token，期待输出下一个token)，zs表示是否使用zero-shot
    #将input_ids和attention_mask传入模型，得到logits和hidden_states
    #对于batch_logits，取出label_words_id对应的logits，得到label_words_logits
    #对于logits进行softmax，得到label_words_logits
    #如果是zero-shot,直接取argmax
    #否则先得到proto_logits,再取argmax
    # 并没有把batch_logits, batch_hiddens全部放在一个列表，而是对于每个批次得到结果，然后存在列表里面
    def test(self, dataloader,zs = False):
        total_size = 0
        res = 0
        for data in dataloader:
            # for d in data: print(self.model.inference(d["input"]))
            label = data[2]
            input = {"input_ids":data[0].to("cuda:{}".format(self.device)),"attention_mask":data[1].to("cuda:{}".format(self.device))}
            output = self.model(**input,output_hidden_states=True)
            batch_logits, batch_hiddens = output.logits[:,-3,:], output.hidden_states[-1].permute(1,0,2)[:,-3,:]
            logits = F.softmax(self.extract_logits(batch_logits))
            # batch_logits = torch.stack(self.extract_logits(batch_logits))
            # batch_hiddens = torch.stack([torch.tensor(b[-1][0]) for b in batch_hiddens])
            if  hasattr(self, "_calibrate_logits") and self._calibrate_logits is not None:
                logits = self.calibrate(label_words_probs=logits)

            if not zs:
                proto_logits = self.sim(self.head(F.normalize(batch_hiddens,dim=-1).float()), self.proto, self.proto_r, logits, self.model_logits_weight).cpu()
                # print(proto_logits)
                pred = torch.argmax(proto_logits, dim=-1).cpu().tolist()
            else:
                pred = torch.argmax(logits, dim=-1).cpu().tolist()
            res += sum([int(i==j) for i,j in zip(pred, label)])
            total_size+=len(label)
            # del input, output,batch_logits, batch_hiddens
            # torch.cuda.empty_cache()
        return res/total_size
    
    # 用于训练，取-3是指去最后一个token的输出，不包括句子结束符
    #首先进行文本校准
    #将input_ids和attention_mask传入模型，得到logits和hidden_states
    #对于batch_logits，取出label_words_id对应的logits，得到label_words_logits
    #剩下的做法和原来一样
    def run(self, train_dataloader, valid_dataloader,test_dataloader):
        if self.calibration:
            input = self.tokenizer(" ", return_tensors="pt").to(self.device)
            # input = {"input_ids":data[0].to("cuda:{}".format(self.device)),"attention_mask":data[1].to("cuda:{}".format(self.device))}
            output = self.model(**input)
            batch_logits = output.logits
            logits =F.softmax(self.extract_logits(batch_logits),dim=-1)
            logits = torch.mean(logits[0],dim=0)
            self._calibrate_logits = (logits / torch.mean(logits)).unsqueeze(0)
        
        # embeds = [[] for _ in range(self.num_classes)]
        # labels = [[] for _ in range(self.num_classes)]
        # model_logits = [[] for _ in range(self.num_classes)]
        total_num = 0
        start_time = time.time()
        with torch.no_grad():
            train_logits, train_embeds,train_labels = self.inference(train_dataloader)
            for i in range(len(train_logits)):
                if  hasattr(self, "_calibrate_logits") and self._calibrate_logits is not None:
                    train_logits[i] = self.calibrate(label_words_probs=train_logits[i])


        embeds = torch.stack(train_embeds,dim=0)
        #print(embeds)
        labels = torch.stack(train_labels,dim=0)
        model_logits = torch.stack(train_logits,dim=0)
        
        self.head.to(self.device)
        self.proto.to(self.device)
        self.proto_r.to(self.device)

        dist = []
        for i in range(self.num_classes):
            mask_matrix = torch.zeros_like(labels)
            mask_matrix[labels == i] = 1
            embeds_i = embeds * mask_matrix.unsqueeze(-1)
            embeds_i = torch.flatten(embeds_i, start_dim=0, end_dim=1)
            non_zero_cols = torch.any(embeds_i != 0, dim=1)
            # 找到所有非零列的索引
            non_zero_cols_idx = torch.nonzero(non_zero_cols).squeeze()
            # 选择非零列
            embeds_i = torch.index_select(embeds_i, dim=0, index=non_zero_cols_idx)

            dist.append(torch.norm(self.head(embeds_i) - self.head(embeds_i.mean(0)), dim=-1).mean())

        # dist = list(map(lambda x: torch.norm(self.head(x) - self.head(x.mean(0)), dim=-1).mean(), embeds))
        self.proto_r.data = torch.stack(dist)

        loss = 0.
        
        for epoch in range(self.epochs):
            x = self.head(embeds)
            self.optimizer.zero_grad()
            loss = self.loss_func(x, model_logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.head.parameters(),0.25)
            self.optimizer.step()
            print("Total epoch: {}. epoch loss: {}".format(epoch, loss.item()))
        print("Total epoch: {}. DecT loss: {}".format(epoch, loss))

        # del train_logits, train_embeds, embeds, labels, model_logits
        # torch.cuda.empty_cache()

        end_time = time.time()
        print("Training time: {}".format(end_time - start_time))
        score = self.test(test_dataloader)
        # res = sum([int(i==j) for i,j in zip(preds, labels)])/len(preds)
        return score
    # 先进行文本校准，然后进行zero-shot test
    
    
    
    def run_zs(self,test_dataloader):
        if self.calibrate_dataloader!=None:
            for idx,data in enumerate(self.calibrate_dataloader):
                input = {"input_ids":data[0].to("cuda:{}".format(self.device)),"attention_mask":data[1].to("cuda:{}".format(self.device))}
                output = self.model(**input)
                batch_logits = output.logits[:,-3,:]
                logits =F.softmax(self.extract_logits(batch_logits),dim=-1)
                self._calibrate_logits = logits / torch.mean(logits)
        score = self.test(test_dataloader,zs=True)
        # res = sum([int(i==j) for i,j in zip(preds, labels)])/len(preds)
        return score