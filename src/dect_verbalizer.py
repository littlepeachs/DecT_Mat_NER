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
from openprompt.prompts import ManualVerbalizer
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, Seq2SeqLMOutput, MaskedLMOutput
import matplotlib.pyplot as plt
from sympy.matrices import Matrix, GramSchmidt
import numpy as np

class DecTVerbalizer(Verbalizer):
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
                 tokenizer: Optional[PreTrainedTokenizer],
                 classes: Optional[List] = None,
                 num_classes: Optional[Sequence[str]] = None,
                 label_words: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
                 prefix: Optional[str] = " ",
                 multi_token_handler: Optional[str] = "mean",
                 post_log_softmax: Optional[bool] = True,
                 lr: Optional[float] = 1e-3,
                 hidden_size: Optional[int] = 1024,
                 mid_dim: Optional[int] = 64,
                 epochs: Optional[int] = 5,
                 model_logits_weight: Optional[float] = 1,
                 device: Optional[int]=0,
                ):
        super().__init__(tokenizer=tokenizer, num_classes=num_classes, classes=classes)
        self.prefix = prefix
        self.multi_token_handler = multi_token_handler
        self.post_log_softmax = post_log_softmax
        self.lr = lr
        self.mid_dim = mid_dim
        self.epochs = epochs
        self.model_logits_weight = model_logits_weight
        self.device = device

        self.hidden_dims = hidden_size
        self.head = nn.Linear(self.hidden_dims, self.mid_dim, bias=False)
        if label_words is not None: # use label words as an initialization
            self.label_words = label_words
        w = torch.empty((self.num_classes, self.mid_dim))
        nn.init.xavier_uniform_(w)
        # w = self.orthogo_tensor(w)
        self.proto = nn.Parameter(w, requires_grad=False)
        r = torch.ones(self.num_classes)
        self.proto_r = nn.Parameter(r, requires_grad=True)
        self.optimizer = torch.optim.Adam(self.group_parameters_proto, lr=self.lr)
    
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

    def process_outputs(self, outputs: Union[torch.Tensor, torch.Tensor], batch: Union[Dict, InputFeatures], **kwargs):
        model_logits = self.process_logits(outputs[1])
        proto_logits = self.process_hiddens(outputs[0], model_logits)
        return proto_logits

    def gather_outputs(self, outputs: ModelOutput):
        logits = outputs.logits
        if isinstance(outputs, Seq2SeqLMOutput):
            ret = outputs.decoder_hidden_states[-1]
        elif isinstance(outputs, MaskedLMOutput) or isinstance(outputs, CausalLMOutputWithCrossAttentions):
            ret = outputs.hidden_states[-1]
        else:
            try:
                ret = outputs.hidden_states[-1]
            except AttributeError:
                raise NotImplementedError(f"Gather outputs method for outputs' type {type(outputs)} not implemented")

        return ret, logits

    @staticmethod
    def sim(x, y, r=0, model_logits=0, model_logits_weight=1):
        x = torch.unsqueeze(x, -2)
        dist = torch.norm((x - y), dim=-1) - model_logits * model_logits_weight - r
        return -dist
    
    def loss_func(self, x, model_logits, labels):
        sim_mat = torch.exp(self.sim(x, self.proto, self.proto_r, model_logits, self.model_logits_weight))
        pos_score = torch.sum(sim_mat * F.one_hot(labels), -1)
        loss = -torch.mean(torch.log(pos_score / sim_mat.sum(-1)))
        # loss+= self.pcl_loss(x,labels)
        return loss
    
    def dot_sim(self,x, y):
        norm_x = F.normalize(x, dim=-1)
        norm_y = F.normalize(y, dim=-1)
        return torch.matmul(norm_x, norm_y.t())
    
    def pcl_loss(self,v_ins,labels):
        # instance-prototype loss

        sim_mat = torch.exp(self.dot_sim(v_ins, self.proto))
        num = sim_mat.shape[0]
        loss = 0.
        # for i in range(num):
        #     pos_score = sim_mat[i][labels[i]]
        #     neg_score = (sim_mat[i].sum() - pos_score)
        #     loss += - torch.log(pos_score / (pos_score + neg_score)).sum()
        # loss = loss / (num)

        # instance-instance loss

        loss_ins = 0.
        for label in range(self.num_classes):
            classes_index = []

            for i in range(v_ins.shape[0]):
                if labels[i]==label:
                    classes_index.append(i)

            for i in classes_index:

                sim_instance = torch.exp(self.dot_sim(v_ins, v_ins[i]))
                pos_ins = sim_instance[classes_index]
                loss_ins += - torch.log(pos_ins / (sim_instance.sum(0))).sum()
        loss_ins = 3*loss_ins / (num * self.num_classes * num)
        loss = loss + loss_ins

        return loss
    
    def orthogo_tensor(self,tensor):
        m, n = tensor.size()
        x_np = tensor.t().numpy()
        matrix = [Matrix(col) for col in x_np.T]
        gram = GramSchmidt(matrix)
        ort_list = []
        for i in range(m):
            vector = []
            for j in range(n):
                vector.append(float(gram[i][j]))
            ort_list.append(vector)
        ort_list = np.mat(ort_list,dtype=np.float32)
        ort_list = torch.from_numpy(ort_list)
        ort_list = F.normalize(ort_list,dim=1)
        ort_list = ort_list*0.5
        return ort_list



    def train_proto(self, model, dataloader, calibrate_dataloader):
        model.eval()
        embeds = [[] for _ in range(self.num_classes)]
        labels = [[] for _ in range(self.num_classes)]
        model_logits = [[] for _ in range(self.num_classes)]
        total_num = 0
        start_time = time.time()
        with torch.no_grad():
            # collect calibration logits
            if calibrate_dataloader!=None:
                for i, batch in enumerate(calibrate_dataloader):
                    batch = batch.to("cuda:{}".format(self.device)).to_dict()
                    outputs = model.prompt_model(batch)
                    outputs = self.gather_outputs(outputs)
                    logits = self.project(model.extract_at_mask(outputs[1], batch))
                    self._calibrate_logits = logits / torch.mean(logits)

            # collect model logits and hidden states
            for i, batch in enumerate(dataloader):
                batch = batch.to("cuda:{}".format(self.device)).to_dict()
                outputs = model.prompt_model(batch)
                outputs = self.gather_outputs(outputs)
                hidden, logits = model.extract_at_mask(outputs[0], batch), model.extract_at_mask(outputs[1], batch)
                logits = self.process_logits(logits)
                total_num += len(hidden)
                for j in range(len(hidden)):
                    label = batch['label'][j]
                    labels[label].append(label)
                    embeds[label].append(hidden[j])
                    model_logits[label].append(logits[j])
            
        embeds = list(map(torch.stack, embeds))
        labels = torch.cat(list(map(torch.stack, labels)))
        model_logits = torch.cat(list(map(torch.stack, model_logits)))

        dist = list(map(lambda x: torch.norm(self.head(x) - self.head(x.mean(0)), dim=-1).mean(), embeds))
        self.proto_r.data = torch.stack(dist)

        loss = 0.
        loss_record = []
        
        for epoch in range(self.epochs):
            x = self.head(torch.cat(embeds))
            self.optimizer.zero_grad()
            loss = self.loss_func(x, model_logits, labels)
            loss.backward()
            loss_record.append(loss.item())
            self.optimizer.step()
        
        print("Total epoch: {}. DecT loss: {}".format(self.epochs, loss))
        end_time = time.time()
        print("Training time: {}".format(end_time - start_time))

        # x_value = range(1,self.epochs+1)
        # y_value = loss_record
        # plt.figure(figsize=(6,4))
        # plt.plot(x_value,y_value,color="red",linewidth=1 )
        # plt.xlabel("x") #xlabel、ylabel：分别设置X、Y轴的标题文字。
        # plt.ylabel("loss")
        # plt.title("损失函数") # title：设置子图的标题。
        # plt.savefig('loss.png',dpi=120,bbox_inches='tight')
