from argparse import ArgumentParser
from cProfile import label
from cmath import log
from curses import raw
from json import decoder
from logging import debug
from turtle import position
import pytorch_lightning as pl
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
# Hide lines below until Lab 5
import wandb
import numpy as np
# Hide lines above until Lab 5

from .base import BaseLitModel
from .util import f1_eval, compute_f1, acc, f1_score
from transformers.optimization import get_linear_schedule_with_warmup

from functools import partial

import random

import pdb

def mask_hook(grad_input, st, ed):
    mask = torch.zeros((grad_input.shape[0], 1)).type_as(grad_input)
    mask[st: ed] += 1.0  # 只优化id为1～8的token
    # for the speaker unused token12
    mask[1:3] += 1.0
    return grad_input * mask

def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()

# class AMSoftmax(nn.Module):
#     def __init__(self,
#                  in_feats,
#                  n_classes=10,
#                  m=0.35,
#                  s=30):
#         super(AMSoftmax, self).__init__()
#         self.m = m
#         self.s = s
#         self.in_feats = in_feats
#         self.W = torch.nn.Linear(in_feats, n_classes)
#         # self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
#         self.ce = nn.CrossEntropyLoss()
#         # nn.init.xavier_normal_(self.W, gain=1)

#     def forward(self, x, lb):
#         assert x.size()[0] == lb.size()[0]
#         assert x.size()[1] == self.in_feats
#         x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
#         x_norm = torch.div(x, x_norm)
#         w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
#         w_norm = torch.div(self.W, w_norm)
#         costh = torch.mm(x_norm, w_norm)
#         # print(x_norm.shape, w_norm.shape, costh.shape)
#         lb_view = lb.view(-1, 1)
#         if lb_view.is_cuda: lb_view = lb_view.cpu()
#         delt_costh = torch.zeros(costh.size()).scatter_(1, lb_view, self.m)
#         if x.is_cuda: delt_costh = delt_costh.cuda()
#         costh_m = costh - delt_costh
#         costh_m_s = self.s * costh_m
#         loss = self.ce(costh_m_s, lb)
#         return loss, costh_m_s

# class AMSoftmax(nn.Module):

#     def __init__(self, in_features, out_features, s=30.0, m=0.35):
#         '''
#         AM Softmax Loss
#         '''
#         super().__init__()
#         self.s = s
#         self.m = m
#         self.in_features = in_features
#         self.out_features = out_features
#         self.fc = nn.Linear(in_features, out_features, bias=False)

#     def forward(self, x, labels):
#         '''
#         input shape (N, in_features)
#         '''
#         assert len(x) == len(labels)
#         assert torch.min(labels) >= 0
#         assert torch.max(labels) < self.out_features
#         for W in self.fc.parameters():
#             W = F.normalize(W, dim=1)

#         x = F.normalize(x, dim=1)

#         wf = self.fc(x)
#         numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
#         excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
#         denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
#         L = numerator - torch.log(denominator)
#         return -torch.mean(L)

# from models.prefix_encoder import PrefixEncoder
# from transformers import RobertaConfig,RobertaModel
#from models.roberta.modeling_roberta_plus import RobertaConfig,RobertaModel
from .prompt_model import AOA_Prompt, PGN_Prompt, Prefix_Prompt,LSTM_Prompt,Attention_Prompt,Entity_Prompt,AOA_Prompt
class BertLitModel(BaseLitModel):
    """
    use AutoModelForMaskedLM, and select the output by another layer in the lit model
    """
    def __init__(self, model, args, tokenizer):
        super().__init__(model, args)
        self.tokenizer = tokenizer
        
        with open(f"{args.data_dir}/rel2id.json","r") as file:
            rel2id = json.load(file)
        
        Na_num = 0
        for k, v in rel2id.items():
            if k == "NA" or k == "no_relation" or k == "Other":
                Na_num = v
                break
        num_relation = len(rel2id)
        # init loss function
        self.loss_fn = multilabel_categorical_crossentropy if "dialogue" in args.data_dir else nn.CrossEntropyLoss()
        # self.loss_fn = AMSoftmax(self.model.config.hidden_size, num_relation)
        # ignore the no_relation class to compute the f1 score
        self.eval_fn = f1_eval if "dialogue" in args.data_dir else partial(f1_score, rel_num=num_relation, na_num=Na_num)
        self.best_f1 = 0
        self.t_lambda = args.t_lambda
        self.t_beta = 0.001
        
        self.label_st_id = tokenizer("[class1]", add_special_tokens=False)['input_ids'][0]
        self.tokenizer = tokenizer
        
        
        # with torch.no_grad():
        #     self.loss_fn.fc.weight = nn.Parameter(self.model.get_output_embeddings().weight[self.label_st_id:self.label_st_id+num_relation])
            # self.loss_fn.fc.bias = nn.Parameter(self.model.get_output_embeddings().bias[self.label_st_id:self.label_st_id+num_relation])


        """P-tuning v2"""
        # self.num_labels = 2
        # #self.bert = BertModel.from_pretrained("path", add_pooling_layer=False)#在modeling_roberta_plus中有定义
        # #self.path = "path"
        # #self.bert.load_state_dict(torch.load(self.path))#这个是读取已有的模型的
        # #self.qa_outputs = torch.nn.Linear(1024, self.num_labels)这个是QA专用的
        # #self.pre_seq_len = 15 #提示序列长度

        #下面这些参数都是在bert的config里面的！！！！！！！
        # self.n_layer = self.config.num_hidden_layers
        # self.n_head = self.config.num_attention_heads
        # self.n_embd = self.config.hidden_size // self.config.num_attention_heads

        # self.dropout = torch.nn.Dropout(0.3)#get_prompt中使用到
        # self.prefix_encoder = PrefixEncoder()
        # self.prefix_tokens = torch.arange(self.pre_seq_len).long()#torch.Size([15])

        # for param in self.bert.parameters():
        #     param.requires_grad=False
        self._init_label_word()

        """Seed label word"""
        self.manual_labelword = {v:k for k,v in rel2id.items()}
        for k,v in self.manual_labelword.items():
            if ("tacred" in args.data_dir) or ("tacrev" in args.data_dir) or ("retacred" in args.data_dir):
                v = v.split(":")[-1]
                v = v.replace("_"," ")
                v = v.replace("NA","irrelevant")
                self.manual_labelword[k] = v
            elif "semeval" in args.data_dir:
                v = v.split("(")[0]
                v = v.replace("-"," of ")
                v = v.replace("Other","irrelevant")
                self.manual_labelword[k] = v
        """verbalizer"""
        #self.label_emb = self.init_verbalizer()
        # dataset_name = args.data_dir.split("/")[1]#'semeval'
        # model_name_or_path = args.model_name_or_path.split("/")[-1]#'roberta-large'
        # label_path = f"./dataset/{model_name_or_path}_{dataset_name}.pt"#'./dataset/roberta-large_semeval.pt'
        # self.label_emb = torch.load(label_path)
        


        """P-tuning"""
        if self.args.prompt_model == "Attention_Prompt":
            self.prompt_model = Attention_Prompt(self.model)
        elif self.args.prompt_model == "LSTM_Prompt":
           self.prompt_model = LSTM_Prompt(self.model)
        elif self.args.prompt_model == "Prefix_Prompt":
            self.prompt_model = Prefix_Prompt(self.model)
        elif self.args.prompt_model == "Entity_Prompt":
            self.prompt_model = Entity_Prompt(self.model)
        elif self.args.prompt_model == "AOA_Prompt":
            self.prompt_model = AOA_Prompt(self.model)
        elif self.args.prompt_model == "PGN_Prompt":
            self.prompt_model = PGN_Prompt(self.model,self.label_emb)
        else:
            self.prompt_model = self.model
        #print(self.prompt_model)

        # self.num_labels = num_relation
        # self.classifier = torch.nn.Linear(self.model.roberta.config.hidden_size, self.num_labels)  ## 全连接层
        # with open(f"{args.data_dir}/seed.json","r") as file:
        #     self.label_seed = json.load(file)
        # seed = {value:key for key,value in rel2id.items()}


    def _init_label_word(self, ):
        args = self.args
        # ./dataset/dataset_name
        dataset_name = args.data_dir.split("/")[1]#'semeval'
        model_name_or_path = args.model_name_or_path.split("/")[-1]#'roberta-large'
        label_path = f"./dataset/{model_name_or_path}_{dataset_name}.pt"#'./dataset/roberta-large_semeval.pt'
        # [num_labels, num_tokens], ignore the unanswerable
        if "dialogue" in args.data_dir:
            label_word_idx = torch.load(label_path)[:-1]
        else:
            label_word_idx = torch.load(label_path)

        num_labels = len(label_word_idx)#19
        
        self.model.resize_token_embeddings(len(self.tokenizer))

        with torch.no_grad():
            word_embeddings = self.model.get_input_embeddings()
            continous_label_word = [a[0] for a in self.tokenizer([f"[class{i}]" for i in range(1, num_labels+1)], add_special_tokens=False)['input_ids']]#连续label word :[50269, 50270, 50271, 50272, 50273, 50274, 50275, 50276, 50277, 50278, 50279, 50280, 50281, 50282, 50283, 50284, 50285, 50286, 50287]
            
            # for abaltion study
            if self.args.init_answer_words:#是否对虚拟答案词进行初始化
                if self.args.init_answer_words_by_one_token:#只通过一个词来进行初始化
                    for i, idx in enumerate(label_word_idx):
                        word_embeddings.weight[continous_label_word[i]] = word_embeddings.weight[idx][-1]#只需要把当前的标签词（关系类别），替换continous_label_word对应的位置
                else:
                    for i, idx in enumerate(label_word_idx):
                        word_embeddings.weight[continous_label_word[i]] = torch.mean(word_embeddings.weight[idx], dim=0)#否则，就是对他们进行取平均(每个类别)
                        #word_embeddings.weight[continous_label_word[i]] = torch.mean(word_embeddings.weight[idx[(idx!=0).nonzero(as_tuple =False)]], dim=0)#否则，就是对他们进行取平均(每个类别)
                # word_embeddings.weight[continous_label_word[i]] = self.relation_embedding[i]

            self.label_emb = word_embeddings.weight[continous_label_word].cuda()
            if self.args.init_type_words:#是否初始化实体类型词
                so_word = [a[0] for a in self.tokenizer(["[obj]","[sub]"], add_special_tokens=False)['input_ids']]#[50289, 50288]
                meaning_word = [a[0] for a in self.tokenizer(["person","organization", "location", "date", "country"], add_special_tokens=False)['input_ids']]#[[5970], [17247, 1938], [41829], [10672], [12659]] -> [5970, 17247, 41829, 10672, 12659]

                for i, idx in enumerate(so_word):
                    word_embeddings.weight[so_word[i]] = torch.mean(word_embeddings.weight[meaning_word], dim=0)#区平均

        
        self.word2label = continous_label_word # a continous list [50269, 50270, 50271, 50272, 50273, 50274, 50275, 50276, 50277, 50278, 50279, 50280, 50281, 50282, 50283, 50284, 50285, 50286, 50287]

    def related_concept(self,concept):
        """从conceptnet中搜索相近的concept"""
        import requests
        result = list()
        response = requests.get(f"https://api.conceptnet.io/related/c/en/{concept}?filter=/c/en")
        obj = response.json()

        for obj in obj['related']:
            (key,weight)= obj['@id'].split("/")[-1],obj['weight']
            result.append((key,weight))
        return result

    def top_K_idx(self,data, k):
        #data = np.array(data)
        idx = data.argsort()[-k:][::-1]
        return idx


    ###########初始化真实标签词########################
    # def init_verbalizer(self,) -> dict:
    #     args = self.args
    #     dataset_name = args.data_dir.split("/")[1]#'semeval'
    #     model_name_or_path = args.model_name_or_path.split("/")[-1]#'roberta-large'
    #     #对每个标签词的平均词向量
    #     label_emb = []
    #     word_embeddings = self.model.get_input_embeddings()
    #     for cls_num,label in self.manual_labelword.items():
    #         label2id = self.tokenizer(label,add_special_tokens=False)["input_ids"]
    #         label_vec = torch.mean(word_embeddings.weight[label2id], dim=0).cuda()
    #         label_emb.append(label_vec)
    #     label_emb = torch.stack(label_emb,dim=0).cuda()
    #     with open(f"./dataset/{model_name_or_path}_{dataset_name}_labelemb.pt", "wb") as file:
    #         torch.save(label_emb, file)
    #         print("save done!")
    #     import pdb
    #     pdb.set_trace()
        
        
        # #开始计算相似度
        # verbalizer = {}
        # for cls_num,label in self.manual_labelword.items():
        #     verbalizer[cls_num]= []
        #     related_dict = self.related_concept(label)#查询和当前label的相关概念
        #     related_score = []
        #     for concept_word,score in related_dict:
        #         concept2id = self.tokenizer(concept_word, add_special_tokens=False)["input_ids"]
        #         concept_emb = torch.mean(word_embeddings.weight[concept2id], dim=0)
                
        #         #把查询到的概念和每一个标签的词向量做cos相似度，并把分数和放入related_score,等待排序
        #         concept_sim_score = 0
        #         for lbs_emb in label_emb:
        #             cos_sim = torch.nn.functional.cosine_similarity(lbs_emb,concept_emb,dim =0)
        #             concept_sim_score+= cos_sim
        #         related_score.append(concept_sim_score)

        #     #取TopK个的index
        #     concept_idx = self.top_K_idx(related_score,min(5,len(related_score)))#取前5个
        #     for i in concept_idx:
        #         word = related_dict[i][0]
        #         fsc = self.similarity_score(label,concept_word,score)
        #         related_score.append(fsc)
        # return label_emb


    def forward(self, x):
        return self.prompt_model(x)


    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, labels, so ,so_type = batch

            
        if self.args.prompt_model == "KnowPrompt":
            result = self.model(input_ids, attention_mask, return_dict=True, output_hidden_states=True)
        elif self.args.prompt_model == "AOA_Prompt":
            result = self.prompt_model(input_ids, attention_mask, labels, so,so_type)
        elif self.args.prompt_model =="PGN_Prompt":
            result,label_weight = self.prompt_model(input_ids, attention_mask, labels, so)
        else:
            result = self.prompt_model(input_ids, attention_mask, labels, so)
        #print("input ids:",input_ids)
        #print("label:",labels)#tensor([ 1, 10,  2, 15,  0,  5, 11, 10,  0,  4,  5,  7,  2,  0, 11, 10],
        #print("so",so)
        logits = result.logits#torch.Size([16, 258, 50295]) 在词表上的logits
        output_embedding = result.hidden_states[-1]#torch.Size([16, 258, 1024]) 它也是一个元组，它的第一个元素是embedding，其余元素是各层的输出，每个元素的形状是(batch_size, sequence_length, hidden_size)
 
        if self.args.prompt_model == "PGN_Prompt":
            logits = self.pvpforseq(logits,input_ids,label_weight)
        else:
            logits = self.pvp(logits, input_ids)#在label上的logits
        # logits = self.model.roberta(input_ids, attention_mask).last_hidden_state
        # loss = self.get_loss(logits, input_ids, labels)

        ########################Contrastive loss######################################################
        mask_output_embedding = result.hidden_states[-1][:,5,:]#torch.Size([16, 1024])
        ct_loss = self.contrastive_loss(mask_output_embedding,labels,T=0.1)
        gen_loss = self.gen_loss(output_embedding)
   
        ke_loss = self.ke_loss(output_embedding, labels, so, input_ids)
        #loss = self.loss_fn(logits, labels) + self.t_lambda * ke_loss
        loss = self.loss_fn(logits,labels)+ self.t_lambda*ct_loss + self.t_beta*gen_loss
        self.log("Train/loss", loss)
        self.log("Train/ke_loss", ct_loss)
        self.log("Train/gen_loss",gen_loss)


        #CLS loss
        # # pooled_output = result.pooler_output #CLS输出 (batch_size, hidden_size)
        # output_embedding = result.hidden_states[-1]
        # pooled_output = output_embedding[:,0,:]#CLS输出
        # logits = self.classifier(pooled_output)#16,19
        # loss_fct = torch.nn.CrossEntropyLoss()
        # cls_loss = loss_fct(logits, labels)
        # self.log("Train/cls_loss", cls_loss)
        # loss = self.loss_fn(logits, labels) + self.t_lambda * cls_loss
        return loss

    def get_loss(self, logits, input_ids, labels):
        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bs = input_ids.shape[0]
        mask_output = logits[torch.arange(bs), mask_idx]
        
        loss = self.loss_fn(mask_output, labels)
        return loss


    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument

        input_ids, attention_mask, labels, _ ,so_type= batch

        if self.args.prompt_model == "KnowPrompt":
            result = self.model(input_ids, attention_mask, return_dict=True, output_hidden_states=True)
        elif self.args.prompt_model == "AOA_Prompt":
            result = self.prompt_model(input_ids, attention_mask, labels, _,so_type)
        elif self.args.prompt_model =="PGN_Prompt":
            result,label_weight = self.prompt_model(input_ids, attention_mask, labels, _)
        else:
            result = self.prompt_model(input_ids, attention_mask, labels, _)
        logits = result.logits#
        # logits = self.model.roberta(input_ids, attention_mask).last_hidden_state
        # loss = self.loss_fn(logits, labels)
       
        if self.args.prompt_model == "PGN_Prompt":
            logits = self.pvpforseq(logits,input_ids,label_weight)
        else:
            logits = self.pvp(logits, input_ids)#在label上的logits

        loss = self.loss_fn(logits, labels)
        self.log("Eval/loss", loss)

        #CLS loss
        # # pooled_output = result.pooler_output #CLS输出 (batch_size, hidden_sizeoutput_embedding = result.hidden_states[-1]ooled_output = output_embedding[:,0,:]#CLS输出
        # output_embedding = result.hidden_states[-1]
        # pooled_output = output_embedding[:,0,:]#CLS输出
        # logits = self.classifier(pooled_output)*10 + logits
        # loss_fct = torch.nn.CrossEntropyLoss()
        # cls_loss = loss_fct(logits, labels)
        # self.log("Eval/cls_loss", cls_loss)
        # loss = self.loss_fn(logits, labels) + self.t_lambda * cls_loss

        return {"eval_logits": logits.detach().cpu().numpy(), "eval_labels": labels.detach().cpu().numpy()}
    
    def validation_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["eval_logits"] for o in outputs])
        labels = np.concatenate([o["eval_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Eval/f1", f1)
        if f1 > self.best_f1:
            self.best_f1 = f1
        self.log("Eval/best_f1", self.best_f1, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, labels, _ ,so_type = batch

        if self.args.prompt_model == "KnowPrompt":
            result = self.model(input_ids, attention_mask, return_dict=True, output_hidden_states=True)
        elif self.args.prompt_model == "AOA_Prompt":
            result = self.prompt_model(input_ids, attention_mask, labels, _,so_type)
        elif self.args.prompt_model =="PGN_Prompt":
            result,label_weight = self.prompt_model(input_ids, attention_mask, labels, _)
        else:
            result = self.prompt_model(input_ids, attention_mask, labels, _)
        logits = result.logits


        if self.args.prompt_model == "PGN_Prompt":
            logits = self.pvpforseq(logits,input_ids,label_weight)
        else:
            logits = self.pvp(logits, input_ids)#在label上的logits


        #CLS loss
        # # pooled_output = result.pooler_output #CLS输出 (batch_size, hidden_sizeoutput_embedding = result.hidden_states[-1]ooled_output = output_embedding[:,0,:]#CLS输出
        # output_embedding = result.hidden_states[-1]
        # pooled_output = output_embedding[:,0,:]#CLS输出
        # logits = self.classifier(pooled_output)*10 + logits

        return {"test_logits": logits.detach().cpu().numpy(), "test_labels": labels.detach().cpu().numpy()}

    def test_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["test_logits"] for o in outputs])
        labels = np.concatenate([o["test_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Test/f1", f1)
        

    @staticmethod
    def add_to_argparse(parser):
        BaseLitModel.add_to_argparse(parser)
        parser.add_argument("--t_lambda", type=float, default=0.01, help="")
        parser.add_argument("--t_gamma", type=float, default=0.3, help="")
        return parser
        
    def pvp(self, logits, input_ids):
        # convert the [batch_size, seq_len, vocab_size] => [batch_size, num_labels]
        #! hard coded
        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)#tensor([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 5], device='cuda:0')
        bs = input_ids.shape[0]#16
        mask_output = logits[torch.arange(bs), mask_idx]#torch.Size([16, 50295]) 应该是每个mask在vocabulary中的分布
        assert mask_idx.shape[0] == bs, "only one mask in sequence!"
        final_output = mask_output[:,self.word2label]#在label word上的一个分布
        
        #refix
        # pdb.set_trace()
        # values,indices =  mask_output.topk(10,dim=1,largest=True, sorted=True)
        return final_output

    def pvpforseq(self, logits, input_ids,label_weight):
        # convert the [batch_size, seq_len, vocab_size] => [batch_size, num_labels]
        #! hard coded
        bs = input_ids.shape[0]#16
        mask_idx = torch.LongTensor([5 for i in range(int(bs))]).cuda()#tensor([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 5], device='cuda:0')
        bs = input_ids.shape[0]#16
  
        mask_output = logits[torch.arange(bs), mask_idx]#torch.Size([16, 50295]) 应该是每个mask在vocabulary中的分布
        assert mask_idx.shape[0] == bs, "only one mask in sequence!"
  
        final_output = mask_output[:,self.word2label] + label_weight*100 #在label word上的一个分布
        
        #refix
        # values,indices =  mask_output.topk(10,dim=1,largest=True, sorted=True)
        return final_output

        
    def contrastive_loss(self,mask_output_embedding,labels,T=0.1):
        batch_size = mask_output_embedding.shape[0]# 16 tensor(16,1024)
        cls_num = self.label_emb.shape[0]#40 tensor(40,1024)
        bs_loss = []
        for i in range(batch_size):
            label_idx = labels[i]
            predict = mask_output_embedding[i].repeat(cls_num,1)
            positive = self.label_emb[label_idx]
            negative = self.label_emb
            pos_similarity = F.cosine_similarity(mask_output_embedding[i].unsqueeze(0),positive.unsqueeze(0)) /T
            neg_similarity = F.cosine_similarity(predict ,negative) /T

            a = torch.exp(neg_similarity)
            b = torch.sum(a,dim=0,keepdim=True)


            bs_loss.append(torch.log(pos_similarity/b))
        loss = torch.Tensor(bs_loss).mean()
        return loss

    def gen_loss(self,logits):
        bsz = logits.shape[0]
        loss_sum = 0
        f = torch.nn.LogSigmoid()
        for i in range(bsz):
            subject_embedding = logits[i, 0]#把输出的对应实体的位置求平均值
            object_embedding = logits[i, 10]
            subject_prompt_embedding = torch.mean(logits[i,1:4],dim=0)
            object_prompt_embedding = torch.mean(logits[i,6:10],dim=0)
            d1 = F.cosine_similarity(subject_embedding.unsqueeze(0),subject_prompt_embedding.unsqueeze(0))
            d2 = F.cosine_similarity(object_embedding.unsqueeze(0),object_prompt_embedding.unsqueeze(0))

            loss = -1.*f(self.args.t_gamma - d1) - f(self.args.t_gamma - d2)
            loss_sum+= loss
        result = loss_sum/bsz
        return result

    def ke_loss(self, logits, labels, so, input_ids):
        subject_embedding = []
        object_embedding = []
        neg_subject_embedding = []
        neg_object_embedding = []
        bsz = logits.shape[0]
        for i in range(bsz):
            subject_embedding.append(torch.mean(logits[i, so[i][0]:so[i][1]], dim=0))#把输出的对应实体的位置求平均值
            object_embedding.append(torch.mean(logits[i, so[i][2]:so[i][3]], dim=0))

            # random select the neg samples
            st_sub = random.randint(1, logits[i].shape[0] - 6)
            span_sub = random.randint(1, 5)
            st_obj = random.randint(1, logits[i].shape[0] - 6)
            span_obj = random.randint(1, 5)
            neg_subject_embedding.append(torch.mean(logits[i, st_sub:st_sub+span_sub], dim=0))
            neg_object_embedding.append(torch.mean(logits[i, st_obj:st_obj+span_obj], dim=0))#其实就是把实体向量换成其他几个向量的平均
            
        subject_embedding = torch.stack(subject_embedding)
        object_embedding = torch.stack(object_embedding)
        neg_subject_embedding = torch.stack(neg_subject_embedding)
        neg_object_embedding = torch.stack(neg_object_embedding)
        # trick , the relation ids is concated, 


        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        mask_output = logits[torch.arange(bsz), mask_idx]
        mask_relation_embedding = mask_output
        real_relation_embedding = self.model.get_output_embeddings().weight[labels+self.label_st_id]
        
        d_1 = torch.norm(subject_embedding + mask_relation_embedding - object_embedding, p=2) / bsz#分数函数：实体embedding+mask的关系embeiding-另一个实体的embedding，进行normalize
        d_2 = torch.norm(neg_subject_embedding + real_relation_embedding - neg_object_embedding, p=2) / bsz
        f = torch.nn.LogSigmoid()
        loss = -1.*f(self.args.t_gamma - d_1) - f(d_2 - self.args.t_gamma)
        
        return loss

    def configure_optimizers(self):
        no_decay_param = ["bias", "LayerNorm.weight"]
        # for param in self.prompt_model.model.roberta.parameters():
        #     param.requires_grad = False
 
        if not self.args.two_steps: 
            parameters = self.prompt_model.named_parameters()
        else:
            # model.bert.embeddings.weight
            parameters = [next(self.prompt_model.named_parameters())]
        # only optimize the embedding parameters
        optimizer_group_parameters = [
            {"params": [p for n, p in parameters if not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in parameters if any(nd in n for nd in no_decay_param)], "weight_decay": 0}
        ]

        
        optimizer = self.optimizer_class(optimizer_group_parameters, lr=self.lr, eps=1e-10)#初始1e-8
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.num_training_steps * 0.1, num_training_steps=self.num_training_steps)
        return {
            "optimizer": optimizer, 
            "lr_scheduler":{
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1,
            }
        }

class TransformerLitModelTwoSteps(BertLitModel):
    def configure_optimizers(self):
        no_decay_param = ["bais", "LayerNorm.weight"]
        optimizer_group_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay_param)], "weight_decay": 0}
        ]
        optimizer = self.optimizer_class(optimizer_group_parameters, lr=self.args.lr_2, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.num_training_steps * 0.1, num_training_steps=self.num_training_steps)
        return {
            "optimizer": optimizer, 
            "lr_scheduler":{
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1,
            }
        }



class DialogueLitModel(BertLitModel):

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, token_type_ids , labels = batch
        result = self.model(input_ids, attention_mask, token_type_ids, return_dict=True, output_hidden_states=True)
        logits = result.logits
        logits = self.pvp(logits, input_ids)
        loss = self.loss_fn(logits, labels) 
        self.log("Train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, token_type_ids , labels = batch
        logits = self.model(input_ids, attention_mask, token_type_ids, return_dict=True).logits
        logits = self.pvp(logits, input_ids)
        loss = self.loss_fn(logits, labels)
        self.log("Eval/loss", loss)
        return {"eval_logits": logits.detach().cpu().numpy(), "eval_labels": labels.detach().cpu().numpy()}
    
    def validation_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["eval_logits"] for o in outputs])
        labels = np.concatenate([o["eval_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Eval/f1", f1)
        if f1 > self.best_f1:
            self.best_f1 = f1
        self.log("Eval/best_f1", self.best_f1, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, token_type_ids , labels = batch
        logits = self.model(input_ids, attention_mask, token_type_ids, return_dict=True).logits
        logits = self.pvp(logits, input_ids)
        return {"test_logits": logits.detach().cpu().numpy(), "test_labels": labels.detach().cpu().numpy()}

    def test_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["test_logits"] for o in outputs])
        labels = np.concatenate([o["test_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Test/f1", f1)
        


    @staticmethod
    def add_to_argparse(parser):
        BaseLitModel.add_to_argparse(parser)
        parser.add_argument("--t_lambda", type=float, default=0.01, help="")
        return parser
        
    def pvp(self, logits, input_ids):
        # convert the [batch_size, seq_len, vocab_size] => [batch_size, num_labels]
        #! hard coded
        _, mask_idx = (input_ids == 103).nonzero(as_tuple=True)
        bs = input_ids.shape[0]
        mask_output = logits[torch.arange(bs), mask_idx]
        assert mask_idx.shape[0] == bs, "only one mask in sequence!"
        final_output = mask_output[:,self.word2label]
        
        return final_output
        
def decode(tokenizer, output_ids):
    return [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in output_ids]

class GPTLitModel(BaseLitModel):
    def __init__(self, model, args , data_config):
        super().__init__(model, args)
        # self.num_training_steps = data_config["num_training_steps"]
        self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fn = multilabel_categorical_crossentropy
        self.best_f1 = 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, cls_idx , labels = batch
        logits = self.model(input_ids, attention_mask=attention_mask, mc_token_ids=cls_idx)
        if not isinstance(logits, torch.Tensor):
            logits = logits.mc_logits

        loss = self.loss_fn(logits, labels)
        self.log("Train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, cls_idx , labels = batch
        logits = self.model(input_ids, attention_mask=attention_mask, mc_token_ids=cls_idx)
        if not isinstance(logits, torch.Tensor):
            logits = logits.mc_logits
        loss = self.loss_fn(logits, labels)
        self.log("Eval/loss", loss)
        return {"eval_logits": logits.detach().cpu().numpy(), "eval_labels": labels.detach().cpu().numpy()}
    
    def validation_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["eval_logits"] for o in outputs])
        labels = np.concatenate([o["eval_labels"] for o in outputs])

        # f1 = compute_f1(logits, labels)["f1"]
        f1 = f1_score(logits, labels)
        self.log("Eval/f1", f1)
        if f1 > self.best_f1:
            self.best_f1 = f1
        self.log("Eval/best_f1", self.best_f1, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argumenT
        input_ids, attention_mask, cls_idx , labels = batch
        logits = self.model(input_ids, attention_mask=attention_mask, mc_token_ids=cls_idx)
        if not isinstance(logits, torch.Tensor):
            logits = logits.mc_logits
        return {"test_logits": logits.detach().cpu().numpy(), "test_labels": labels.detach().cpu().numpy()}

    def test_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["test_logits"] for o in outputs])
        labels = np.concatenate([o["test_labels"] for o in outputs])

        f1 = f1_score(logits, labels)
        # f1 = acc(logits, labels)
        self.log("Test/f1", f1)

from models.trie import get_trie
class BartRELitModel(BaseLitModel):
    def __init__(self, model, args, tokenizer=None):
        super().__init__(model, args)
        self.best_f1 = 0
        self.first = True

        with open(f"{args.data_dir}/rel2id.json","r") as file:
            rel2id = json.load(file)

        Na_num = 0
        for k, v in rel2id.items():
            if k == "NA" or k == "no_relation" or k == "Other":
                Na_num = v
                break
        num_relation = len(rel2id)
        # init loss function
        self.loss_fn = multilabel_categorical_crossentropy if "dialogue" in args.data_dir else nn.CrossEntropyLoss()
        # ignore the no_relation class to compute the f1 score
        self.eval_fn = f1_eval if "dialogue" in args.data_dir else partial(f1_score, rel_num=num_relation, na_num=Na_num)
        
        self.tokenizer = tokenizer
        self.trie, self.rel2id = get_trie(args, tokenizer=tokenizer)
        
        self.decode = partial(decode, tokenizer=self.tokenizer)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        real_label  = batch.pop("label")
        loss = self.model(**batch).loss
        self.log("Train/loss", loss)
        return loss
        
        

    def validation_step(self, batch, batch_idx):
        real_label = batch.pop("label")
        labels = batch.pop("labels")
        batch.pop("decoder_input_ids")
        topk = 1
        outputs = self.model.generate(**batch, 
            prefix_allowed_tokens_fn=lambda batch_id, sent: self.trie.get(sent.tolist()),
            num_beams=topk, num_return_sequences=topk,
            output_scores=True,
            min_length=0,
            max_length=32,
        ).cpu()
        # calculate the rank in the decoder output 

        pad_id = self.tokenizer.pad_token_id
        outputs = self.decode(output_ids=outputs)
        labels = self.decode(output_ids=labels)
        
        preds = torch.tensor([self.rel2id[o] for o in outputs])
        true = real_label


        return {"eval_logits": preds.detach().cpu().numpy(), "eval_labels": true.detach().cpu().numpy()}


    
    def validation_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["eval_logits"] for o in outputs])
        labels = np.concatenate([o["eval_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Eval/f1", f1)
        if f1 > self.best_f1 and not self.first:
            self.best_f1 = f1
        self.first = False
        self.log("Eval/best_f1", self.best_f1, prog_bar=True, on_epoch=True)
       

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        real_label = batch.pop("label")
        labels = batch.pop("labels")
        batch.pop("decoder_input_ids")
        topk = 1
        outputs = self.model.generate(**batch, 
            prefix_allowed_tokens_fn=lambda batch_id, sent: self.trie.get(sent.tolist()),
            num_beams=topk, num_return_sequences=topk,
            output_scores=True,
            min_length=0,
            max_length=32,
        ).cpu()
        # calculate the rank in the decoder output 

        pad_id = self.tokenizer.pad_token_id
        outputs = self.decode(output_ids=outputs)
        labels = self.decode(output_ids=labels)
        
        preds = torch.tensor([self.rel2id[o] for o in outputs])
        true = real_label


        return {"test_logits": preds.detach().cpu().numpy(), "test_labels": true.detach().cpu().numpy()}

    def test_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["test_logits"] for o in outputs])
        labels = np.concatenate([o["test_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Test/f1", f1)
        

    def configure_optimizers(self):
        no_decay_param = ["bias", "LayerNorm.weight"]

        optimizer_group_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay_param)], "weight_decay": 0}
        ]

        optimizer = self.optimizer_class(optimizer_group_parameters, lr=self.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.num_training_steps * 0.1, num_training_steps=self.num_training_steps)
        return {
            "optimizer": optimizer, 
            "lr_scheduler":{
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1,
            }
        }
