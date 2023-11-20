from turtle import forward
import torch
import torch.nn as nn

import pdb


class Prefix_Prompt(torch.nn.Module):
    def __init__(self,model) -> None:
        super().__init__()
        self.model = model
        self.config = self.model.roberta.config
        #self.roberta = RobertaModel.from_pretrained("roberta-large")
        self.embeddings = self.model.roberta.embeddings
        self.pre_seq_len = 4
        self.dropout = torch.nn.Dropout(0.1)

        for param in self.model.roberta.parameters():
            param.requires_grad = False

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = torch.nn.Embedding(self.pre_seq_len, self.config.hidden_size)


    def forward(self,input_ids,attention_mask,labels,so):
        batch_size = input_ids.shape[0]

        token_type_ids = None
        position_ids = None
        head_mask = None

        #embedding--------------------------------------------
        raw_embedding = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )#[batch_size, max_length, embedding_size]-> [16,256,1024]

        prompts = self.get_prompt(batch_size=batch_size)#torch.Size([16, 4, 1024])
        inputs_embeds = torch.cat((prompts, raw_embedding), dim=1)
        prefix_attention_mask = torch.ones(batch_size, prompts.shape[1]).type_as(attention_mask)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        result = self.model(
            #input_ids, 
            #attention_mask, 
            attention_mask = attention_mask,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
            return_dict=True, 
            output_hidden_states=True)
        return result

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.model.roberta.device)
        #prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1)
        prompts = self.prefix_encoder(prefix_tokens)
        return prompts

class LSTM_Prompt(torch.nn.Module):
    def __init__(self,model) -> None:
        super().__init__()
        self.model = model
        self.config = self.model.roberta.config
        self.embeddings = self.model.roberta.embeddings
        self.dropout = torch.nn.Dropout(0.1)

        for param in self.model.roberta.parameters():
                param.requires_grad = True
        
        self.encoder = torch.nn.LSTM(1024,512,batch_first=True,bidirectional=True,dropout=0.1)
        self.init_hidden = self.init_hidden()

    def forward(self,input_ids,attention_mask,labels,so):
        batch_size = input_ids.shape[0]

        token_type_ids = None
        position_ids = None
        head_mask = None

        #embedding--------------------------------------------
        raw_embedding = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )#[batch_size, max_length, embedding_size]-> [16,256,1024]

        #使用lstm生成prompt-------------------------------------
        prompt_embedding,(h,c) = self.encoder(raw_embedding)
        prompt_batch = []
        for index, example in enumerate(prompt_embedding):
            prompt1 = prompt_embedding[index][so[index][0]:so[index][1]].mean(dim=0).unsqueeze(0)
            prompt2 = prompt_embedding[index][so[index][2]:so[index][3]].mean(dim=0).unsqueeze(0)
            prompt = torch.cat((prompt1, prompt2), 0)
            prompt_batch.append(prompt)
        prompts = torch.stack(prompt_batch,0)#torch.Size([16, 2, 1024])
        inputs_embeds = torch.cat((prompts, raw_embedding), dim=1)
        prefix_attention_mask = torch.ones(batch_size, prompts.shape[1]).type_as(attention_mask)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        result = self.model(
            #input_ids, 
            #attention_mask, 
            attention_mask = attention_mask,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
            return_dict=True, 
            output_hidden_states=True)
        return result
    def init_hidden(self):
        return torch.zeros(1,1,self.hidden_size).cuda()

class Entity_Prompt(torch.nn.Module):
    def __init__(self,model) -> None:
        super().__init__()
        self.model = model
        self.config = self.model.roberta.config
        self.embeddings = self.model.roberta.embeddings
        self.dropout = torch.nn.Dropout(0.1)

        for param in self.model.roberta.parameters():
                param.requires_grad = True
        
    def forward(self,input_ids,attention_mask,labels,so):
        batch_size = input_ids.shape[0]

        token_type_ids = None
        position_ids = None
        head_mask = None

        #embedding--------------------------------------------
        raw_embedding = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )#[batch_size, max_length, embedding_size]-> [16,256,1024]

        prompt_batch = []
        for index, example in enumerate(raw_embedding):
            prompt1 = raw_embedding[index][so[index][0]:so[index][1]].mean(dim=0).unsqueeze(0)#torch.Size([1,1024])
            prompt2 = raw_embedding[index][so[index][2]:so[index][3]].mean(dim=0).unsqueeze(0)
            prompt = torch.cat((prompt1, prompt2), 0)#torch.Size([2,1024])
            prompt_batch.append(prompt)#16 * torch.Size([2,1024]))
        prompts = torch.stack(prompt_batch,0)#torch.Size([16, 2, 1024])
        inputs_embeds = torch.cat((prompts, raw_embedding), dim=1)#拼接prompt和input
        prefix_attention_mask = torch.ones(batch_size, prompts.shape[1]).type_as(attention_mask)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        result = self.model(
            #input_ids, 
            #attention_mask, 
            attention_mask = attention_mask,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
            return_dict=True, 
            output_hidden_states=True)
        return result

from .attention import ScaledDotProductAttention, SelfAttention
class Attention_Prompt(torch.nn.Module):
    def __init__(self,model) -> None:
        super().__init__()
        self.model = model
        self.config = self.model.roberta.config
        self.embeddings = self.model.roberta.embeddings
        #self.pre_seq_len = 2#prompt长度
        self.dropout = torch.nn.Dropout(0.1)

        for param in self.model.roberta.parameters():
                param.requires_grad = True

        self.self_attention = SelfAttention(n_head=8, d_k=1024, d_v=1024, d_x=1024, d_o=1024)
            
        self.FFN = nn.Sequential(
                    nn.Linear(1024,4096),
                    nn.ELU(),
                    nn.Linear(4096,1024),
                    nn.ELU()
                )

    def forward(self,input_ids,attention_mask,labels,so):
        pdb.set_trace()
        batch_size = input_ids.shape[0]

        token_type_ids = None
        position_ids = None
        head_mask = None

        #embedding--------------------------------------------
        raw_embedding = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )#[batch_size, max_length, embedding_size]-> [16,256,1024]
        mask = self.get_mask(attention_mask)#torch.Size([16, 256, 256])
        attn,prompt_embedding = self.self_attention(raw_embedding,mask=mask)#torch.Size([128, 256, 256]), torch.Size([16, 256, 1024])
        prompt_embedding = self.FFN(prompt_embedding)
        #使用lstm生成prompt-------------------------------------

        prompt_batch = []
        for index, example in enumerate(prompt_embedding):
            prompt1 = prompt_embedding[index][so[index][0]:so[index][1]].mean(dim=0).unsqueeze(0)
            prompt2 = prompt_embedding[index][so[index][2]:so[index][3]].mean(dim=0).unsqueeze(0)
            prompt = torch.cat((prompt1, prompt2), 0)
            prompt_batch.append(prompt)
        prompts = torch.stack(prompt_batch,0)#torch.Size([16, 2, 1024])
        inputs_embeds = torch.cat((prompts, raw_embedding), dim=1)
        prefix_attention_mask = torch.ones(batch_size, prompts.shape[1]).type_as(attention_mask)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        result = self.model(
            #input_ids, 
            #attention_mask, 
            attention_mask = attention_mask,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
            return_dict=True, 
            output_hidden_states=True)
        return result

    def get_mask(self,attention_mask):
        mask = []
        for i in range(attention_mask.shape[0]):
            att = torch.unsqueeze(attention_mask[i],0)
            m = att.t() * att
            mask.append(m.unsqueeze(0))
        mask = torch.cat(mask,0).bool()
        return mask


from .attention import MultiHeadAttention,ScaledDotProductAttention
import numpy as np
from axial_attention import AxialAttention,AxialPositionalEmbedding
class AOA_Prompt(torch.nn.Module):
    def __init__(self,model,label_emb) -> None:
        super().__init__()
        self.model = model
        self.config = self.model.roberta.config
        self.embeddings = self.model.roberta.embeddings
        #self.pre_seq_len = 2#prompt长度
        self.dropout = torch.nn.Dropout(0.1)

        self.word_embeddings = self.model.get_input_embeddings()

        for param in self.model.roberta.parameters():
                param.requires_grad = True

        self.mha = MultiHeadAttention(n_head=2, d_k_=1024, d_v_=1024, d_k=1024, d_v=1024, d_o=1024)

        self.dot_attn = ScaledDotProductAttention(scale=np.power(1024, 0.5))

        # self.axial_pos = AxialPositionalEmbedding(
        #     dim = 512,
        #     shape = (20, 20)
        # )#输入size：[1，521,20,20]

        self.axial_att = AxialAttention(
            dim = 3,               # embedding dimension
            dim_index = 1,         # where is the embedding dimension
            dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
            heads = 1,             # number of heads for multi-head attention
            num_dimensions = 2,    # number of axial dimensions (images is 2, video is 3, or more)
            sum_axial_out = True   # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
        )#输入size：[batch , dim , -1 ,-1]


        self.label_emb = label_emb

    def forward(self,input_ids,attention_mask,labels,so,so_type):
        batch_size = input_ids.shape[0]

        token_type_ids = None
        position_ids = None
        head_mask = None

        #embedding--------------------------------------------
        raw_embedding = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )#[batch_size, max_length, embedding_size]-> [16,256,1024]
        # mask = self.get_mask(attention_mask)#torch.Size([16, 256, 256])
        # mask = None
        # attn,prompt_embedding = self.self_attention(raw_embedding,mask=mask)#torch.Size([128, 256, 256]), torch.Size([16, 256, 1024])

        ent = []
        #取出实体和实体相关信息: entity_info
        for index, example in enumerate(input_ids):
            entity1 = input_ids[index][so[index][0]:so[index][1]]
            entity2 = input_ids[index][so[index][2]:so[index][3]]
            entity1_emb = torch.mean(self.word_embeddings.weight[entity1], dim=0)
            entity2_emb = torch.mean(self.word_embeddings.weight[entity2], dim=0)
            entity_emb = torch.stack([entity1_emb,entity2_emb],dim=0)
            
            idx = so_type[index][(so_type[index]!=1).nonzero(as_tuple=False)].view(1,-1).squeeze()
            type_emb = self.word_embeddings.weight[idx]

            entity_info = torch.cat([entity_emb,type_emb],dim=0)
            if 6-entity_info.shape[0] > 0:
                new = torch.mean(entity_info,dim=0).unsqueeze(dim=0)
                entity_info = torch.cat([entity_info]+ [new for i in range(6-entity_info.shape[0])],dim=0)
            try:
                assert entity_info.shape[0]==6
            except Exception as e:
                print(entity_info.shape)
            ent.append(entity_info)
        pdb.set_trace()
        ent_emb = torch.stack(ent,dim=0)
        attn, output = self.dot_attn(self.label_emb.repeat(16,1,1),ent_emb, ent_emb)
        # mask = self.get_mask(attention_mask)#torch.Size([16, 256, 256])
        attn, output = self.dot_attn(output, raw_embedding, raw_embedding, mask=None)
        # attn,prompt_embedding  = self.mha(self.label_emb.repeat(16,1,1),raw_embedding,raw_embedding)#torch.Size([32, 40, 256]), torch.Size([16, 41, 1024])
        #----------------------- Heatmap ----------------------
        # from .heatmap import get_heatmap
        # get_heatmap((attn[0,:,:60]).cpu().numpy().tolist())
        #------------------------------------------------------
        # attn = self.row_att(attn.unsqueeze(1)).squeeze(1)
        # prompts = torch.bmm(attn, raw_embedding)
        prompts = output
        """
        prompt_batch = []
        for index, example in enumerate(prompt_embedding):
            prompt1 = prompt_embedding[index][so[index][0]:so[index][1]].mean(dim=0).unsqueeze(0)
            prompt2 = prompt_embedding[index][so[index][2]:so[index][3]].mean(dim=0).unsqueeze(0)
            prompt = torch.cat((prompt1, prompt2), 0)
            prompt_batch.append(prompt)
        prompts = torch.stack(prompt_batch,0)#torch.Size([16, 2, 1024])
        """
        inputs_embeds = torch.cat((prompts, raw_embedding), dim=1)
        prefix_attention_mask = torch.ones(batch_size, prompts.shape[1]).type_as(attention_mask)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        result = self.model(
            #input_ids, 
            #attention_mask, 
            attention_mask = attention_mask,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
            return_dict=True, 
            output_hidden_states=True)
        return result

    def get_mask(self,attention_mask):
        mask = []
        for i in range(attention_mask.shape[0]):
            att = torch.unsqueeze(attention_mask[i],0)
            m = att.t() * att
            mask.append(m.unsqueeze(0))
        mask = torch.cat(mask,0).bool()
        return mask


class LSTM_Prompt(torch.nn.Module):
    def __init__(self,model) -> None:
        super().__init__()
        self.model = model
        self.config = self.model.roberta.config
        self.embeddings = self.model.roberta.embeddings
        self.dropout = torch.nn.Dropout(0.1)

        for param in self.model.roberta.parameters():
                param.requires_grad = True
        
        self.encoder = torch.nn.LSTM(1024,512,batch_first=True,bidirectional=True,dropout=0.1)
        self.init_hidden = self.init_hidden()

    def forward(self,input_ids,attention_mask,labels,so):
        batch_size = input_ids.shape[0]

        token_type_ids = None
        position_ids = None
        head_mask = None

        #embedding--------------------------------------------
        raw_embedding = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )#[batch_size, max_length, embedding_size]-> [16,256,1024]

        #使用lstm生成prompt-------------------------------------
        prompt_embedding,(h,c) = self.encoder(raw_embedding)
        prompt_batch = []
        for index, example in enumerate(prompt_embedding):
            prompt1 = prompt_embedding[index][so[index][0]:so[index][1]].mean(dim=0).unsqueeze(0)
            prompt2 = prompt_embedding[index][so[index][2]:so[index][3]].mean(dim=0).unsqueeze(0)
            prompt = torch.cat((prompt1, prompt2), 0)
            prompt_batch.append(prompt)
        prompts = torch.stack(prompt_batch,0)#torch.Size([16, 2, 1024])
        inputs_embeds = torch.cat((prompts, raw_embedding), dim=1)
        prefix_attention_mask = torch.ones(batch_size, prompts.shape[1]).type_as(attention_mask)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        result = self.model(
            #input_ids, 
            #attention_mask, 
            attention_mask = attention_mask,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
            return_dict=True, 
            output_hidden_states=True)
        return result
    def init_hidden(self):
        return torch.zeros(1,1,self.hidden_size).cuda()

from .seq2seq import Encoder,Decoder,Attention,Seq2Seq
from .aoa import AOA
class PGN_Prompt(torch.nn.Module):
    def __init__(self,model,label_emb) -> None:
        super().__init__()
        self.model = model
        self.config = self.model.roberta.config
        self.embeddings = self.model.roberta.embeddings
        self.dropout = torch.nn.Dropout(0.1)
        self.word_embeddings = self.model.get_input_embeddings()
        self.label_emb = label_emb
        #self.label_emb = torch.rand(16,40,1024).cuda()
        self.aoa = AOA(1024)

        #######PGN initial###############
        OUTPUT_DIM = 50295
        ENC_EMB_DIM = 1024
        DEC_EMB_DIM = 1024
        ENC_HID_DIM = 512
        DEC_HID_DIM = 1024
        ENC_DROPOUT = 0.1
        DEC_DROPOUT = 0.1

        enc = Encoder(ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
        attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
        dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
        self.pgn = Seq2Seq(enc, dec )



        for param in self.model.roberta.parameters():
                param.requires_grad = True
        
    def forward(self,input_ids,attention_mask,labels,so):
        batch_size = input_ids.shape[0]

        ################input embedding#########################
        raw_embedding = self.embeddings(
            input_ids=input_ids,
            position_ids=None,
            token_type_ids=None,
        )#[batch_size, max_length, embedding_size]-> [16,256,1024]
        ##############Attention on Attention#####################
        input_weight,label_weight = self.aoa(raw_embedding,self.label_emb.repeat(batch_size,1,1).cuda())
        ###############Entity embedding#########################
        head_entity = []
        tail_entity = []
        for index, example in enumerate(raw_embedding):
            head = raw_embedding[index][so[index][0]:so[index][1]].mean(dim=0).unsqueeze(0)#torch.Size([1,1024])
            tail = raw_embedding[index][so[index][2]:so[index][3]].mean(dim=0).unsqueeze(0)
            head_entity.append(head)
            tail_entity.append(tail)
        head_entity_emb = torch.cat(head_entity)
        tail_entity_emb = torch.cat(tail_entity)

        ################Pointer Generator Network################
        output_head,attention_head = self.pgn(raw_embedding,5,head_entity_emb,input_weight)#input_embedding, generate_prompt_len ,inital_token[16,1024] -> [16,5,1024], [16,256]
        output_tail,attention_tail = self.pgn(raw_embedding,5,tail_entity_emb,input_weight)#input_embedding, generate_prompt_len ,inital_token[16,1024] -> [16,5,1024], [16,256]
        
        ################组织prompt#################################
        mask_token_emb = self.word_embeddings.weight[50264].repeat(batch_size,1)
        try:
            prompts = torch.cat([head_entity_emb.unsqueeze(0),output_head.cuda()[1:],mask_token_emb.unsqueeze(0),output_tail.cuda()[1:],tail_entity_emb.unsqueeze(0)],dim=0).transpose(0,1)
        except:
            print(input_ids)
        ###############Input######################################
        inputs_embeds = torch.cat((prompts, raw_embedding), dim=1)#拼接prompt和input
        prefix_attention_mask = torch.ones(batch_size, prompts.shape[1]).type_as(attention_mask)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        result = self.model(
            #input_ids, 
            #attention_mask, 
            attention_mask = attention_mask,
            head_mask = None,
            inputs_embeds = inputs_embeds,
            return_dict=True, 
            output_hidden_states=True)
        return result,label_weight