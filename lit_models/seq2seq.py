import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import AutoConfig,AutoModel,AutoTokenizer
import numpy as np

class Encoder(nn.Module):
    def __init__(self, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_embeddings): 
        '''
        input_embeddings = [batch_size, length, dim]
        '''
        embedded = input_embeddings.transpose(0, 1) # embedded = [src_len, batch_size, emb_dim]
        
        # enc_output = [src_len, batch_size, hid_dim * num_directions]
        # enc_hidden = [n_layers * num_directions, batch_size, hid_dim]
        enc_output, enc_hidden = self.rnn(embedded) # if h_0 is not give, it will be set 0 acquiescently

        # enc_hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # enc_output are always from the last layer
        
        # enc_hidden [-2, :, : ] is the last of the forwards RNN 
        # enc_hidden [-1, :, : ] is the last of the backwards RNN
        
        # initial decoder hidden is final hidden state of the forwards and backwards 
        # encoder RNNs fed through a linear layer
        # s = [batch_size, dec_hid_dim]
        s = torch.tanh(self.fc(torch.cat((enc_hidden[-2,:,:], enc_hidden[-1,:,:]), dim = 1)))
        
        return enc_output, s
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim, bias=False)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, s, enc_output):
        
        # s = [batch_size, dec_hid_dim]
        # enc_output = [src_len, batch_size, enc_hid_dim * 2]
        
        batch_size = enc_output.shape[1]
        src_len = enc_output.shape[0]
        
        # repeat decoder hidden state src_len times
        # s = [batch_size, src_len, dec_hid_dim]
        # enc_output = [batch_size, src_len, enc_hid_dim * 2]
        s = s.unsqueeze(1).repeat(1, src_len, 1)
        enc_output = enc_output.transpose(0, 1)
        
        # energy = [batch_size, src_len, dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim = 2)))
        
        # attention = [batch_size, src_len]
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.dec_hid_dim = dec_hid_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        self.d_ = nn.Linear(dec_hid_dim,1)
        self.e_ = nn.Linear(emb_dim,1)
        #self.c_ = nn.Linear(enc_hid_dim*2,1)
        
    def forward(self, dec_input, s, enc_output,input_embedding,input_weight):

        # dec_input = [batch_size]
        # s = [batch_size, dec_hid_dim]
        # enc_output = [src_len, batch_size, enc_hid_dim * 2]
        
        # dec_input = dec_input.unsqueeze(1) # dec_input = [batch_size, 1]
        # embedded = self.dropout(self.embedding(dec_input)).transpose(0, 1) # embedded = [1, batch_size, emb_dim]
        embedded = dec_input.unsqueeze(0)
        
        # a = [batch_size, 1, src_len]  
        a = self.attention(s, enc_output).unsqueeze(1)
        
        # enc_output = [batch_size, src_len, enc_hid_dim * 2]
        enc_output = enc_output.transpose(0, 1)

        # c = [1, batch_size, enc_hid_dim * 2]
        c = torch.bmm(a, enc_output).transpose(0, 1)

        # rnn_input = [1, batch_size, (enc_hid_dim * 2) + emb_dim]
        #rnn_input = torch.cat((embedded, c), dim = 2)
        rnn_input = torch.cat((embedded, c), dim = 2)
        
        # dec_output = [src_len(=1), batch_size, dec_hid_dim]
        # dec_hidden = [n_layers * num_directions, batch_size, dec_hid_dim]
        dec_output, dec_hidden = self.rnn(rnn_input, s.unsqueeze(0))
        
        # embedded = [batch_size, emb_dim]
        # dec_output = [batch_size, dec_hid_dim]
        # c = [batch_size, enc_hid_dim * 2]
        embedded = embedded.squeeze(0)
        dec_output = dec_output.squeeze(0)
        c = c.squeeze(0)
        
        # pred = [batch_size, output_dim]
        # pred = self.fc_out(torch.cat((dec_output, c, embedded), dim = 1))
        
        #return pred, dec_hidden.squeeze(0)#output,s

        #p_gen
        d_o = self.d_(dec_output)
        #c_o = self.c_(c)
        e_o = self.e_(embedded.squeeze(0))
        p_gen = torch.sigmoid(d_o  + e_o)

        input_att = input_weight.unsqueeze(1)
        """试试input_embedding换成encoder_output"""
        weight_input = torch.bmm(input_att, input_embedding).squeeze(1)
        #weight_input = torch.bmm(a, input_embedding).squeeze(1)
        dec_output= (1-p_gen)*weight_input+ p_gen*dec_output
        return dec_output,dec_hidden.squeeze(0),a

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        #self.device = device
        
    def forward(self, input_embedding,trg_len,dec_input,input_weight):
        
        # src = [src_len, batch_size]
        # trg = [trg_len, batch_size]
        # teacher_forcing_ratio is probability to use teacher forcing
        
        #batch_size = src.shape[1]
        #trg_len = trg.shape[0]
        batch_size= input_embedding.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        # tensor to store decoder outputs
        # outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        outputs = torch.zeros(trg_len, batch_size, self.decoder.dec_hid_dim)
        
        # enc_output is all hidden states of the input sequence, back and forwards
        # s is the final forward and backward hidden states, passed through a linear layer
        enc_output, s = self.encoder(input_embedding)
                
        # first input to the decoder is the <sos> tokens
        #dec_input = torch.FloatTensor(np.random.randint(low=-10,high=10,size=(16,1024)))
        attention_list = []
        for t in range(1, trg_len):
            
            # insert dec_input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            dec_output, s ,a = self.decoder(dec_input, s, enc_output,input_embedding,input_weight)
            
            # place predictions in a tensor holding predictions for each token
            outputs[t] = dec_output
            attention_list.append(a)
            
            # decide if we are going to use teacher forcing or not
            #teacher_force = random.random() < teacher_forcing_ratio
            
            # get the highest predicted token from our predictions
            top1 = dec_output.argmax(1) 
            
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            dec_input =  dec_output
            #print(top1)
            attention_mean = torch.mean(torch.cat(attention_list,dim=1),dim=1)
        return outputs,attention_mean



if __name__ =="__main__":
    # INPUT_DIM = len(SRC.vocab)
    # OUTPUT_DIM = len(TRG.vocab)
    OUTPUT_DIM = 50295
    ENC_EMB_DIM = 1024
    DEC_EMB_DIM = 1024
    ENC_HID_DIM = 512
    DEC_HID_DIM = 1024
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1

    # attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    # enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    # dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

    # model = Seq2Seq(enc, dec, device).to(device)
    # TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    # criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.rand(16,256,1024)
    dec_input = torch.FloatTensor(np.random.randint(low=-10,high=10,size=(16,1024)))
    enc = Encoder(ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
    model = Seq2Seq(enc, dec )
    outputs,attention_mean = model(x,3,dec_input)
    import pdb
    pdb.set_trace()
    result = outputs.transpose(0,1)
    print(result.shape)
    