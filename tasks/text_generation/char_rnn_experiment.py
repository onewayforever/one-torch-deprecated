#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import os
from torchtext import data
import pandas as pd
import re
from torchtext import vocab
import time
import random
import os
#from utils import log
from torch.utils.data import Dataset
import one_torch_utils as otu
import one_torch_models as otm

import torchtext.vocab as Vocab
import pickle



scale=0.99
delimiter=','
W2V_TXT_FILE="word2vec.txt"

max_input_length=32

text_col_idx = 4

print('max_input_length',max_input_length)

vocab_size=0
pad_idx=0

HPARAMS = {
    #"W2V_PATH":W2V_PATH,
    'HIDDEN_DIM': 512,
    'EMBED_DIM': 512,
    'N_LAYERS': 2,
    'BIDIRECTIONAL': False,
    'DROPOUT':0.5,
    #'PAD_IDX':TEXT.vocab.stoi[TEXT.pad_token]
}

#print('src locals()')
#print(locals())
#print('src locals()')
#print(globals())
#print('try update vars')
begins = ["床前明月光","离离原上草","天若有情天亦老"]
otu.update_vars_by_conf(globals())
print('W2V_TXT_FILE',W2V_TXT_FILE)


TEXT = data.ReversibleField(batch_first = True,
                  sequential=True,
                  use_vocab = True,
                  tokenize=lambda x:list(x),
                  #tokenize = tokenize_and_encode,
                  init_token = '<sos>',
                  eos_token = '<eos>',
                  pad_token = '<pad>',
                  unk_token = '<unk>' 
                  )


def load_data(path):
    dataset = data.TabularDataset(
        path = path, format = 'csv', skip_header = True,
        csv_reader_params={'delimiter':delimiter},
        fields=[(None, None)]*(text_col_idx-1) + [('text', TEXT)]
        )
    total_len=len(dataset)
    fetch = dataset.split(scale)[0]
    size = len(fetch.examples)
    return fetch,size


def create_train_dataset_fn(path):
    global pad_idx
    global vocab_size
    vectors=Vocab.Vectors(name=W2V_TXT_FILE)
    dataset,CV_size = load_data(path)
    print('start build vocab dict')
    TEXT.build_vocab(dataset,vectors=vectors)
    pad_idx = TEXT.vocab.stoi['<pad>']
    vocab_size= len(TEXT.vocab.stoi)
    print('vocab_size',vocab_size)
    print('pad_idx',pad_idx)
    return dataset
   


def create_dataloader(dataset,batch_size):
    train_iterator = data.BucketIterator(dataset, batch_size = batch_size,sort_key=lambda x: len(x.text),sort=True)
    return train_iterator

def data_preprocess_fn(runtime,experiment,original_data):
    
    text = original_data.text
    y = torch.cat([text[:,1:],(torch.ones(text.shape[0],1)*pad_idx).long()],dim=1)
    #print('y',y.shape)
    #print(y)
    #print(text)
    return text,y

def preview_data(data):
    print(data)   
    #print(data.title)
    #print(data.author)
    print('text char:\t',data.text)
    print('text encode:\t',TEXT.numericalize(''.join(data.text)))
    #print('text decode:\t',str(bert_tokenizer.decode(data.text)))



class CharRNNModel(nn.Module):
    def __init__(self,  hparams):
        super(CharRNNModel, self).__init__()
        self.num_layers = hparams['N_LAYERS']
        self.hidden_dim = hparams['HIDDEN_DIM'] 
        bidirectional = hparams['BIDIRECTIONAL']
        dropout = hparams['DROPOUT']

        #embedding_dim = hparams['EMBED_DIM']
        weights = torch.Tensor(TEXT.vocab.vectors)
        print('w2v',weights.shape)
        self.word_to_vec = nn.Embedding.from_pretrained(weights)
        embedding_dim = weights.shape[1]
        #self.word_to_vec = nn.Embedding(vocab_size, embedding_dim)

        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1
        #self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=self.num_layers,batch_first=True,bidirectional=bidirectional,dropout=dropout)
        self.encoder = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=self.num_layers,batch_first=True,bidirectional=bidirectional,dropout=dropout)
        self.decoder = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=self.num_layers,batch_first=True,bidirectional=bidirectional,dropout=dropout)
        self.decode_dense1 = otm.dense_layer(self.directions*self.hidden_dim, self.directions*self.hidden_dim,norm="LayerNorm",activation='ReLU')
        self.decode_dense2 = otm.dense_layer(self.directions*self.hidden_dim, self.directions*self.hidden_dim,norm="LayerNorm",activation='ReLU')
        self.linear_out = nn.Linear(self.directions*self.hidden_dim, vocab_size)

    def forward(self, input, hidden=None):
        text=input 
        device = text.device
        batch_size,seq_len = text.shape
        embedded = self.word_to_vec(text)
    
        # embeds_size:(seq_len,batch_size,embedding_dim)
        if hidden is None:
            #print('info',info.shape)
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
            c0 = torch.zeros(self.num_layers*self.directions, batch_size, self.hidden_dim).to(device)
        else:
            h0, c0 = hidden
        output, hidden = self.encoder(embedded, (h0, c0))
        output, hidden = self.decoder(embedded, hidden)
        output, hidden = self.lstm(embedded, (h0, c0))
        # output_size:(seq_len*batch_size,vocab_size)
        print('lstm output',output.shape)
        output = self.linear_out(self.decode_dense2(self.decode_dense1(output)))
        print('input shape',text.shape,'output shape',output.shape)
        return output, hidden




#model = PoetryModel(HPARAMS)


def bert_view_data(tensor,idx):
    l = []
    for i in tensor[idx,:]:
        #s = vocab.itos[i]
        l.append(i)
    #s = ''.join(l)
    #print(s)
    return str(bert_tokenizer.decode(l)).replace('[CLS]','').replace('[SEP]','').replace('[PAD]','').replace('[UNK]','')




criterion=torch.nn.CrossEntropyLoss()

def loss_evaluation_fn(runtime,experiment):
    output=runtime['output']
    target=runtime['target']
    output = output[0]
    #print(output.shape,target.shape)    
    C = output.shape[2]
    loss = criterion(output.view(-1,C),target.view(-1))
    return loss,loss.item()

def custom_train_fn(runtime,experiment):
    models = experiment.get('custom_models')
    device = experiment.get('device')
    optimizers = experiment.get('custom_optimizers')
    #criterions = experiment.get('loss_criterions')
    data = runtime['input']
    #target = runtime['target']
#def default_train_fn(models,criterions,optimizers,data,target,device):
    model = models[0]
    optimizer = optimizers[0]
    optimizer.zero_grad()
    target=runtime['target']
    #logger.info(data)
    #print(data.type)
    #print(next(model.parameters()).device)
    output = model(data)
    runtime['output'] = output
    #logger.info(output,target)
    #loss = F.nll_loss(output, target)
    #print(output.shape,target.unsqueeze(1).shape)
    #loss = loss_criterion.to(device)(data,target,output)
    #loss,loss_value = experiment.get('loss_evaluation_fn')(runtime,experiment)
    C = output[0].shape[2]
    loss = criterion(output[0].view(-1,C),target.view(-1))
    #loss = loss_criterion.to(device)(output,target)
    #with amp.scale_loss(loss, optimizer) as scaled_loss:
    #    scaled_loss.backward()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()
    return output,{'loss':loss.item()}

def pick_top_n(preds, top_n=5):
    top_pred_prob, top_pred_label = torch.topk(preds, top_n, 1)
    top_pred_prob /= torch.sum(top_pred_prob)
    top_pred_prob = top_pred_prob.squeeze(0).cpu().numpy()
    top_pred_label = top_pred_label.squeeze(0).cpu().numpy()
    c = np.random.choice(top_pred_label, size=1, p=top_pred_prob)
    return c


def generate_text(begin,model,device):


    text_len = 30 
    #samples = [convert.word_to_int(c) for c in begin]
    #samples = bert_tokenizer.encode(begin,add_special_tokens=False) #[:-1]
    with torch.no_grad():
        samples = TEXT.numericalize(begin).view(1,-1)
        #print('samples',samples)
        input_txt = torch.LongTensor(samples)
        input_txt = input_txt.to(device)
        #input_txt = torch.Tensor(input_txt)
        _, init_state = model(input_txt)
        result = samples
        #print('result',result)
        model_input = input_txt[:, -1][:, None]
        for i in range(text_len):
        #print(i)
        #print(model_input.shape)
        #print(model_input)
            out, init_state = model(model_input, init_state)
        #print('out',out.shape)
        #print(out)
            pred = torch.argmax(out,dim=2)
        #pred = pick_top_n(out.data)
        #print('pred',pred.shape)
        #print(pred)
            model_input = pred #torch.LongTensor(pred)
        #print(int(pred[0]))
        #print(pred)
        #print('cat',result.shape,pred.shape)
            result = torch.cat([result,pred.cpu()],dim=1)
        #print(result)
    #text = convert.arr_to_text(result)
    #print(result)
    #text = str(bert_tokenizer.decode(result)) 
    #print('  ')
    #print('Generate encode is: {}'.format(result))
    text = TEXT.reverse(result.data)

    return text

def generate_text_samples(runtime,experiment,ret):
    model = experiment['custom_models'][0]
    device = experiment['device']

    model = model.eval()
    text_list=[]
    for begin in begins:
        print('begin',begin)
        text = generate_text(begin,model,device)
        text_list.append(text)

    return {'display':'\n'.join(map(lambda x:x[0],text_list))}

def create_custom_models_fn(runtime,experiment):
    return [CharRNNModel(HPARAMS)]

def load_model_dynamic_params(data):
    global vocab_size
    vocab_size=data["vocab_size"]

def store_model_dynamic_params():
    global vocab_size
    return {"vocab_size":vocab_size}

def gen_text_by_input(runtime,experiment,args):
    #print(args)
    #print(runtime)
    #print(experiment)
    model = experiment['custom_models'][0]
    device = experiment['device']
    begin=args
    model = model.eval()
    text = generate_text(begin,model,device)
    print(text[0])

def post_save_models(path):
    print("Save vocab")
    with open (os.path.join(path,"vocab.bin"), 'wb') as f: 
        pickle.dump(TEXT.vocab, f)

def pre_load_models(path):
    global TEXT
    global vocab_size
    with open (os.path.join(path,"vocab.bin"), 'rb') as f: 
        TEXT.vocab = pickle.load(f)
        vocab_size= len(TEXT.vocab.stoi)
        print('vocab_size',vocab_size)

Experiment={
    "hparams":{'optim':'Adam',
               'lr':1e-4,
               'loader_n_worker':4,
               'batch_size':16,
               'n_epochs':50,
               'Adam':{'betas':(0.5,0.999)}
              },
    # Define Experiment Model
    "create_custom_models_fn":create_custom_models_fn,
    "load_model_dynamic_params":load_model_dynamic_params,
    "store_model_dynamic_params":store_model_dynamic_params,
    "post_save_models":post_save_models,
    "pre_load_models":pre_load_models,
    #"custom_models":[model],
    # Define function to create train dataset
    "create_train_dataset_fn":create_train_dataset_fn,
    # Define function to create validate dataset
    # Define callback function to collate dataset in dataloader, can be None
    "collate_fn_by_dataset":None,
    # Define callback function to preprocess data in each iteration, can be None
    "create_train_loader_fn":create_dataloader,
    "preview_dataset_item":preview_data,
    "data_preprocess_fn":data_preprocess_fn,
    # Define Loss function
    #"loss_criterions":[torch.nn.CrossEntropyLoss()],
    #"loss_criterions":[torch.nn.CrossEntropyLoss()],
    "loss_evaluation_fn":loss_evaluation_fn,
    "custom_train_fn":custom_train_fn,
    # Define function to deep insight result in each iteration, can be None
    #"post_epoch_val_fn":(otu.epoch_insight_classification,{'nclass':N_class}),
    #"post_batch_val_fn":otu.batch_result_extract,
    "post_epoch_train_fn":generate_text_samples,
    #"post_batch_train_fn":otu.batch_result_extract,
    #"validate_batch_val_result_fn":validate_batch_val_result_fn,
    "checkpoint_n_epoch":10,
    "train_validate_each_n_epoch":1,
    "train_validate_final_with_best":True,
    "interact_cmd":[("gen",gen_text_by_input,"gen <the begin words for generating>")]
}

