#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import os
#sys.path.append('./drive/My Drive/Colab Notebooks/')
from torchtext import data
import pandas as pd
import re
from torchtext import vocab
import time
import random
import os
#from utils import log
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset
import one_torch_utils as otu

N_class=2
DATA_SET='../../data/'

from transformers import AutoTokenizer, AutoModel
bert_tokenizer = AutoTokenizer.from_pretrained(os.path.join(DATA_SET,"ernie-1.0"))


#bert_tokenizer=BertTokenizer.from_pretrained('bert-base-chinese')
print(len(bert_tokenizer.vocab))
tokens = bert_tokenizer.tokenize('[CLS]ºÃÏñÊÇ')
print(tokens)
tokens = bert_tokenizer.tokenize('ÊÇÊ²Ã´')
print(tokens)

init_token = bert_tokenizer.cls_token
eos_token = bert_tokenizer.sep_token
pad_token = bert_tokenizer.pad_token
unk_token = bert_tokenizer.unk_token

print(init_token, eos_token, pad_token, unk_token)
init_token_idx = bert_tokenizer.convert_tokens_to_ids(init_token)
eos_token_idx = bert_tokenizer.convert_tokens_to_ids(eos_token)
pad_token_idx = bert_tokenizer.convert_tokens_to_ids(pad_token)
unk_token_idx = bert_tokenizer.convert_tokens_to_ids(unk_token)

print(init_token_idx, eos_token_idx, pad_token_idx, unk_token_idx)
init_token_idx = bert_tokenizer.cls_token_id
eos_token_idx = bert_tokenizer.sep_token_id
pad_token_idx = bert_tokenizer.pad_token_id
unk_token_idx = bert_tokenizer.unk_token_id

print(init_token_idx, eos_token_idx, pad_token_idx, unk_token_idx)

max_input_length = bert_tokenizer.max_model_input_sizes['bert-base-chinese']

print(max_input_length)

def tokenize_and_cut(sentence):
    #tokens = bert_tokenizer.tokenize('[CLS]' + sentence) 
    tokens = bert_tokenizer.tokenize( sentence)
    tokens = tokens[:max_input_length-3]
    return tokens

TEXT = data.Field(batch_first = True,
                  use_vocab = False,
                  tokenize = bert_tokenizer.encode,
                  #preprocessing = bert_tokenizer.convert_tokens_to_ids,
                  #init_token = init_token_idx,
                  #eos_token = eos_token_idx,
                  pad_token = pad_token_idx,
                  #unk_token = unk_token_idx
                  )

def format_score(score):
    return int((float(score)+1)/2)

LABEL = data.Field(sequential=False,use_vocab=False,dtype = torch.float,preprocessing=format_score)



scale=0.1
def load_data(path):
    dataset = data.TabularDataset(
        path = path, format = 'csv', skip_header = True,
        #å¨è¿éï¼å¿é¡»è®©æä»¬çtrain.csvãval.csvæååå«ä»¥labelåtextä¸ºååçä¸¤ä¸ªåï¼test.csvåå«ä»¥textä¸ºååçåå°±å¯ä»¥äº
        fields=[('score', LABEL),
                ('title', TEXT)
            ])
    #total_len=len(dataset)
    #require_len=int(scale*total_len)
    #print('æ°æ®éè§æ¨¡:'.format(len(dataset)))
    #print('è§æ¨¡:'.format(len(dataset)))
    #print('^Hç¤ºä¾',vars(dataset[5]))
    fetch = dataset.split(scale)[0]
    print('view',vars(fetch[5]))
    
    print(fetch.examples[5])
    size = len(fetch.examples)
    print('sum',size)
    return fetch,size

class SenDataset(Dataset):
    def __init__(self,path):
        self.dataset = data.TabularDataset(
            path = path, format = 'csv', skip_header = True,
            #å¨è¿éï¼å¿é¡»è®©æä»¬çtrain.csvãval.csvæååå«ä»¥labelåtextä¸ºååçä¸¤ä¸ªåï¼test.csvåå«ä»¥textä¸ºååçåå°±å¯ä»¥äº
            fields=[('score', LABEL),
                    ('title', TEXT)
                ])
    def __getitem__(self,index):
        item=self.dataset.examples[index]
        X = torch.Tensor(item.title)
        Y = torch.Tensor([item.score])
        return X,Y
    def __len__(self):
        return len(self.dataset) 


def create_train_dataset_fn(path):
    dataset,CV_size = load_data(path)
    return dataset
   
    return SenDataset(path)
    dataset,CV_size = load_data(path)
    print('create_val_dataset_fn')
    print(dataset[0])
    print(len(dataset))
    print(dataset.examples[0])
    print(dataset.examples[0].title)
    print(dataset.examples[0].score)
    print(dir(dataset.examples[0]))
    return dataset.examples

def create_val_dataset_fn(path):
    dataset,CV_size = load_data(path)
    return dataset
    return SenDataset(path)
    test_dataset,Test_size = load_data(path)
    return test_dataset.examples

def create_dataloader(dataset,batch_size):
    train_iterator = data.BucketIterator(dataset, batch_size = batch_size,sort=False)
    return train_iterator

def data_preprocess_fn(runtime,experiment,original_data):
    #print(original_data.score)
    #print(original_data.title)
    return original_data.title,original_data.score.long()


bert = AutoModel.from_pretrained(os.path.join(DATA_SET,"ernie-1.0"))

"""#### 2.2.1 BERT+Flat"""

class BERTFlatModel(nn.Module):
    def __init__(self,bert,hparams):

        super().__init__()
        hidden_dim = hparams['HIDDEN_DIM']
        output_dim = hparams['OUTPUT_DIM']
        n_layers = hparams['N_LAYERS']
        bidirectional = hparams['BIDIRECTIONAL']
        dropout = hparams['DROPOUT']

        self.bert = bert

        embedding_dim = bert.config.to_dict()['hidden_size']

        self.bidirectional = bidirectional
        
        embedding_dim = bert.config.to_dict()['hidden_size']

        self.fc1 = nn.Linear(embedding_dim,384)
        self.fc2 = nn.Linear(384,64)
        self.fc3 = nn.Linear(64,output_dim)
        #self.fc = nn.Linear(embedding_dim,output_dim)
    def forward(self, text):

        with torch.no_grad():
            bert_output = self.bert(text)
            embedded = bert_output[0][:,0,]


        fc1 = torch.relu(self.fc1(embedded))
        #print("fc1",fc1.shape)

        fc2 = torch.relu(self.fc2(fc1))
        return F.log_softmax(self.fc3(fc2), dim=1)
HPARAMS = {
    #"W2V_PATH":W2V_PATH,
    'OUTPUT_DIM': N_class,
    'HIDDEN_DIM': 150,
    'N_LAYERS': 1,
    'BIDIRECTIONAL': False,
    'DROPOUT':0.5,
    #'PAD_IDX':TEXT.vocab.stoi[TEXT.pad_token]
}

#model = BERTFlatModel(bert, HPARAMS)

class BERTGRUModel(nn.Module):
    def __init__(self,bert,hparams):
        
        super().__init__()
        hidden_dim = hparams['HIDDEN_DIM']
        output_dim = hparams['OUTPUT_DIM']
        n_layers = hparams['N_LAYERS']
        bidirectional = hparams['BIDIRECTIONAL']
        dropout = hparams['DROPOUT']
        
        self.bert = bert
        self.bidirectional = bidirectional
        
        embedding_dim = bert.config.to_dict()['hidden_size']

        self.gru = nn.GRU(embedding_dim, 
                       hidden_dim, 
                       num_layers=n_layers, 
                       bidirectional=bidirectional, 
                       batch_first = True,
                       dropout=dropout)
        if bidirectional is True:
            #self.bn = nn.BatchNorm1d(hidden_dim * 2)
            self.fc1 = nn.Linear(hidden_dim * 2, 128)
        else:
            #self.bn = nn.BatchNorm1d(hidden_dim)
            self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(dropout,inplace=True)
        
    def forward(self, text):
        
        #text = [batch size, sent len]
        #print("text input",text.shape)
        #print(text)
        with torch.no_grad():
            bert_output = self.bert(text)
            #print('bertout',bert_output[0].shape)
            embedded = bert_output[0]
 
        #print("bert output",embedded.shape)
        #print(text[0])
        #print(embedded[0])
        #st.write('embedded_shape',embedded.shape)
        #embedded = [sent len, batch size, emb dim]
        packed_output, hidden = self.gru(embedded)

        #print("hidden",hidden.shape)

        if self.bidirectional is True:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
            #fc1 = F.relu(self.fc1(self.bn(hidden.squeeze(0))))
            #fc1 = F.relu(self.fc1(hidden.squeeze(0)))
        else:
            hidden = self.dropout(hidden[-1,:,:])
            #hidden = self.dropout(hidden)
            #fc1 = F.relu(self.fc1(hidden.squeeze(0)))
        #print("hidden 2",hidden.shape)
        fc1 = F.relu(self.fc1(hidden.squeeze(0)))
            #fc1 = F.relu(self.fc1(self.bn(hidden.squeeze(0))))
        fc2 = torch.relu(self.fc2(fc1))
        fc3 = self.fc3(fc2)
        #print("fc3",fc3.shape)
        output = F.log_softmax(fc3, dim=1)
        #print('output',output.shape)
        return output
        fc2 = F.relu(self.fc2(fc1))
        #print(fc2.shape)
        fc3 = torch.sigmoid(self.fc3(fc2))
        #print(fc3.shape)
        return fc3
#print(repr(bert))


#model = BERTFlatModel(bert, HPARAMS)
model = BERTGRUModel(bert, HPARAMS)


def bert_view_data(tensor,idx):
    l = []
    for i in tensor[idx,:]:
        #s = vocab.itos[i]
        l.append(i)
    #s = ''.join(l)
    #print(s)
    return str(bert_tokenizer.decode(l)).replace('[CLS]','').replace('[SEP]','').replace('[PAD]','').replace('[UNK]','')

def bert_record_error(preds,y,tensor,error_list):
    care = (preds != y)
    for i in range(len(care)):
        if care[i] == True:
            #print('find err')
            error_list.append((bert_view_data(tensor,i),preds[i].item(),y[i].item()))

def validate_pre_epoch_val_result_fn(runtime,experiment,ret):
    otu.save_results_to_csv(runtime,experiment,'errors',[['sentence','pred','target']])
    return ret

def validate_batch_val_result_fn(runtime,experiment,ret):
    if ret.get('errors') is None:
        ret['errors']=[]
    
    errors=ret.get('errors')
    pred=torch.argmax(runtime.get('output'),dim=1)
    target=runtime.get('target')
    input=runtime.get('input')
    #print(input.shape,target.shape,pred.shape)
    #print(pred)
    #print(input)
    #print(target)
    care = (pred != target)
    for i in range(len(care)):
        if care[i] == True:
            sen=bert_view_data(input,i)
            if sen:
                #errors.append((sen,pred[i].item(),target[i].item()))
                otu.save_results_to_csv(runtime,experiment,'errors',[[sen,pred[i].item(),target[i].item()]])
    return ret


Experiment={
    "hparams":{'optim':'Adam',
               'lr':2e-4,
               'loader_n_worker':4,
               'batch_size':256,
               'checkpoint_n_epoch':10,
               'train_epoch_val_n_batch':-1,
               'Adam':{'betas':(0.5,0.999)}
              },
    # Define Experiment Model
    "custom_models":[model],
    # Define function to create train dataset
    "create_train_dataset_fn":create_train_dataset_fn,
    # Define function to create validate dataset
    "create_val_dataset_fn":create_val_dataset_fn,    
    # Define callback function to collate dataset in dataloader, can be None
    "collate_fn_by_dataset":None,
    # Define callback function to preprocess data in each iteration, can be None
    "create_train_loader_fn":create_dataloader,
    "create_val_loader_fn":create_dataloader,
    "data_preprocess_fn":data_preprocess_fn,
    # Define Loss function
    "loss_criterions":[torch.nn.NLLLoss()],
    #"loss_evaluation_fn":loss_evaluation_fn,
    # Define function to deep insight result in each iteration, can be None
    "post_epoch_val_fn":(otu.epoch_insight_classification,{'nclass':N_class}),
    "post_batch_val_fn":otu.batch_result_extract,
    "post_epoch_train_fn":(otu.epoch_insight_classification,{'nclass':N_class}),
    "post_batch_train_fn":otu.batch_result_extract,
    "validate_batch_val_result_fn":validate_batch_val_result_fn
}

