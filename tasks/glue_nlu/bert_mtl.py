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
import time
import random
import os
#from utils import log
from torch.utils.data import Dataset
import one_torch_utils as otu
import one_torch_models as otm
import sklearn.metrics

import pickle
import csv

from transformers import BertTokenizer, AutoModel


BERT_PRETRAINED = "/dev/shm/bert-base-uncased"

bert=None

#W2V_TXT_FILE="/dev/shm/glove.840B.300d.txt"
tokenizer=BertTokenizer.from_pretrained(BERT_PRETRAINED,use_fast=True)

print('done tokenizer')

max_input_length=512

text_col_idx = 4

print('max_input_length',max_input_length)

token2sen='[2SEN]'

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
scale=1
#glue_tasks=['CoLA']
glue_tasks=['CoLA','SST-2']
#glue_tasks=['CoLA','STS-B']
#glue_tasks=['QQP','QNLI','CoLA','SST-2','WNLI']
#glue_tasks=['RTE','QQP','QNLI','MNLI','CoLA','SST-2','WNLI']
#glue_tasks=['RTE','QQP','QNLI','MRPC','MNLI','CoLA','SST-2','WNLI']
#glue_tasks=['CoLA', 'SST-2','MNLI','MRPC','QNLI','QQP','RTE']
#glue_tasks=['QNLI']#, 'RTE', 'SST-2', 'STS-B', 'WNLI']
#glue_tasks=['WNLI']
#glue_tasks=['CoLA', 'MRPC', 'QNLI', 'RTE', 'SST-2', 'STS-B', 'WNLI']
##glue_tasks=['CoLA','MNLI', 'MRPC', 'QNLI', 'QQP', 'RTE', 'SST-2', 'STS-B', 'WNLI']
otu.update_vars_by_conf(globals())
#print('W2V_TXT_FILE',W2V_TXT_FILE)
w2v_vector = None


tokenizer.add_special_tokens({'additional_special_tokens':[token2sen]})

cls_token_idx = tokenizer.cls_token_id
sep_token_idx = tokenizer.sep_token_id
eos_token_idx = tokenizer.eos_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id

TEXT = data.Field(batch_first = True,
                  sequential=True,
                  use_vocab = False,
                  #tokenize=lambda x: x.split(),
                  #tokenize = tokenizer.encode,
                  tokenize = lambda x:tokenizer.encode(x,add_special_tokens=False),
                  #tokenize='spacy',  tokenizer_language='en_core_web_sm',
                  #tokenize = tokenize_and_encode,
                  pad_token = pad_token_idx,
                  )

def bert_numericalize(data):
    print(data)
    return ''

#TEXT.numericalize = bert_numericalize
#vectors=Vocab.Vectors(name=W2V_TXT_FILE)
#w2v_vector = vectors
#print(type(vectors.itos),len(vectors.itos),vectors.itos[:100])
#print('itos',len(TEXT.vocab.itos))
#print('itos',TEXT.vocab.itos[4000:4200])
#for t in ['the','The',',','.','?']:
#    print(t,'->',TEXT.vocab.stoi[t])

def triple_fn(x):
    #print(x)
    if x=='contradiction':
        return 0
    if x=='neutral':
        return 1
    if x=='entailment':
        return 2

def entailment_fn(x):
    if x=='entailment':
        return 1
    else:
        return 0

BINARY = data.Field(sequential=False,use_vocab=False,dtype = torch.long)
FLOAT = data.Field(sequential=False,use_vocab=False,dtype = torch.float)
TRIPLE = data.Field(sequential=False,use_vocab=False,dtype = torch.long,preprocessing=triple_fn)
ENTAILMENT = data.Field(sequential=False,use_vocab=False,dtype = torch.long,preprocessing=entailment_fn)

def load_data(path,task,mode):
    datapath=os.path.join(path,task,'{}.tsv'.format(mode))
    print(datapath)
    if task=='CoLA':
        dataset = data.TabularDataset(
            path = datapath, format = 'tsv',
            csv_reader_params={'quoting':csv.QUOTE_NONE},
            fields=[(None, None),('label',BINARY),(None,None),('sentence1', TEXT)]
            )
    elif task=='MNLI':
        dataset = data.TabularDataset(
            path = datapath, format = 'tsv', skip_header = True,
            csv_reader_params={'quoting':csv.QUOTE_NONE},
            fields=[('index', None),('promptID',None),('pairID',None),
                    ('genre',None),
                    ('sentence1_binary_parse',None),
                    ('sentence2_binary_parse',None),
                    ('sentence1_parse',None),
                    ('sentence2_parse',None),
                    ('sentence1', TEXT),
                    ('sentence2', TEXT),
                    ('label1', None),
                    ('label', TRIPLE)]
            )
    elif task=='MRPC':
        dataset = data.TabularDataset(
            path = datapath, format = 'tsv', skip_header = True,
            csv_reader_params={'quoting':csv.QUOTE_NONE},
            fields=[('label', BINARY),(None,None),(None,None),
                    ('sentence1', TEXT),
                    ('sentence2', TEXT)]
            )
    elif task=='QNLI':
        dataset = data.TabularDataset(
            path = datapath, format = 'tsv', skip_header = True,
            csv_reader_params={'quoting':csv.QUOTE_NONE},
            fields=[('index', None),
                    ('sentence1', TEXT),
                    ('sentence2', TEXT),
                    ('label', ENTAILMENT)]
            )
    elif task=='QQP':
        dataset = data.TabularDataset(
            path = datapath, format = 'tsv', skip_header = True,
            csv_reader_params={'quoting':csv.QUOTE_NONE},
            fields=[('index', None),(None,None),(None,None),
                    ('sentence1', TEXT),
                    ('sentence2', TEXT),
                    ('label', BINARY)]
            )
    elif task=='RTE':
        dataset = data.TabularDataset(
            path = datapath, format = 'tsv', skip_header = True,
            csv_reader_params={'quoting':csv.QUOTE_NONE},
            fields=[('index', None),
                    ('sentence1', TEXT),
                    ('sentence2', TEXT),
                    ('label', ENTAILMENT)]
            )
    elif task=='SST-2':
        dataset = data.TabularDataset(
            path = datapath, format = 'tsv', skip_header = True,
            csv_reader_params={'quoting':csv.QUOTE_NONE},
            fields=[('sentence1', TEXT),
                    ('label', BINARY)]
            )
    elif task=='STS-B':
        dataset = data.TabularDataset(
            path = datapath, format = 'tsv', skip_header = True,
            csv_reader_params={'quoting':csv.QUOTE_NONE},
            fields=[('index', None),
                    ('genre',None),
                    ('filename',None),
                    ('year',None),
                    ('old_index',None),
                    ('source1',None),
                    ('source2',None),
                    ('sentence1', TEXT),
                    ('sentence2', TEXT),
                    ('label', FLOAT)]
            )
    elif task=='WNLI':
        dataset = data.TabularDataset(
            path = datapath, format = 'tsv', skip_header = True,
            csv_reader_params={'quoting':csv.QUOTE_NONE},
            fields=[('index', None),
                    ('sentence1', TEXT),
                    ('sentence2', TEXT),
                    ('label', BINARY)]
            )
    else:
        print('Undefined task',task)
        return None,0

    print('done load ',task)

    total_len=len(dataset.examples)
    #print(dataset.examples[0])
    #print(dataset.examples[0].text)
    print(vars(dataset.examples[0]))
    if scale<1:
        return dataset.split(scale)[0],int(total_len*scale)
    else:
        return dataset,total_len


def create_train_dataset_fn(path):
    global pad_idx
    global vocab_size
    global TEXT
    global w2v_vector
    '''
    vectors=Vocab.Vectors(name=W2V_TXT_FILE)
    w2v_vector = vectors
    print(type(vectors.itos),len(vectors.itos),vectors.itos[:100])
    TEXT.vocab=Vocab.build_vocab_from_iterator([vectors.itos])
    print('itos',len(TEXT.vocab.itos))
    print('itos',TEXT.vocab.itos[4000:4200])
    for t in ['the','The',',','.','?']:
        print(t,'->',TEXT.vocab.stoi[t])
    '''
    all_datasets = {}
    for task in glue_tasks:
        dataset,CV_size = load_data(path,task,'train')
        if dataset is not None:
            all_datasets[task]=dataset
    return all_datasets 
  
def create_val_dataset_fn(path): 
    all_datasets = {}
    for task in glue_tasks:
        dataset,CV_size = load_data(path,task,'dev')
        if dataset is not None:
            all_datasets[task]=dataset
    return all_datasets 


def create_dataloader(dataset,batch_size):
    if isinstance(dataset,dict):
        all_loaders={}
        for task in dataset:
            all_loaders[task] = data.BucketIterator(dataset[task], batch_size = batch_size,sort_key=lambda x: len(x.sentence1),sort=False)
            #all_loaders[task] = data.BucketIterator(dataset[task], batch_size = batch_size,sort_key=lambda x: len(x.sentence1),sort=True)
        return all_loaders
    else:
        train_iterator = data.BucketIterator(dataset, batch_size = batch_size,sort_key=lambda x: len(x.sentence1),sort=False)
        #train_iterator = data.BucketIterator(dataset, batch_size = batch_size,sort_key=lambda x: len(x.sentence1),sort=True)
        return train_iterator

def data_preprocess_fn(runtime,experiment,original_data,task=None):
    #print('task',task) 
    batch_size = original_data.sentence1.shape[0]
    if task in ["CoLA","SST-2"]: 
        x = original_data.sentence1
    else:
        #print(task)
        #print(original_data.sentence1.shape,original_data.sentence2.shape)
        #x = (original_data.sentence1,original_data.sentence2)
        #print(original_data.sentence1,original_data.sentence2)
        sep = (torch.ones(batch_size,1)*sep_token_idx).long()
        x = torch.cat([original_data.sentence1,sep,original_data.sentence2],dim=1)
    cls = (torch.ones(batch_size,1)*cls_token_idx).long()
    x = torch.cat([cls,x],dim=1)
    y = original_data.label
    #y = torch.cat([text[:,1:],(torch.ones(text.shape[0],1)*pad_idx).long()],dim=1)
    #print('y',y.shape)
    #print(y)
    #print(text)
    return x,y

def preview_data(data,task):
    #print(data)   
    #print(dir(data))   
    #print(data.title)
    #print(data.author)
    if task=='CoLA':
        print(data.sentence1)
        print('text words:\t',tokenizer.decode(data.sentence1),'label',data.label)
        print('text encode:\t',data.sentence1)
    elif task=='MNLI':
        print('sen1 :\t',tokenizer.decode(data.sentence1),data.sentence1)
        print('sen2 :\t',tokenizer.decode(data.sentence2),data.sentence2)
        print('label:\t',data.label) 
    elif task=='MRPC':
        print('sen1 :\t',tokenizer.decode(data.sentence1),data.sentence1)
        print('sen2 :\t',tokenizer.decode(data.sentence2),data.sentence2)
        print('label:\t',data.label) 
    elif task=='QNLI':
        print('question:\t',tokenizer.decode(data.sentence1),data.sentence1)
        print('sen :\t',tokenizer.decode(data.sentence2),data.sentence2)
        print('label:\t',data.label) 
    elif task=='QQP':
        print('question1:\t',tokenizer.decode(data.sentence1),data.sentence1)
        print('question2:\t',tokenizer.decode(data.sentence2),data.sentence2)
        print('label:\t',data.label) 
    elif task=='RTE':
        print('sen1 :\t',tokenizer.decode(data.sentence1),data.sentence1)
        print('sen2 :\t',tokenizer.decode(data.sentence2),data.sentence2)
        print('label:\t',data.label) 
    elif task=='SST-2':
        print('sen1 :\t',tokenizer.decode(data.sentence1),data.sentence1)
        print('label:\t',data.label) 
    elif task=='STS-B':
        print('sen1 :\t',tokenizer.decode(data.sentence1),data.sentence1)
        print('sen2 :\t',tokenizer.decode(data.sentence2),data.sentence2)
        print('score:',data.label)
    elif task=='WNLI':
        print('sen1 :\t',tokenizer.decode(data.sentence1),data.sentence1)
        print('sen2 :\t',tokenizer.decode(data.sentence2),data.sentence2)
        print('label:',data.label)
    #print('text decode:\t',str(bert_tokenizer.decode(data.text)))



class BertModel(nn.Module):
    def __init__(self,  bert,hparams):
        super(BertModel, self).__init__()
        self.num_layers = hparams['N_LAYERS']
        self.hidden_dim = hparams['HIDDEN_DIM'] 
        bidirectional = hparams['BIDIRECTIONAL']
        dropout = hparams['DROPOUT']

        #for param in bert.parameters():
        #    param.requires_grad = False
        
        self.bert = bert

        embedding_dim = bert.config.to_dict()['hidden_size']
        self.bert_max_position_embeddings = bert.config.to_dict()['max_position_embeddings']

        self.bidirectional = bidirectional
        

        #self.dropout = nn.Dropout(dropout,inplace=True)

        
        self.task_layer={}
        self.output_layer={}
        common_output = embedding_dim
        for task in glue_tasks:
            self.task_layer[task]=nn.Sequential(*[otm.dense_layer(common_output,common_output,norm="LayerNorm",activation="ReLU",dropout=0.5),
                                               otm.dense_layer(common_output,common_output,norm="LayerNorm",activation="ReLU",dropout=0.5),
                                               otm.dense_layer(common_output,common_output,norm="LayerNorm",activation="ReLU",dropout=0.5),
                                               otm.dense_layer(common_output,common_output,norm="LayerNorm",activation="ReLU",dropout=0.5)])
        self.task_layer=nn.ModuleDict(self.task_layer)
        for task in glue_tasks:
            if task in ['CoLA','MRPC','QNLI','QQP','RTE','SST-2','WNLI']:
                self.output_layer[task] = nn.Linear(common_output,2)
            if task in ['MNLI']:
                self.output_layer[task] = nn.Linear(common_output,3)
            if task in ['STS-B']:
                self.output_layer[task] = nn.Linear(common_output,1)
        self.output_layer=nn.ModuleDict(self.output_layer)

    def forward(self, input, task=None):
        #print('input',input.shape)
        text=input[:,:self.bert_max_position_embeddings] 
        device = text.device
        batch_size,seq_len = text.shape
        
        #with torch.no_grad():
        bert_output = self.bert(text)
        embedded = bert_output[0][:,0,]
        #embedded = bert_output[1]
        #print('\nbert0',bert_output[0].shape,'bert1',bert_output[1].shape,' use ',embedded.shape)

        #print('bert output',embedded.shape)
    
        #hidden = self.dropout(hidden[-1,:,:])
        # output_size:(seq_len*batch_size,vocab_size)
        output = self.task_layer[task](embedded)
        #print('lstm output',output.shape)
        output = self.output_layer[task](output)
        #print('final output',output.shape)
        return output 



criterion=torch.nn.CrossEntropyLoss()
regression_criterion = torch.nn.MSELoss(reduction='mean')


def loss_evaluation_fn(runtime,experiment):
    output=runtime['output']
    target=runtime['target']
    #output = output[0]
    task = runtime.get('task')
    if task in ['CoLA','SST-2','MRPC','QNLI','QQP','RTE','WNLI']: 
        C = 2
    #print(output.shape,target.shape)    
    elif task in ['MNLI']:
        C = 3
    elif task in ['STS-B']:
        loss = regression_criterion(output.squeeze(dim=1),target)
        return loss,loss.item()
    else:
        print('bug on')
        exit(0)
    #print(task)
    #print('input',runtime.get('input'),runtime.get('input').shape)
    #print('output',output)
    #print('target',target)
    #print('output shape',output.shape,'target shape',target.shape)
    #print('output',output.shape,'target',target.shape)
    loss = criterion(output,target)
    
    models = experiment.get('custom_models')
    model = models[0]
    loss += otm.regularization(model)*1e-5
    #L1_reg = 0
    #for param in model.parameters():
    #    if param.requires_grad:
    #        L1_reg += torch.sum(torch.abs(param))
    #loss = loss + L1_reg*1e-5
    #loss = criterion(output.view(-1,C),target.view(-1))
    return loss,loss.item()

def custom_train_fn(runtime,experiment):
    models = experiment.get('custom_models')
    device = experiment.get('device')
    optimizers = experiment.get('custom_optimizers')
    #criterions = experiment.get('loss_criterions')
    data = runtime['input']
    task = runtime['task']
    #target = runtime['target']
    #def default_train_fn(models,criterions,optimizers,data,target,device):
    
    #print(data) 
    ##print(task) 
    model = models[0]
    optimizer = optimizers[0]
    optimizer.zero_grad()
    target=runtime['target']
    #logger.info(data)
    #print(data.type)
    #print(next(model.parameters()).device)
    output = model(data,task)
    runtime['output'] = output
    #logger.info(output,target)
    #loss = F.nll_loss(output, target)
    #print(output.shape,target.unsqueeze(1).shape)
    #loss = loss_criterion.to(device)(data,target,output)
    loss,loss_value = experiment.get('loss_evaluation_fn')(runtime,experiment)
    #C = output.shape[1]
    #loss = criterion(output.view(-1,C),target.view(-1))
    #loss = loss_criterion.to(device)(output,target)
    #with amp.scale_loss(loss, optimizer) as scaled_loss:
    #    scaled_loss.backward()
    loss.backward()
    #nn.utils.clip_grad_norm_(model.parameters(), 5)
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


def create_custom_models_fn(runtime,experiment):
    global bert
    bert = AutoModel.from_pretrained(BERT_PRETRAINED)
    bert.resize_token_embeddings(len(tokenizer))
    model = BertModel(bert,HPARAMS)
    #from IPython import embed; embed("custom")
    return [model]


def matthews_corrcoef(runtime,experiment,ret):
    pred = ret.get('pred')
    target = ret.get('target')
    mcc = sklearn.metrics.matthews_corrcoef(pred,target)
    otu.logger.info('MCC:{}'.format(mcc))
    return ret
    
def print_hook(runtime,experiment,ret,info=""):
    otu.logger.info(info)
    return ret


Experiment={
    "hparams":{
               'optim':'Adam',
               #'optim':'SGD',
               'lr':2e-5,
               'loader_n_worker':2,
               #'batch_size':256,
               'batch_size':64,
               'n_epochs':200,
               'Adam':{'betas':(0.5,0.999),'weight_decay':0.},
               #'Adam':{'betas':(0.5,0.999),'weight_decay':0.1},
               'SGD':{'momentum':0.5,'weight_decay':0},
               'lr_scheduler':'StepLR'
              },
    # Define Experiment Model
    "create_custom_models_fn":create_custom_models_fn,
    #"pre_load_models":pre_load_models,
    #"custom_models":[model],
    # Define function to create train dataset
    "create_train_dataset_fn":create_train_dataset_fn,
    "create_val_dataset_fn":create_val_dataset_fn,
    # Define function to create validate dataset
    # Define callback function to collate dataset in dataloader, can be None
    "collate_fn_by_dataset":None,
    # Define callback function to preprocess data in each iteration, can be None
    "create_train_loader_fn":create_dataloader,
    "create_val_loader_fn":create_dataloader,
    "multi_task_labels":glue_tasks,
    "preview_dataset_item":preview_data,
    "data_preprocess_fn":data_preprocess_fn,
    # Define Loss function
    #"loss_criterions":[torch.nn.CrossEntropyLoss()],
    #"loss_criterions":[torch.nn.CrossEntropyLoss()],
    "loss_evaluation_fn":loss_evaluation_fn,
    "custom_train_fn":custom_train_fn,
    # Define function to deep insight result in each iteration, can be None
    "post_epoch_val_fn":{"CoLA":[(print_hook,{'info':'post_epoch_val_fn'}),otu.get_epoch_results,matthews_corrcoef,(otu.epoch_insight_classification,{'nclass':2})],
                         #"CoLA":(otu.epoch_insight_classification,{'nclass':2}),
                         "SST-2":(otu.epoch_insight_classification,{'nclass':2}),
                         "MRPC":(otu.epoch_insight_classification,{'nclass':2}),
                         "QNLI":(otu.epoch_insight_classification,{'nclass':2}),
                         "QQP":(otu.epoch_insight_classification,{'nclass':2}),
                         "RTE":(otu.epoch_insight_classification,{'nclass':2}),
                         "MNLI":(otu.epoch_insight_classification,{'nclass':3})},
    "post_batch_val_fn":otu.batch_result_extract,
    "post_batch_train_fn":otu.batch_result_extract,
    "post_epoch_train_fn":{"CoLA":[(print_hook,{'info':'post_epoch_train_fn'}),otu.get_epoch_results,matthews_corrcoef,(otu.epoch_insight_classification,{'nclass':2})]},
    #"validate_batch_val_result_fn":validate_batch_val_result_fn,
    #"checkpoint_n_epoch":10,
    "train_validate_each_n_epoch":1,
    "train_validate_final_with_best":True,
}

