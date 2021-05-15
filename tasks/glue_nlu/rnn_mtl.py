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
import csv



W2V_TXT_FILE="/dev/shm/glove.840B.300d.txt"

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
scale=1
#glue_tasks=['MNLI']
#glue_tasks=['QQP','QNLI','MNLI','CoLA','SST-2','WNLI']
#glue_tasks=['RTE','QQP','QNLI','MNLI','CoLA','SST-2','WNLI']
glue_tasks=['RTE','QQP','QNLI','MRPC','MNLI','CoLA','SST-2','WNLI']
#glue_tasks=['CoLA', 'SST-2','MNLI','MRPC','QNLI','QQP','RTE']
#glue_tasks=['CoLA','MNLI', 'MRPC', 'QNLI', 'QQP', 'RTE', 'SST-2', 'STS-B', 'WNLI']
otu.update_vars_by_conf(globals())
print('W2V_TXT_FILE',W2V_TXT_FILE)
w2v_vector = None



TEXT = data.Field(batch_first = True,
                  sequential=True,
                  use_vocab = True,
                  #tokenize=lambda x: x.split(),
                  tokenize='spacy',  tokenizer_language='en_core_web_sm',
                  #tokenize = tokenize_and_encode,
                  init_token = '<sos>',
                  eos_token = '<eos>',
                  pad_token = '<pad>',
                  unk_token = '<unk>' 
                  )
vectors=Vocab.Vectors(name=W2V_TXT_FILE)
w2v_vector = vectors
print(type(vectors.itos),len(vectors.itos),vectors.itos[:100])
TEXT.vocab=Vocab.build_vocab_from_iterator([vectors.itos])
print('itos',len(TEXT.vocab.itos))
print('itos',TEXT.vocab.itos[4000:4200])
for t in ['the','The',',','.','?']:
    print(t,'->',TEXT.vocab.stoi[t])

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
            all_loaders[task] = data.BucketIterator(dataset[task], batch_size = batch_size)
        return all_loaders
    else:
        train_iterator = data.BucketIterator(dataset, batch_size = batch_size,sort_key=lambda x: len(x.text),sort=True)
        return train_iterator

def data_preprocess_fn(runtime,experiment,original_data,task=None):
    #print('task',task) 
    if task in ["CoLA","SST-2"]: 
        x = original_data.sentence1
    else:
        #print(task)
        #print(original_data.sentence1.shape,original_data.sentence2.shape)
        #x = (original_data.sentence1,original_data.sentence2)
        #print(original_data.sentence1,original_data.sentence2)
        sep = torch.ones(original_data.sentence1.shape[0],1).long()
        x = torch.cat([original_data.sentence1,sep,original_data.sentence2],dim=1)
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
        print('text words:\t',data.sentence1,'label',data.label)
        print('text encode:\t',TEXT.numericalize([data.sentence1]))
    elif task=='MNLI':
        print('sen1 :\t',data.sentence1,TEXT.numericalize([data.sentence1]))
        print('sen2 :\t',data.sentence2,TEXT.numericalize([data.sentence2]))
        print('label:\t',data.label) 
    elif task=='MRPC':
        print('sen1 :\t',data.sentence1,TEXT.numericalize([data.sentence1]))
        print('sen2 :\t',data.sentence2,TEXT.numericalize([data.sentence2]))
        print('label:\t',data.label) 
    elif task=='QNLI':
        print('question:\t',data.sentence1,TEXT.numericalize([data.sentence1]))
        print('sen :\t',data.sentence2,TEXT.numericalize([data.sentence2]))
        print('label:\t',data.label) 
    elif task=='QQP':
        print('question1:\t',data.sentence1,TEXT.numericalize([data.sentence1]))
        print('question2:\t',data.sentence2,TEXT.numericalize([data.sentence2]))
        print('label:\t',data.label) 
    elif task=='RTE':
        print('sen1 :\t',data.sentence1,TEXT.numericalize([data.sentence1]))
        print('sen2 :\t',data.sentence2,TEXT.numericalize([data.sentence2]))
        print('label:\t',data.label) 
    elif task=='SST-2':
        print('sen1 :\t',data.sentence1,TEXT.numericalize([data.sentence1]))
        print('label:\t',data.label) 
    elif task=='STS-B':
        print('sen1 :\t',data.sentence1,TEXT.numericalize([data.sentence1]))
        print('sen2 :\t',data.sentence2,TEXT.numericalize([data.sentence2]))
        print('score:',data.label)
    elif task=='WNLI':
        print('sen1 :\t',data.sentence1,TEXT.numericalize([data.sentence1]))
        print('sen2 :\t',data.sentence2,TEXT.numericalize([data.sentence2]))
        print('label:',data.label)
    #print('text decode:\t',str(bert_tokenizer.decode(data.text)))



class RNNModel(nn.Module):
    def __init__(self,  hparams):
        super(RNNModel, self).__init__()
        self.num_layers = hparams['N_LAYERS']
        self.hidden_dim = hparams['HIDDEN_DIM'] 
        bidirectional = hparams['BIDIRECTIONAL']
        dropout = hparams['DROPOUT']

        #embedding_dim = hparams['EMBED_DIM']
        print('load w2v')
        print(w2v_vector.vectors)
        weights = w2v_vector.vectors
        print('w2v',weights.shape)
        self.word_to_vec = nn.Embedding.from_pretrained(weights)
        embedding_dim = weights.shape[1]
        #self.word_to_vec = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout,inplace=True)

        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1
        #self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=self.num_layers,batch_first=True,bidirectional=bidirectional,dropout=dropout)
        self.encoder = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=self.num_layers,batch_first=True,bidirectional=bidirectional,dropout=dropout)
        #self.decoder = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=self.num_layers,batch_first=True,bidirectional=bidirectional,dropout=dropout)
        self.decode_dense1 = otm.dense_layer(self.directions*self.hidden_dim, self.directions*self.hidden_dim,norm="LayerNorm",activation='ReLU')
        self.decode_dense2 = otm.dense_layer(self.directions*self.hidden_dim, self.directions*self.hidden_dim,norm="LayerNorm",activation='ReLU')
        self.linear_out = nn.Linear(self.directions*self.hidden_dim, vocab_size)
        
        self.task_layer={}
        self.output_layer={}
        common_output = self.directions*self.hidden_dim
        for task in glue_tasks:
            self.task_layer[task]=nn.Sequential(*[otm.dense_layer(common_output,common_output,norm="LayerNorm",activation="ReLU"),
                                               otm.dense_layer(common_output,common_output,norm="LayerNorm",activation="ReLU")])
        self.task_layer=nn.ModuleDict(self.task_layer)
        for task in ['CoLA','MRPC','QNLI','QQP','RTE','SST-2','WNLI']:
            self.output_layer[task] = nn.Linear(common_output,2)
        for task in ['MNLI']:
            self.output_layer[task] = nn.Linear(common_output,3)
        for task in ['STS-B']:
            self.output_layer[task] = nn.Linear(common_output,1)
        self.output_layer=nn.ModuleDict(self.output_layer)

    def forward(self, input, task=None):
        #print('input',input.shape)
        text=input 
        device = text.device
        batch_size,seq_len = text.shape
        embedded = self.word_to_vec(text)
    
        _, (hidden,c) = self.encoder(embedded)
        #print('embedded',embedded.shape)
        #print('hidden',hidden)
        #print('hidden',hidden.shape)
        if self.bidirectional is True:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
            #hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
            #fc1 = F.relu(self.fc1(self.bn(hidden.squeeze(0))))
            #fc1 = F.relu(self.fc1(hidden.squeeze(0)))
        else:
            hidden = hidden[-1,:,:]
            #hidden = self.dropout(hidden[-1,:,:])
        # output_size:(seq_len*batch_size,vocab_size)
        output = self.task_layer[task](hidden)
        #print('lstm output',output.shape)
        output = self.output_layer[task](output)
        #print('final output',output.shape)
        return output 




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
    #output = output[0]
    task = runtime.get('task')
    if task in ['CoLA','SST-2','MRPC','QNLI','QQP','RTE','WNLI']: 
        C = 2
    #print(output.shape,target.shape)    
    elif task in ['MNLI']:
        C = 3
    else:
        print('bug on')
        exit(0)
    #print(task)
    #print('input',runtime.get('input'),runtime.get('input').shape)
    #print('output',output)
    #print('target',target)
    #print('output',output.shape,'target',target.shape)
    loss = criterion(output.view(-1,C),target.view(-1))
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
    #loss,loss_value = experiment.get('loss_evaluation_fn')(runtime,experiment)
    C = output.shape[1]
    loss = criterion(output.view(-1,C),target.view(-1))
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
    return [RNNModel(HPARAMS)]

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
               'loader_n_worker':2,
               'batch_size':32,
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
    "post_epoch_val_fn":{"CoLA":(otu.epoch_insight_classification,{'nclass':2}),
                         "SST-2":(otu.epoch_insight_classification,{'nclass':2}),
                         "MRPC":(otu.epoch_insight_classification,{'nclass':2}),
                         "QNLI":(otu.epoch_insight_classification,{'nclass':2}),
                         "QQP":(otu.epoch_insight_classification,{'nclass':2}),
                         "RTE":(otu.epoch_insight_classification,{'nclass':2}),
                         "MNLI":(otu.epoch_insight_classification,{'nclass':3})},
    "post_batch_val_fn":otu.batch_result_extract,
    "post_epoch_train_fn":None,#generate_text_samples,
    #"post_batch_train_fn":otu.batch_result_extract,
    #"validate_batch_val_result_fn":validate_batch_val_result_fn,
    "checkpoint_n_epoch":10,
    "train_validate_each_n_epoch":1,
    "train_validate_final_with_best":True,
}

