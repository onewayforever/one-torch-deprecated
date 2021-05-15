import sys
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader,Dataset
import torch.optim as optim
import matplotlib.pyplot as plt
import datetime
import time
import numpy as np
import signal
import math
import importlib
from collections import defaultdict
import one_torch_utils as otu
import shutil
import torch.distributed as dist
import tqdm
from torch.utils.tensorboard import SummaryWriter
from types import MethodType
import readline
import io
import yaml
#sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030') 

torch.set_printoptions(linewidth=128)
#torch.set_printoptions(threshold=12000)

readline.parse_and_bind('tab: complete')

sig_int_flag=False

ddp_flag=False

tb_writer=None

os.makedirs('experiment_home',exist_ok=True)

now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
logger = None

experiment_id=None
just_a_try=False

start_epoch=0
Experiment=None
experiment_home_dir=None
Runtime={
    '__rank':-1,
    'task':None,
    'total_train_size':0,
    'total_train_batches':0,
    'total_val_size':0,
    'total_val_batches':0,
    'results':{'csv':{},'file':{},'img':{}}
}

HPARAMS_DEFAULT = {
        'n_epochs':100,
        'lr':1e-3,
        'batch_size':64,
        'train_batch_size':64,
        'val_batch_size':64,
        'infer_batch_size':64,
        #'no_cuda':True,
        #'no_cuda':False,
        'loader_n_worker':0,
        'pin_memory':True,
        'optim':'SGD',
        'lr_scheduler':None,
        'early_stop':0,
        'Adam':{'betas':(0.5,0.999)},
        'SGD':{'momentum':0.5},
        'ReduceLROnPlateau':{'mode':'min','patience':3,'factor':0.8},
        'StepLR':{'step_size':5,'gamma':0.8},
        'LambdaLR':{'lr_lambda':lambda x:1-min(99,x)/100},
        'gradient_accumulations':1,
        'seed':12345,
        'local_rank':-1,
        'backend':'nccl',
}

EXPERIMENT_DEFAULT={
        'no_cuda':False,
        'train_dataset_path':None,
        'val_dataset_path':None,
        'infer_dataset_path':None,
        'checkpoint_warmup':5,
        'log_file':'experiment.log',
        'use_checkpoint':-1,
        'checkpoint_n_epoch':0,
        'log_interval':0,
        'experiment_path':None,
        'multi_task_labels':None,
        'use_tensorboard':False,
        'validate_n_batch_in_train':1,     # only validate n batch while training 
        'train_validate_each_n_batch':0, # validate on every n batch 
        'train_validate_each_n_epoch':0,  # validate on every n epoch
        'train_validate_final_with_best':False  # validate at the end of trainning with best model
}
    
best_loss=99999
best_epoch=-1


def sigint_handler(signum, frame):
    global sig_int_flag
    print('\ncatched CTRL-Z signal!\n')
    sig_int_flag=True


signal.signal(signal.SIGTSTP, sigint_handler)

def run_experiment_callback(fn_name,*params):
    #print(fn_name)
    ret={}
    if just_a_try:
        return ret
    global Experiment
    global Runtime 
    if Runtime['__rank']!=0:
        return ret
    fn_info_list = Experiment.get(fn_name)
    if fn_info_list is None:
        return ret
    #print('run {}'.format(fn_name))
    if not isinstance(fn_info_list,list):
        fn_info_list=[fn_info_list]
    for fn_info in fn_info_list:
        if isinstance(fn_info,dict):
            task = Runtime['task']
            if fn_info.get(task):
                fn_info = fn_info[task]
            else:
                continue
        if isinstance(fn_info,tuple):
            if len(fn_info)==1:
                ret=fn(Runtime,Experiment,ret)
            else:
                fn, kwargs=fn_info
                ret=fn(Runtime,Experiment,ret,**kwargs)
        else:
            ret=fn_info(Runtime,Experiment,ret)
        assert ret is not None
    assert ret is not None
    sys.stdout.flush()
    return ret

def epoch_time(elapsed_time):
#def epoch_time(start_time, end_time):
#    elapsed_time = end_time - start_time
    if isinstance(elapsed_time,dict):
        elapsed_time=sum(elapsed_time.values())
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def to_ddp(models):
    global Runtime
    global Experiment
    print('ddp of local_rank {} binding to gpu {}'.format(Runtime['local_rank'],torch.cuda.current_device()))
    logger.info('ddp of local_rank {} binding to gpu {}'.format(Runtime['local_rank'],torch.cuda.current_device()))
    models = list(map(lambda x:torch.nn.parallel.DistributedDataParallel(x,device_ids=[torch.cuda.current_device()],output_device=torch.cuda.current_device()),models))
    #models = list(map(lambda x:torch.nn.SyncBatchNorm.convert_sync_batchnorm(x),models))
    return models

def to_device(tensors,device):
    output=[]
    for item in tensors:
        if isinstance(item,list):
            output.append(to_device(item,device))
        elif isinstance(item,tuple):
            output.append(to_device(item,device))
        elif isinstance(item,torch.Tensor):
            output.append(item.to(device))
        else:
            output.append(item)
    return output

def extract_loss_info(loss_info):
    #print(loss_info)
    info=''
    if isinstance(loss_info,dict):
        for k in loss_info:
            #print(k,loss_info[k])
            if isinstance(loss_info[k],dict):
                for j in loss_info[k]:
                    info+='{}:{:.6f} '.format(k,loss_info[k][j])
            else:
                info+='{}:{:.6f} '.format(k,loss_info[k])
    elif isinstance(loss_info,list):
        if all(map(lambda x:len(x)==1,loss_info)):
            for task in loss_info:
                info+='{}:{:.6f} '.format(k,loss_info[k])
        else:
            for task in loss_info:
                info+='{}  '.format(task)
                for k in loss_info[task]:
                    info+='{}:{:.6f} '.format(k,loss_info[task][k])
    return info

def default_infer_fn(runtime,experiment):
    models = experiment.get('custom_models')
    device = experiment.get('device')
    if len(models)!=1:
        return None,{}
    data = runtime['input']
    #target = runtime['target']
    model = models[0]
    #logger.info(data)
    #print(data.type)
    #print(next(model.parameters()).device)
    output = model(data)
    runtime['output'] = output
    #logger.info(output,target)
    #loss = F.nll_loss(output, target)
    #print(output.shape,target.unsqueeze(1).shape)
    #loss = loss_criterion.to(device)(data,target,output)
    #loss = loss_criterion.to(device)(output,target)
    #with amp.scale_loss(loss, optimizer) as scaled_loss:
    #    scaled_loss.backward()
    return output

def default_val_fn(runtime,experiment):
#def default_val_fn(models,criterions,data,target,device):
    models = experiment.get('custom_models')
    device = experiment.get('device')
    if len(models)!=1:
        return None,{}
    data = runtime['input']
    #target = runtime['target']
    model = models[0]
    #logger.info(data)
    #print(data.type)
    #print(next(model.parameters()).device)
    task = runtime.get('task')
    if task is None:
        output = model(data)
    else:
        output = model(data,task)
    runtime['output'] = output
    #logger.info(output,target)
    #loss = F.nll_loss(output, target)
    #print(output.shape,target.unsqueeze(1).shape)
    #loss = loss_criterion.to(device)(data,target,output)
    loss,loss_value = experiment.get('loss_evaluation_fn')(runtime,experiment)
    #loss = loss_criterion.to(device)(output,target)
    #with amp.scale_loss(loss, optimizer) as scaled_loss:
    #    scaled_loss.backward()
    return output,{'loss':loss_value}


def default_train_fn(runtime,experiment):
    models = experiment.get('custom_models')
    device = experiment.get('device')
    optimizers = experiment.get('custom_optimizers')
    #criterions = experiment.get('loss_criterions')
    data = runtime['input']
    #target = runtime['target']
#def default_train_fn(models,criterions,optimizers,data,target,device):
    assert len(models)==1 and len(optimizers)==1
    assert experiment.get('loss_evaluation_fn')
    model = models[0]
    optimizer = optimizers[0]
    optimizer.zero_grad()
    #logger.info(data)
    #print(data.type)
    #print(next(model.parameters()).device)
    task = runtime.get('task')
    if task is None:
        output = model(data)
    else:
        output = model(data,task)
    runtime['output'] = output
    #logger.info(output,target)
    #loss = F.nll_loss(output, target)
    #print(output.shape,target.unsqueeze(1).shape)
    #loss = loss_criterion.to(device)(data,target,output)
    loss,loss_value = experiment.get('loss_evaluation_fn')(runtime,experiment)
    #loss = loss_criterion.to(device)(output,target)
    #with amp.scale_loss(loss, optimizer) as scaled_loss:
    #    scaled_loss.backward()
    loss.backward()
    optimizer.step()
    return output,{'loss':loss_value}

def default_data_preprocess_fn(runtime,experiment,original_data,task=None):
    return original_data[0],original_data[1]

def default_evaluation_fn(runtime,experiment):
    output = runtime.get('output')
    target = runtime.get('target')
    loss_criterions = experiment.get('loss_criterions')
    assert len(loss_criterions) == 1
    loss_criterion = loss_criterions[0]
    #print(output.shape,target.shape)
    if isinstance(output,tuple):
        output = output[0]
    loss = loss_criterion(output,target) 
    return loss,loss.item()
    

def save_train_epoch_list(epoch_list):
    if just_a_try:
        return
    if Runtime['__rank']!=0:
        return
    length=len(epoch_list)
    if length==0:
        return 
    learning_curve_path=os.path.join(experiment_home_dir,'learning_curve.png')
    logger.info('Save Learning curve path to {}'.format(learning_curve_path))
    plt.title('Result Analysis')
    for source in ['train','val']:
        source_list=list(map(lambda x:x[source],epoch_list))
        if source_list[0] is None:
            continue
        for k in source_list[0]:
            if 'loss' in k:
                plt.plot(list(range(length)), np.array(list(map(lambda x:x[k],source_list))),label='_'.join([source,k]))
    plt.legend()  
    plt.xlabel('iteration times')
    plt.ylabel('loss')
    plt.savefig(learning_curve_path)
    if tb_writer:
        first = epoch_list[0]
        all_keys=list(set(list(first['train'].keys())+list(first['val'].keys())))
        for i in range(len(epoch_list)):
            item=epoch_list[i]
            for k in all_keys:
                if item['train'].get(k) and item['val'].get(k):
                    tb_writer.add_scalars(k,{'train':item['train'].get(k),'val':item['val'].get(k)},i)
                elif item['train'].get(k) : 
                    tb_writer.add_scalars(k,{'train':item['train'].get(k)},i)
    

def epoch_train(Runtime,Experiment):
    HPARAMS=Experiment['hparams']
    models = Experiment['custom_models']
    train_loader = Experiment['train_loader']
    val_loader = Experiment['val_loader']
    device = Experiment['device']
    list(map(lambda x:x.train(),models))
    train_loss_info=defaultdict(lambda: 0) 
    loss_info={}
    Runtime['epoch_results']=[]
    run_experiment_callback('pre_epoch_train_fn')
    total_num=0
    if Experiment['multi_task_labels'] is not None:
        #print('create Multi task loader')
        train_loader = otu.MTLoader(train_loader)
    for batch_idx, original_data in enumerate(tqdm.tqdm(train_loader, ncols=100,desc="Training Epoch {}".format(Runtime['epoch']))):
        if sig_int_flag:
            break
        if Experiment['multi_task_labels'] is None:
            data, target = Experiment['data_preprocess_fn'](Runtime,Experiment,original_data)
            task = None
        else:
            task_data,task = original_data
            data, target = Experiment['data_preprocess_fn'](Runtime,Experiment,task_data,task)
        Runtime['task']=task
        #if batch_idx==0:
        #    print(target)
        [data, target] = to_device([data,target],device)
        #data, target = data.to(device), target.to(device)
        Runtime['input']=data
        Runtime['target']=target
        Runtime['batch_idx']=batch_idx
        output, loss_info = Experiment['custom_train_fn'](Runtime,Experiment)
        Runtime['batches_done']+=1
        Runtime['output']=output
        Runtime['batch_train_loss_info']=loss_info
        assert isinstance(loss_info,dict) 
        for k in loss_info:
            if 'loss' in k:
                train_loss_info[k]+=loss_info[k] 
        total_num+=1

        batch_info=run_experiment_callback('post_batch_train_fn')
        if batch_info is not None:
            Runtime['epoch_results'].append(batch_info)
        
        if val_loader is not None  and Experiment.get('train_validate_each_n_batch')>0:
            if Runtime['batches_done']%Experiment['train_validate_each_n_batch']==0:
                validate(Runtime,Experiment)

    epoch_insight_result = run_experiment_callback('post_epoch_train_fn')

    for k in loss_info:
        train_loss_info[k]/=total_num

    return train_loss_info,epoch_insight_result

def validate(Runtime,Experiment):
    all_tasks = Experiment.get('multi_task_labels')
    if all_tasks is None:
        start_time = time.time()
        loss_info,insight_result = validate_task(Runtime,Experiment,task=None)
        end_time = time.time()
        return end_time-start_time,loss_info,insight_result 
    task_loss = {}
    task_time = {}
    task_insight_result = {}
    for task in all_tasks:
        start_time = time.time()
        loss_info,insight_result = validate_task(Runtime,Experiment,task=task)
        end_time = time.time()
        task_loss[task] = loss_info
        task_insight_result[task] = insight_result
        task_time[task]=end_time-start_time
    return task_time,task_loss,task_insight_result


def validate_task(Runtime,Experiment,task=None):
    models = Experiment.get('custom_models')
    val_loader = Experiment.get('val_loader')
    if task is None:
        dataloader = val_loader
    else:
        dataloader = val_loader.get(task) 
    if dataloader is None:
        print('No validate loader for task {}'.format(task))
        return 0,{},{}
    device = Experiment['device']
    list(map(lambda x:x.eval(),models))
    val_loss_info=defaultdict(lambda: 0) 
    #model.eval()
    #val_loss = 0
    result_list=[]
    loss_info={}
    #guess_list=[] ##
    #epoch_insight_record_fn=Experiment.get('val_epoch_insight_record_fn')
    #epoch_insight_fn=Experiment.get('val_epoch_insight_fn')
    Runtime['epoch_results']=[]
    run_experiment_callback('pre_epoch_val_fn')
    if Runtime['step']=='validate':
        run_experiment_callback('validate_pre_epoch_val_result_fn')
    validate_n_batch_in_train = Experiment['validate_n_batch_in_train']
    total_num=0
    with torch.no_grad():
        for batch_idx, original_data in enumerate(dataloader):
            #print(batch_idx)
            if Runtime['step']=='train' and sig_int_flag:
                break
            if Runtime['step']=='train' and validate_n_batch_in_train > 0 and batch_idx >= validate_n_batch_in_train :
                break
            if task is None:
                data, target = Experiment['data_preprocess_fn'](Runtime,Experiment,original_data)
            else:
                data, target = Experiment['data_preprocess_fn'](Runtime,Experiment,original_data,task)
            [data, target] = to_device([data,target],device)
            Runtime['input']=data
            Runtime['target']=target
            Runtime['batch_idx']=batch_idx
            Runtime['task']=task
            output, loss_info = Experiment['custom_val_fn'](Runtime,Experiment)
            assert isinstance(loss_info,dict) 
            Runtime['output']=output
            Runtime['batch_val_loss_info']=loss_info
   
            for k in loss_info:
                if 'loss' in k:
                    val_loss_info[k]+=loss_info[k] 
            total_num+=1
        
            batch_info=run_experiment_callback('post_batch_val_fn')
            if batch_info is not None:
                Runtime['epoch_results'].append(batch_info)

            if Runtime['step']=='validate':
                run_experiment_callback('validate_batch_val_result_fn')
            #if epoch_insight_record_fn:
            #    #Runtime['epoch_results'].append(epoch_insight_record_fn(Runtime))
            #    batch_info=epoch_insight_record_fn(Runtime,Experiment)
            #    if batch_info is not None:
            #        Runtime['epoch_results'].append(batch_info)
                #result_list.append((output.clone().detach().cpu(),target.clone().detach().cpu()))
                #guess_list.append((torch.rand(output.shape),target.clone().detach().cpu()))  ##

    
    #if epoch_insight_fn and len(Runtime['epoch_results'])>0:
    #    #print(result_list)
    #    epoch_insight_result = run_experiment_callback('val_epoch_insight_fn')
    #    #epoch_insight_result = epoch_insight_fn(Runtime)
    #    #epoch_insight_result += epoch_insight_fn(guess_list)  ##
    epoch_insight_result = run_experiment_callback('post_epoch_val_fn')

    #total_num=len(val_loader)
    for k in loss_info:
        val_loss_info[k]/=total_num
            
    if Runtime['step']=='validate':
        run_experiment_callback('validate_val_result_fn')

    #print('end validate')
    return val_loss_info,epoch_insight_result

def inference(Runtime,Experiment):
    models = Experiment.get('custom_models')
    infer_loader = Experiment.get('infer_loader')
    device = Experiment['device']
    list(map(lambda x:x.eval(),models))
    #val_loss_info=defaultdict(lambda: 0) 
    #model.eval()
    #val_loss = 0
    result_list=[]
    loss_info={}
    #guess_list=[] ##
    #epoch_insight_result=None
    #epoch_insight_record_fn=Experiment.get('val_epoch_insight_record_fn')
    #epoch_insight_fn=Experiment.get('val_epoch_insight_fn')
    #Runtime['epoch_results']=[]
    with torch.no_grad():
        for batch_idx, original_data in enumerate(infer_loader):
            #print(batch_idx)
            if sig_int_flag:
                break
            data = Experiment['infer_data_preprocess_fn'](Runtime,Experiment,original_data)
            [data]  = to_device([data],device)
            #print(data)
            #print(type(data))
            Runtime['input']=data
            Runtime['target']=None
            Runtime['batch_idx']=batch_idx
            output = Experiment['custom_infer_fn'](Runtime,Experiment)
            Runtime['output']=output
            run_experiment_callback('post_batch_infer_fn')

    return 

def preview_dataset(dataset,loader):
    global Runtime 
    global Experiment

    def preview_batch(self,arg):
        if loader is not None:
            try:
                n = int(arg)
                assert n>=0 and n<=max_batch
            except Exception as e:
                print('please input an index number [0 - {}]'.format(max_batch))
                return
            iterator = iter(loader)
            while True: 
                batch=next(iterator)
                n-=1
                if n<=0:
                    break
            batch = Experiment['data_preprocess_fn'](Runtime,Experiment,batch)  
            #print(batch)
            otu.preview_tensor(batch)
            Experiment['preview_batch_item'](batch)

    def preview_mtl_batch(self,args):
        try:
            task,idx = args.split()
            n = int(idx)
        except Exception as e:
            print('Format: <task_name>  <index>')
            return
        if task not in all_labels:
            print('task should in {}'.format(all_labels))
            return
        if loader is not None:
            iterator = iter(loader[task])
            while True: 
                batch=next(iterator)
                n-=1
                if n<=0:
                    break
            batch = Experiment['data_preprocess_fn'](Runtime,Experiment,batch,task)  
            #print(batch)
            otu.preview_tensor(batch)
            Experiment['preview_batch_item'](batch)
    def preview_nth_data(self,arg):
        try:
            n = int(arg)
            assert n>=0 and n<=max_index
        except Exception as e:
            print('please input an index number [0 - {}]'.format(max_index))
            return
        Experiment['preview_dataset_item'](dataset[n])
    
    def preview_mtl_nth_data(self,args):
        try:
            task,idx = args.split()
            n = int(idx)
        except Exception as e:
            print('Format: <task_name>  <index>')
            return
        if task not in all_labels:
            print('task should in {}'.format(all_labels))
            return
        if n<0 or n>max_indices[task]:
            print('please input an index number [0 - {}]'.format(max_indices[task]))
            return
        Experiment['preview_dataset_item'](dataset[task][n],task)

    if loader is not None:
        all_labels = Experiment.get("multi_task_labels")
        if all_labels:
            otu.CLI.do_batch = preview_mtl_batch
            otu.CLI.do_b = otu.CLI.do_batch 
        else:
            max_batch=len(loader)-1
            otu.CLI.do_batch = preview_batch
            otu.CLI.do_b = otu.CLI.do_batch 
    if dataset is not None:
        all_labels = Experiment.get("multi_task_labels")
        if all_labels:
            assert isinstance(dataset,dict)
            max_indices={}
            for task in all_labels:
                max_indices[task]=len(dataset[task])-1
            otu.CLI.do_data = preview_mtl_nth_data
            otu.CLI.do_d = otu.CLI.do_data
        else:
            max_index=len(dataset)-1
            otu.CLI.do_data = preview_nth_data
            otu.CLI.do_d = otu.CLI.do_data
    cli=otu.CLI('preview > ')
    cli.cmdloop(intro=" ==== Preview Dataset ==== ")

def interact_wrapper():
    global Runtime
    global Experiment
    prompt = Experiment.get('interact_prompt')
    if prompt is None:
        prompt = "#> "
    intro= Experiment.get('interact_intro')
    if intro is None:
        intro = " Interact "
    cmds = Experiment.get('interact_cmd')
    if isinstance(cmds,list):
        for cmd,function,helper in cmds:
            def cli_function(self,args):
                return function(Runtime,Experiment,args)
            def cli_help(self):
                print(helper)
            setattr(otu.CLI,'do_{}'.format(cmd),cli_function)
            setattr(otu.CLI,'help_{}'.format(cmd),cli_help)
    else:
        print('No cli cmd registered, please check')
        return
    cli=otu.CLI(prompt)
    cli.cmdloop(intro=" ==== {} ==== ".format(intro))

def init_experiment(action):
    global Experiment
    global Runtime
    Runtime['step']='init'
    HPARAMS=Experiment['hparams']
    custom_models=Experiment.get('custom_models')
    loss_criterions=Experiment['loss_criterions']
    torch.manual_seed(HPARAMS['seed'])
    use_cuda = not Experiment['no_cuda'] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    #gpu_count = torch.cuda.device_count()
    #print('gpu_count',gpu_count)
    #for i in range(gpu_count):
    #    print('gpu:',torch.cuda.get_device_name(i))
    #print('device',device)
    #print('current device',torch.cuda.current_device())
    Experiment['device']=device
    local_rank=HPARAMS['local_rank']

    if use_cuda:
        torch.cuda.set_device(local_rank)

    '''
    if HPARAMS['ddp']:
        dist.init_process_group(backend=HPARAMS['backend'])
        print('init ddp',dist.get_rank())
        Runtime['__rank']=dist.get_rank()
    else:
        Runtime['__rank']=0
    '''

    ## load dataset
    '''
    if Runtime['action']=='train':
        train_path=HPARAMS['dataset'].split(',')[0]
        val_path=HPARAMS['dataset'].split(',')[1] if len(HPARAMS['dataset'].split(','))>1 else None
    else:
        train_path=None
        val_path=HPARAMS['dataset'].split(',')[0]
    '''
    #if just_a_try:
    #    HPARAMS['batch_size']=max(1,int(HPARAMS['batch_size']/20))
    #    HPARAMS['train_batch_size']=max(1,int(HPARAMS['train_batch_size']/20))
    #    HPARAMS['val_batch_size']=max(1,int(HPARAMS['val_batch_size']/20))
    #print(HPARAMS)
    train_path=Experiment['train_dataset_path']
    val_path=Experiment['val_dataset_path']
    infer_path=Experiment['infer_dataset_path']
    #print(train_path)
    #print(val_path)
    #print(infer_path)
    ## Load Dataset ##
    train_dataset=None
    val_dataset=None
    infer_dataset=None
    train_loader=None
    val_loader=None
    infer_loader=None
    train_sampler=None
    val_sampler=None
    infer_sampler=None
    if train_path is not None:
        if Experiment.get('create_train_dataset_fn'):
            train_dataset=Experiment['create_train_dataset_fn'](train_path)
        #if Experiment.get('create_train_loader_fn'):
        #    train_loader=Experiment['create_train_loader_fn'](train_path)
    if val_path is not None:
        #print(val_path)
        if Experiment.get('create_val_dataset_fn'):
            val_dataset=Experiment['create_val_dataset_fn'](val_path)
        #if Experiment.get('create_val_loader_fn'):
        #    val_loader=Experiment['create_val_loader_fn'](val_path)
    if infer_path is not None:
        if Experiment.get('create_infer_dataset_fn'):
            infer_dataset=Experiment['create_infer_dataset_fn'](infer_path)
            #print(infer_dataset)
        #if Experiment.get('create_infer_loader_fn'):
        #    infer_loader=Experiment['create_infer_loader_fn'](infer_path)

    if train_dataset:
        if isinstance(train_dataset,dict):
            Runtime['total_train_size']={}
            for k in train_dataset:
                Runtime['total_train_size'][k]=len(train_dataset[k])
        else:
            Runtime['total_train_size']=len(train_dataset)
        logger.info('Train samples:{}'.format(Runtime['total_train_size']))
    if val_dataset:
        if isinstance(val_dataset,dict):
            Runtime['total_val_size']={}
            for k in val_dataset:
                Runtime['total_val_size'][k]=len(val_dataset[k])
        else:
            Runtime['total_val_size']=len(val_dataset)
        logger.info('Val samples:{}'.format(Runtime['total_val_size']))
    if infer_dataset:
        if isinstance(infer_dataset,dict):
            Runtime['total_infer_size']={}
            for k in infer_dataset:
                Runtime['total_infer_size'][k]=len(infer_dataset[k])
        else:
            Runtime['total_infer_size']=len(infer_dataset)
        logger.info('Infer samples:{}'.format(Runtime['total_infer_size']))



    def create_collate_fn(dataset,which=''):
        if Experiment.get('collate_{}fn_by_dataset'.format(which)):
            return Experiment.get('collate_{}fn_by_dataset'.format(which))(dataset) 
        elif Experiment.get('collate_{}fn'.format(which)):
            return  Experiment.get('collate_{}fn'.format(which))
        return None

    kw_data_loader_args={'pin_memory':HPARAMS['pin_memory']}
    if HPARAMS['loader_n_worker']>0:
        kw_data_loader_args['num_workers']=HPARAMS['loader_n_worker'] 
   
    #print('dataset',train_dataset)

    if ddp_flag:
        if train_loader is None and train_dataset is not None:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            fn=create_collate_fn(train_dataset)
            if Experiment.get('create_train_loader_fn'):
                train_loader = Experiment['create_train_loader_fn'](train_dataset,batch_size=HPARAMS['train_batch_size'])
            else:
                train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=HPARAMS['train_batch_size'], shuffle=False, sampler=train_sampler,collate_fn=fn,**kw_data_loader_args,drop_last=True)
        if val_loader is None and val_dataset is not None:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
            fn=create_collate_fn(val_dataset)
            if Experiment.get('create_val_loader_fn'):
                val_loader = Experiment['create_val_loader_fn'](val_dataset,batch_size=HPARAMS['val_batch_size'])
            else:
                val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=HPARAMS['val_batch_size'], shuffle=False, sampler=val_sampler,collate_fn=fn,**kw_data_loader_args)
        if infer_loader is None and infer_dataset is not None:
            infer_sampler = torch.utils.data.distributed.DistributedSampler(infer_dataset)
            fn=create_collate_fn(infer_dataset,'infer_')
            if Experiment.get('create_infer_loader_fn'):
                infer_loader = Experiment['create_infer_loader_fn'](infer_dataset,batch_size=HPARAMS['infer_batch_size'])
            else:
                infer_loader = torch.utils.data.DataLoader(infer_dataset,batch_size=HPARAMS['infer_batch_size'], shuffle=False, sampler=infer_sampler,collate_fn=fn,**kw_data_loader_args,drop_last=False)
    else:
        if train_loader is None and train_dataset is not None:
            #print('dataset',train_dataset)
            fn=create_collate_fn(train_dataset)
            if Experiment.get('create_train_loader_fn'):
                train_loader = Experiment['create_train_loader_fn'](train_dataset,batch_size=HPARAMS['train_batch_size'])
            else:
                train_loader = DataLoader(train_dataset, **kw_data_loader_args, batch_size=HPARAMS['train_batch_size'],shuffle=True,collate_fn=fn,drop_last=True)
                #print('loader',train_loader)
        if val_loader is None and val_dataset is not None:
            fn=create_collate_fn(val_dataset)
            if Experiment.get('create_val_loader_fn'):
                val_loader = Experiment['create_val_loader_fn'](val_dataset,batch_size=HPARAMS['val_batch_size'])
            else:
                val_loader = DataLoader(val_dataset, **kw_data_loader_args, batch_size=HPARAMS['val_batch_size'],shuffle=False,collate_fn=fn,drop_last=False)
        if infer_loader is None and infer_dataset is not None:
            fn=create_collate_fn(infer_dataset,'infer_')
            if Experiment.get('create_infer_loader_fn'):
                infer_loader = Experiment['create_infer_loader_fn'](infer_dataset,batch_size=HPARAMS['infer_batch_size'])
            else:
                infer_loader = DataLoader(infer_dataset, **kw_data_loader_args, batch_size=HPARAMS['infer_batch_size'],shuffle=False,collate_fn=fn,drop_last=False)


    if action=='train':
        assert train_loader
        Runtime['total_train_batches']=len(train_loader)
    if action=='val':
        assert val_loader
        Runtime['total_val_batches']=len(val_loader)
    if action=='infer':
        assert infer_loader
        Runtime['total_infer_batches']=len(infer_loader)
    if action=='view':
        if train_dataset is not None or train_loader is not None:
            preview_dataset(train_dataset,train_loader)
        else:
            print('ERROR: No Dataset loaded, Please check!')
        return
   
    print("done load data") 

    if custom_models is None and Experiment.get('create_custom_models_fn') is not None:
        custom_models = Experiment.get('create_custom_models_fn')(Runtime,Experiment)
        Experiment['custom_models'] = custom_models
    
    print("done create model") 

    if custom_models is None:
        print('Error: Need define torch models, by custom_models objects  or by create_custom_models_fn')
        exit(0)


    list(map(lambda x:x.to(device),custom_models))
    list(map(lambda x:x.to(device),loss_criterions))

    

    if ddp_flag:
        to_ddp(custom_models)
    logger.info("### custom models info ###")
    total_parameters=0
    total_parameters_size=0
    trainable_parameters=0
    trainable_parameters_size=0
    for model in custom_models:
        paramter_info= otu.insight_parameters(model)
        print(paramter_info)
        logger.info("### model info ###")
        logger.info(repr(model))
        logger.info(model._get_name())
        logger.info("### Model {} Parameters:{} size:{}MB\tTrainable Parameters:{} size:{}MB###".format(model._get_name(),
                               paramter_info['total_parameters'],
                               int(paramter_info['total_parameters_size']/1024/1024),
                               paramter_info['trainable_parameters'],
                               int(paramter_info['trainable_parameters_size']/1024/1024)))
        total_parameters+=paramter_info['total_parameters']
        total_parameters_size+=paramter_info['total_parameters_size']
        trainable_parameters+=paramter_info['trainable_parameters']
        trainable_parameters_size+=paramter_info['trainable_parameters_size']
        try:
            if tb_writer:
                tb_writer.add_graph(model)
        except Exception as e:
            #fails all the time, don't know why
            pass
    logger.info("### Total Parameters:{} size:{}MB\tTrainable Parameters:{} size:{}MB###".format(total_parameters,
                         int(total_parameters_size/1024/1024),
                         trainable_parameters,
                         int(trainable_parameters_size/1024/1024)))
    logger.info("### loss criterion info ###")
    for loss_criterion in loss_criterions:
        logger.info(repr(loss_criterion))

    Experiment['train_loader']=train_loader
    Experiment['val_loader']=val_loader
    Experiment['infer_loader']=infer_loader
    Experiment['train_sampler']=train_sampler
    Experiment['val_sampler']=val_sampler
    Experiment['infer_sampler']=infer_sampler


    Runtime['batch_size']=HPARAMS['batch_size']
    Runtime['train_batch_size']=HPARAMS['train_batch_size']
    Runtime['val_batch_size']=HPARAMS['val_batch_size']
    Runtime['infer_batch_size']=HPARAMS['infer_batch_size']

def log_action_info(work_time,loss_info,insight_info,mode='val',prefix='\t'):
    if isinstance(work_time,dict):
        tasks = work_time.keys()
        for task in tasks:
            logger.info('Epoch {} Val Detail:\n'.format(epoch)+str(insight_info[task].get('display')))
    else:
        logger.info('Epoch {} Val Detail:\n'.format(epoch)+str(insight_info.get('display')))
        

def train_wrapper():
    global Runtime
    global Experiment
    global best_loss
    global best_epoch
    Runtime['runtime_id'] = 'train_{}'.format(now) 
    init_experiment('train')
    HPARAMS=Experiment['hparams']
    custom_models = Experiment['custom_models']
    custom_optimizers = Experiment['custom_optimizers']
    custom_schedulers = Experiment['custom_schedulers']
    #global loss_criterions
    custom_parameters = Experiment['custom_parameters']
    train_loader = Experiment['train_loader']
    val_loader = Experiment['val_loader']

    assert train_loader is not None

    if len(custom_parameters) == 0:
        for model in custom_models:
            parameter = model.parameters()
            custom_parameters.append(parameter)

    for parameter in custom_parameters:
        if HPARAMS['optim'] == 'SGD':
            optimizer = optim.SGD(parameter, lr=HPARAMS['lr'], **HPARAMS['SGD'])
        elif HPARAMS['optim'] == 'Adam':
            optimizer = optim.Adam(parameter, lr=HPARAMS['lr'],**HPARAMS['Adam'])

        lr_scheduler_info=HPARAMS['lr_scheduler']
        if lr_scheduler_info =='ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **HPARAMS['ReduceLROnPlateau'])
        elif lr_scheduler_info == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer, **HPARAMS['StepLR'])
        elif lr_scheduler_info == 'LambdaLR':
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, **HPARAMS['LambdaLR'])
        else:
            scheduler = None
        custom_optimizers.append(optimizer)
        custom_schedulers.append(scheduler)

    current_lr = HPARAMS['lr']
    
    stop_loss=999999
    patience = HPARAMS['early_stop']
    val_loss=0
    train_insight={}
    val_insight={}
    sep='\n\t'
    #sep='| '
    train_epoch_list=[]
    print('Start Train iteration...')
    Runtime['step']='train'
    Runtime['train_batches']=len(train_loader)
    train_loss_info={}
    val_loss_info={}
    run_experiment_callback('pre_train_fn')
    Runtime['batches_done']=0
    checkpoint_n_epoch=Experiment.get('checkpoint_n_epoch') if Experiment.get('checkpoint_n_epoch') else 0
    check_loss=None
    for epoch in range(start_epoch,HPARAMS['n_epochs']):
        if sig_int_flag:
            print('Exit Training\n')
            break
        if Experiment['train_sampler']:
            Experiment['train_sampler'].set_epoch(epoch)
        Runtime['epoch']=epoch
        Runtime['trace']='epoch_train'
        start_time = time.time()
        train_loss_info,train_insight = epoch_train(Runtime,Experiment)
        if sig_int_flag:
            print('Exit Training\n')
            break
        #train_loss_info,train_insight = epoch_train(args, custom_models, loss_criterions,device, train_loader, custom_optimizers, epoch)
        epoch_loss=train_loss_info.get('loss')
        if HPARAMS['early_stop'] > 0:
            if epoch_loss:
                check_loss = epoch_loss 
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(end_time - start_time)
        for scheduler in custom_schedulers:
            if lr_scheduler_info == 'ReduceLROnPlateau' and train_loss_info.get('loss'):
                scheduler.step(train_loss_info.get('loss'))
            elif lr_scheduler_info is not None:
                scheduler.step()
            if scheduler:
                if scheduler._last_lr != current_lr:
                    print('use learning rate',scheduler._last_lr)
                    current_lr = scheduler._last_lr

        if val_loader and Experiment.get('train_validate_each_n_epoch')>0:
            if epoch%Experiment.get('train_validate_each_n_epoch')==0:
                Runtime['trace']='epoch_val'
                #print('run validate per epoch')
                val_time,val_loss_info,val_insight = validate(Runtime,Experiment)
                #val_loss_info,val_insight = validate(args, custom_models, loss_criterions,device, val_loader,n_batch=HPARAMS['epoch_val_n_batch'],from_epoch=epoch)
                if isinstance(val_time,dict):
                    epoch_loss = sum(map(lambda x:x[1].get('loss'),val_loss_info.items()))
                else:
                    epoch_loss = val_loss_info.get('loss')
                if HPARAMS['early_stop'] > 0:
                    if epoch_loss:
                        check_loss = epoch_loss
        save_flag=False
        if epoch_loss:
            if epoch_loss < best_loss:
                best_epoch=epoch
                best_loss = epoch_loss
                save_flag=True
                save_models()
        if checkpoint_n_epoch>0 and (epoch % checkpoint_n_epoch==0):
            save_flag=True
            if epoch >= Experiment['checkpoint_warmup']:
                save_checkpoints(epoch)
        #if save_flag and epoch >= Experiment['checkpoint_warmup']:
        #    save_checkpoints(epoch)
        loss_str=extract_loss_info(train_loss_info)
        epoch_log=f'Epoch: {epoch:02} | Elapse: {epoch_mins}m {epoch_secs}s |\tTrain: {loss_str} '
        if val_loader and Experiment['train_validate_each_n_epoch']>0:
            loss_str=extract_loss_info(val_loss_info)
            epoch_log += f'Val: {loss_str} '
        logger.info(epoch_log)
        if train_insight.get('display'):
            logger.info('Epoch {} Train Detail:\n'.format(epoch)+str(train_insight.get('display')))
        if val_insight.get('display'):
            logger.info('Epoch {} Val Detail:\n'.format(epoch)+str(val_insight.get('display')))
        elif Experiment.get('multi_task_labels'):
            logger.info('Epoch {} Val Detail:\n'.format(epoch))
            for task in Experiment['multi_task_labels']:
                if val_insight.get(task):
                    if val_insight[task].get('display'):
                        logger.info('{}\t'.format(task)+val_insight[task]['display'])
            #epoch_log+=(sep+'Val { '+str(val_insight)+' } ')
        
        #logger.info(epoch_log)
        train_epoch_list.append({'train':train_loss_info,'val':val_loss_info})
        #use early stop
        if HPARAMS['early_stop'] > 0:
            if check_loss is None:
                print('ERROR: loss field should be set in train_loss_info or val_loss_info, Check code or disable early_stop')
                exit(0)
            if check_loss < stop_loss:
                stop_loss = check_loss
                patience = HPARAMS['early_stop']
            else:
                patience-=1            
            if patience==0:
                logger.info("Evoke Early Stopping!")
                break
    logger.info('Train {} Epochs'.format(epoch)) 
    run_experiment_callback('post_train_fn')
    save_train_epoch_list(train_epoch_list)

def show_results():
    global Runtime
    #print(Runtime['results'])
    results_path=[]
    for k in Runtime['results']:
        item = Runtime['results'][k]
        for tag in item:
            results_path.append(item[tag]['path'])
    #print(results_path)
    if len(results_path)>0:
        logger.info('Save results to')
        for path in results_path:
            logger.info('--> {}'.format(path))

def validate_wrapper():
    global Runtime
    global Experiment

    init_experiment('val')
    start_time = time.time()
    Runtime['runtime_id'] = 'validate_{}'.format(now) 
    Runtime['step']='validate'
    run_experiment_callback('pre_val_fn')
    task_time,val_loss_info,val_insight = validate(Runtime,Experiment)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(task_time)
    loss_str=extract_loss_info(val_loss_info)
    val_log=f'Validate Elapse: {epoch_mins}m {epoch_secs}s |\tLoss: {loss_str} '
    logger.info(val_log)
    if val_insight.get('display'):
        logger.info(val_insight.get('display'))
    elif Experiment.get('multi_task_labels'):
        for task in Experiment['multi_task_labels']:
            if val_insight.get(task):
                if val_insight[task].get('display'):
                    logger.info('{}'.format(task)+val_insight[task]['display'])
    run_experiment_callback('post_val_fn')
    show_results()

def final_validate():
    global Runtime
    global Experiment
    Runtime['step']='validate'
    logger.info('Start Final Validate')
    task_time,val_loss_info,val_insight = validate(Runtime,Experiment)
    loss_str=extract_loss_info(val_loss_info)
    epoch_mins, epoch_secs = epoch_time(task_time)
    val_log=f'Validate Elapse: {epoch_mins}m {epoch_secs}s |\tLoss: {loss_str} '
    logger.info(val_log)
    if val_insight.get('display'):
        logger.info(val_insight.get('display'))
    elif Experiment.get('multi_task_labels'):
        for task in Experiment['multi_task_labels']:
            if val_insight.get(task):
                if val_insight[task].get('display'):
                    logger.info('{}'.format(task)+val_insight[task]['display'])
    show_results()

def inference_wrapper():
    global Runtime

    init_experiment('infer')
    start_time = time.time()
    Runtime['runtime_id'] = 'inference_{}'.format(now) 
    Runtime['step']='inference'
    run_experiment_callback('pre_infer_fn')
    inference(Runtime,Experiment)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(end_time - start_time)
    infer_log=f'Inference Elapse: {epoch_mins}m {epoch_secs}s'
    logger.info(infer_log)
    run_experiment_callback('post_infer_fn')
    show_results()


def parse_experiment_info(Experiment):
    #custom_models=Experiment.get('custom_models')
    #assert custom_models is not None
    if Experiment.get('data_preprocess_fn') is None:
        Experiment['data_preprocess_fn'] = default_data_preprocess_fn
    if Experiment.get('custom_train_fn') is None:
        Experiment['custom_train_fn'] = default_train_fn
    if Experiment.get('custom_val_fn') is None:
        Experiment['custom_val_fn'] = default_val_fn
    if Experiment.get('custom_infer_fn') is None:
        Experiment['custom_infer_fn'] = default_infer_fn
    if Experiment.get('custom_parameters') is None:
        Experiment['custom_parameters'] = []
    if Experiment.get('loss_criterions') is None:
        Experiment['loss_criterions'] = []
    if Experiment.get('loss_evaluation_fn') is None:
        Experiment['loss_evaluation_fn'] = default_evaluation_fn
    if Experiment.get('preview_dataset_item') is None:
        Experiment['preview_dataset_item'] = print 
    if Experiment.get('preview_batch_item') is None:
        Experiment['preview_batch_item'] = otu.preview_tensor
    if Experiment.get('hparams') is None:
        Experiment['hparams']=HPARAMS_DEFAULT
    else:
        hparams=Experiment.get('hparams')
        for k in hparams.keys():
            HPARAMS_DEFAULT[k]=hparams[k]

        if hparams.get('batch_size'):
            HPARAMS_DEFAULT['batch_size']=hparams.get('batch_size')
            HPARAMS_DEFAULT['train_batch_size']=hparams.get('batch_size')
            HPARAMS_DEFAULT['val_batch_size']=hparams.get('batch_size')
            HPARAMS_DEFAULT['infer_batch_size']=hparams.get('batch_size')
            
        if hparams.get('train_batch_size'):
            HPARAMS_DEFAULT['train_batch_size']=hparams.get('train_batch_size')

        if hparams.get('val_batch_size'):
            HPARAMS_DEFAULT['val_batch_size']=hparams.get('val_batch_size')

        if hparams.get('infer_batch_size'):
            HPARAMS_DEFAULT['infer_batch_size']=hparams.get('infer_batch_size')

        Experiment['hparams']=HPARAMS_DEFAULT
    if Experiment.get('custom_optimizers') is None:
        Experiment['custom_optimizers']=[]
    if Experiment.get('custom_schedulers') is None:
        Experiment['custom_schedulers']=[]
    for k in EXPERIMENT_DEFAULT:
        if Experiment.get(k) is None:
            Experiment[k]=EXPERIMENT_DEFAULT[k]
        

def load_models(path):
    global Runtime
    global Experiment
    if Experiment.get('pre_load_models') is not None:
        Experiment.get('pre_load_models')(path)
    custom_models=Experiment.get('custom_models')
    if custom_models is None and Experiment.get('create_custom_models_fn') is not None:
        if Experiment.get('load_model_dynamic_params') is not None:
            with open(os.path.join(path,"config.json")) as f:
                data = json.load(f)
            Experiment.get('load_model_dynamic_params')(data)
        custom_models = Experiment.get('create_custom_models_fn')(Runtime,Experiment)
        Experiment['custom_models'] = custom_models

    for idx,model in enumerate(custom_models):
        model_name = model._get_name()
        #print(model_name,os.path.join(path,"{}_{}.pt".format(model_name,idx)))
        pickle=torch.load(os.path.join(path,"{}_{}.pt".format(model_name,idx)))
        #print(pickle.keys())
        #for k in pickle:
        #    print(k,pickle[k].shape)
        model.load_state_dict(pickle)
        #model.load_state_dict(torch.load(os.path.join(path,"{}_{}.pt".format(model_name,idx))))
    if Experiment.get('post_load_models') is not None:
        Experiment.get('post_load_models')(path)


def load_checkpoints():
    global start_epoch
    global Experiment
    HPARAMS=Experiment['hparams']
    # init weight by checkpoints
    former_experiment_dir=Experiment.get('experiment_path')
    if former_experiment_dir:
        checkpoints_path=os.path.join(former_experiment_dir,'checkpoints')
        try:
            all_checkpoints = list(map(lambda x:int(x),os.listdir(checkpoints_path)))
            last_checkpont=max(all_checkpoints)
        except Experiment as e:
            print('## EXIT ##  No checkpoints found at experiment_dir:{}'.format(former_experiment_dir))
            exit(0)
        use_checkpoint = Experiment.get('use_checkpoint')
        if use_checkpoint<0:
            print('use last checkpoints {} of experiment_dir:{}'.format(last_checkpont,former_experiment_dir));
            use_checkpoint=last_checkpont
        elif use_checkpoint not in all_checkpoints:
            print('## EXIT ##  checkpoint {} not exist in experiment_dir:{}'.format(use_checkpoint,former_experiment_dir))
            exit(0)
        print('Use checkpoint {} of experiment_dir:{}'.format(use_checkpoint,former_experiment_dir))
        load_models(os.path.join(checkpoints_path,f'{use_checkpoint}'))
        #
        #custom_models=Experiment.get('custom_models')
        #for idx,model in enumerate(custom_models):
        #    model_name = model._get_name()
        #    model.load_state_dict(torch.load(os.path.join(checkpoints_path,f'{use_checkpoint}',"{}_{}.pt".format(model_name,idx))))
        #all_files=otu.list_all_files(checkpoints_path)
        #print(all_files)
        start_epoch=use_checkpoint+1
    else:
        start_epoch=0   
        custom_models=Experiment.get('custom_models')
        if Experiment.get('model_init_fn'):
            for model in custom_models:
                model.apply(Experiment['model_init_fn'])
        if Experiment.get('init_models_weights'):
            Experiment['init_models_weights'](custom_models)


def load_experiment(experiment_dir,experiment_src_path):
    global Experiment
    experiment_home,experiment_name = os.path.split(experiment_dir.rstrip('/'))

    if experiment_src_path:
        try:
            print('### WARNNING ### Use designated src file, may be different from the one in {}'.format(experiment_dir))
            abs_path=os.path.abspath(experiment_src_path)
            module_dir,file_name=os.path.split(abs_path)
            sys.path.append(module_dir)
            module_name=file_name.split('.')[0]
            module = importlib.import_module(module_name)
            Experiment = module.Experiment
            #print('module file path',module.__file__)
        except Exception as e:
            print(e)
            print('Please Check experiment src file')
            exit(0)
    else:
        assert '#' in experiment_name
        try:
            experiment_src_name,create_time = experiment_name.split('#')
            sys.path.append(os.path.join(experiment_dir,'src'))
            module = importlib.import_module(experiment_src_name)
            Experiment = module.Experiment
        except Exception as e:
            print(e)
            print('Please Check experiment path')
            exit(0)

    parse_experiment_info(Experiment)
    Experiment['home']=experiment_dir
    Experiment['id']=experiment_name
    otu.init(Runtime,Experiment)

    #custom_models=Experiment.get('custom_models')
        
    models_path=os.path.join(experiment_dir,'models')
    print(models_path)
    load_models(models_path)
    
    try:
        models_path=os.path.join(experiment_dir,'models')
        #print(models_path)
        load_models(models_path)
        #for idx,model in enumerate(custom_models):
        #    model_name = model._get_name()
        #    model.load_state_dict(torch.load(os.path.join(models_path,"{}_{}.pt".format(model_name,idx))))
        return
    except Exception as e:
        print('Open models fail! ')
            
    try:
        print('Try using latest checkpoint')
        checkpoints_path=os.path.join(experiment_dir,'checkpoints')
        all_checkpoints = list(map(lambda x:int(x),os.listdir(checkpoints_path)))
        last_checkpont=max(all_checkpoints)
        load_models(os.path.join(checkpoints_path,f'{last_checkpoint}'))
        #for idx,model in enumerate(custom_models):
        #    model_name = model._get_name()
        #    model.load_state_dict(torch.load(os.path.join(checkpoints_path,f'{last_checkpoint}',"{}_{}.pt".format(model_name,idx))))
        return
    except Exception as e:
        print('Open checkpoint fail! ')

    print('Load experiment failed!')
    exit(0)

    
def create_experiment(args):
    global Runtime 
    global Experiment
    global experiment_id
    global experiment_home_dir
    global tb_writer
    experiment_src_path = args.src
    try:
        abs_path=os.path.abspath(experiment_src_path)
        module_dir,file_name=os.path.split(abs_path)
        sys.path.append(module_dir)
        module_name=file_name.split('.')[0]
        module = importlib.import_module(module_name)
        Experiment = module.Experiment
        #print('module file path',module.__file__)
    except Exception as e:
        print(e)
        print('Create Experiment Fail! Please Check experiment src file')
        exit(0)

    parse_experiment_info(Experiment)
    if args.hparam:
        update_hparams_by_conf('train',args.hparam)
    update_config_by_args(args)


    if not just_a_try:
        if Experiment.get('train_dataset_path') is None:
            print('Need dataset to train')
            exit(0)
        dataset_name=os.path.split(Experiment.get('train_dataset_path'))[1].split('.')[0]
        experiment_id='{}_{}'.format(module_name,dataset_name)
        if args.hparam:
            param_name = os.path.split(args.hparam)[1].split('.')[0]
            experiment_id=experiment_id+'_{}'.format(param_name)
        experiment_id = experiment_id + '#{}'.format(now)
        

        #print(Experiment.get('train_dataset_path'))
        #print('dataset_name',dataset_name)

        #experiment_id='{}_{}#{}'.format(module_name,dataset_name,now)
        experiment_home_dir='experiment_home/{}'.format(experiment_id)
        #print(experiment_home_dir)
        os.makedirs(experiment_home_dir,exist_ok=True)
        os.makedirs(os.path.join(experiment_home_dir,'models'),exist_ok=True)
        os.makedirs(os.path.join(experiment_home_dir,'checkpoints'),exist_ok=True)
        os.makedirs(os.path.join(experiment_home_dir,'src'),exist_ok=True)
        os.makedirs(os.path.join(experiment_home_dir,'results'),exist_ok=True)
        shutil.copy(module.__file__,os.path.join(experiment_home_dir,'src'))
        srcs=Experiment.get('codes_to_backup')
        #print('srcs',srcs)
        if srcs and len(srcs)>0:
            src_dir=os.path.split(module.__file__)[0]
            #print(src_dir)
            for f in srcs:
                src=os.path.join(src_dir,f)
                #print(src)
                if os.path.isdir(src):
                    shutil.copytree(src,os.path.join(experiment_home_dir,'src',f))
                else:
                    shutil.copy(src,os.path.join(experiment_home_dir,'src'))
        if Runtime['__rank']==0:
            tb_writer = SummaryWriter(os.path.join(experiment_home_dir,'tensorboard'))
        Experiment['home']=experiment_home_dir
        Experiment['id']=experiment_id
        Experiment['tensorboard']=tb_writer

    #update hparams by experiment hparams
    '''
    hparams=Experiment.get('hparams')
    if hparams:
        for k in hparams:
            HPARAMS[k]=hparams[k]
    '''

    otu.init(Runtime,Experiment)




def update_hparams_by_conf(action,filename):
    global Runtime
    global Experiment
    if filename is None:
        return
    with open(filename) as f:
        content = yaml.load(f)
        print(content)
        for a in ['train','val','infer']:
            params=content.get(a)
            if params is None:
                continue
            for k in params:
                if k in ['dataset_path']:
                    Experiment['_'.join([a,k])]=params[k]
                elif k in ['batch_size']:
                    Experiment['hparams'][a+k]=params[k]
                else:
                    Experiment['hparams'][k]=params[k]


def update_config_by_args(args):
    #global HPARAMS
    global Runtime
    global Experiment
    Runtime['action']=args.action
    for key in filter(lambda x:not x.startswith('_'),dir(args)):
        param = args.__getattribute__(key)
        if param:
            Experiment[key]=param
    gpus = os.getenv('CUDA_VISIBLE_DEVICES')
    if gpus:
        #print('Use environment GPUs {}'.format(gpus))
        pass
    else:
        gpus = args.gpus
        #print('Use GPUs {}'.format(gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    Runtime['gpus']=list(map(lambda x:int(x),gpus.split(',')))
    if args.action=='train' or args.action=='view':
        if args.dataset:
            paths = args.dataset.split(',')
            if len(paths)==1:
                Experiment['train_dataset_path']=paths[0]
                Experiment['val_dataset_path']=None
            elif len(paths)>1:
                Experiment['train_dataset_path']=paths[0]
                Experiment['val_dataset_path']=paths[1]
    elif args.action=='val':
        if args.dataset:
            paths = args.dataset.split(',')
            Experiment['val_dataset_path']=paths[0]

    use_cuda = not Experiment['no_cuda'] and torch.cuda.is_available()
    print('use gpus:{}'.format(gpus)) if use_cuda else print('use cpu')
    device = torch.device("cuda" if use_cuda else "cpu")
    Experiment['device']=device
    

def log_hparams():
    logger.info('HPARAMS:')
    logger.info(Experiment['hparams'])
    logger.info('CONFIGS:')
    config={}
    for k in EXPERIMENT_DEFAULT:
        config[k]=Experiment[k]
    logger.info(config)
    logger.info('RUNTIME:')
    logger.info(Runtime)
    if Runtime['__rank']==0 and tb_writer:
        print(Experiment['hparams'])
        tmp={}
        for k in Experiment['hparams']:
            v=Experiment['hparams'][k]
            if isinstance(v,tuple) or isinstance(v,list) or isinstance(v,dict)  or hasattr(v,'__call__') or v is None:
                tmp[k]=str(v)
            else:
                tmp[k]=v
        print(tmp)
        tb_writer.add_hparams(tmp,{})


def __save_models(custom_models,path):
    global Experiment
    if Experiment.get('store_model_dynamic_params') is not None:
        with open(os.path.join(path,"config.json"),'w') as f:
            data = Experiment.get('store_model_dynamic_params')()
            json.dump(data,f,indent=4)
    
    for idx,model in enumerate(custom_models):
        model_name = model._get_name()
        file_path = os.path.join(path,'{}_{}.pt'.format(model_name,idx))
        torch.save(model.state_dict(),file_path)


def save_models():
    if just_a_try:
        return
    if Runtime['__rank']!=0:
        return
    models_path=os.path.join(experiment_home_dir,'models')
    #print('Now saving models...')
    #if best_epoch>0:
    #    best_path = os.path.join(experiment_home_dir,'checkpoints','{}'.format(best_epoch))
    #    if os.path.exists(best_path):
    #        logger.info('Save best checkpoint epoch:{} loss:{} as final models'.format(best_epoch,best_loss))
    #        shutil.rmtree(models_path)
    #        shutil.copytree(best_path,models_path)
    #        if Experiment.get('post_save_models') is not None:
    #            Experiment.get('post_save_models')(models_path)
    #        return
    
    if Experiment.get('pre_save_models') is not None:
        Experiment.get('pre_save_models')(models_path)
    custom_models = Experiment.get('custom_models')
    __save_models(custom_models,models_path)
    if Experiment.get('post_save_models') is not None:
        Experiment.get('post_save_models')(models_path)
    logger.info('### Save {} models to path:{}'.format(len(custom_models),models_path))


def save_checkpoints(epoch):
    global Experiment
    global Runtime
    if just_a_try:
        return
    if Runtime['__rank']!=0:
        return
    #print('Now saving models...')
    custom_models = Experiment.get('custom_models')
    checkpoints_home_dir=os.path.join(experiment_home_dir,'checkpoints')
    checkpoints_path=os.path.join(checkpoints_home_dir,'{}'.format(epoch))
    os.makedirs(checkpoints_path,exist_ok=True)
    __save_models(custom_models,checkpoints_path)
    logger.info('### Epoch {} Save {} models to checkpoint:{}'.format(epoch,len(custom_models),checkpoints_path))

def check_distributed(args):
    global Runtime
    global ddp_flag
    if args.local_rank is not None:
        dist.init_process_group(backend=args.backend)
        print('init ddp',dist.get_rank())
        Runtime['__rank']=dist.get_rank()
        ddp_flag=True
    else:
        Runtime['__rank']=0
    Runtime['local_rank'] = args.local_rank if args.local_rank is not None else 0


if __name__=="__main__":
    parser = argparse.ArgumentParser(prog='global_behavior')
    parser.add_argument("-a","--action",choices=['train','val','infer','view','interact'],help="actions to take")
    parser.add_argument("-d",'--dataset',help="Dataset path(s) to load. for trainning, train_path,val_path separate by comma,val_path can be omitted")
    parser.add_argument('--no_cuda', action='store_true', help='disables CUDA training')
    parser.add_argument('--seed', type=int,  metavar='S',help='Random seed (default: 12345)')
    parser.add_argument('--local_rank',  type=int,help='node rank for distributed training')
    parser.add_argument('--gpus',default="0", help='GPU Ids to use, separated by comma')
    parser.add_argument("--backend",choices=['gloo','nccl'],default='nccl',help="commiunication lib toto take")
    parser.add_argument('--log_interval', type=int, help='how many batches to wait before logging training status')
    parser.add_argument('--log_file', help='logging file')
    parser.add_argument("--src",help="Experiment src module to train")
    parser.add_argument("--hparam",help="Experiment hparam file to load")
    parser.add_argument("-e","--experiment_path",help="Experiment path to load")
    parser.add_argument("--use_tensorboard",action='store_true',default=False,help="Enable tensorboard")
    parser.add_argument("-c","--create",action='store_true',default=False,help="Create new experiment logs")
    parser.add_argument("--use_checkpoint",type=int,default=-1,help="to use the Nth checkpoint in the experiment to initialize train weights")
    args = parser.parse_args()
    check_distributed(args)
    if args.action=='view':
        assert args.src is not None
        just_a_try = True
        #print('step 1')
        otu.load_vars_by_conf(args.hparam)
        create_experiment(args)
        #if args.hparam:
        #    update_hparams_by_conf('train',args.hparam)
        #update_config_by_args(args)
        #print('step 2')
        logger = otu.logger_init()
        init_experiment('view')
        #print('step 3')
        exit(0)
    elif args.action=='val':
        assert args.experiment_path is not None
        otu.load_vars_by_conf(args.hparam)
        load_experiment(args.experiment_path,args.src)
        update_hparams_by_conf('val',args.hparam)
        update_config_by_args(args)
        logger = otu.logger_init()
        validate_wrapper()
        exit(0)
    elif args.action=='infer':
        assert args.experiment_path is not None
        otu.load_vars_by_conf(args.hparam)
        load_experiment(args.experiment_path,args.src)
        update_hparams_by_conf('infer',args.hparam)
        update_config_by_args(args)
        logger = otu.logger_init()
        inference_wrapper()
        exit(0)
    elif args.action=='interact':
        assert args.experiment_path is not None
        otu.load_vars_by_conf(args.hparam)
        load_experiment(args.experiment_path,args.src)
        update_hparams_by_conf('infer',args.hparam)
        update_config_by_args(args)
        logger = otu.logger_init()
        interact_wrapper()
        exit(0)
    
    assert args.action =='train'
    assert args.src is not None
    just_a_try = not args.create

    
    otu.load_vars_by_conf(args.hparam)

    create_experiment(args)
    
    #if args.hparam:
    #    update_hparams_by_conf('train',args.hparam)
    #update_config_by_args(args)

    logger = otu.logger_init(Experiment['log_file'],os.path.join(experiment_home_dir,'log') if not just_a_try else None)
    

    #print('logger id:',logger.id)
    if just_a_try:
        logger.info('\n\n{}\n### WARNING: Run for debug, only 1/20 batches, without create experiment and helper functions will not be called###\n### Use -c to create experiment and save ###\n{}\n\n'.format('##'*70,'##'*70))
    else:
        logger.info(f'\n\n######################### Experiment Start at {now} ##################\n')
        logger.info('New experiment created in path:{}'.format(experiment_home_dir))

    log_hparams()

    run_experiment_callback("init_fn")
    
    load_checkpoints()
    train_wrapper()
    #save_models()  #save models by every improvement
    if Experiment.get('val_loader') and not just_a_try and Experiment['train_validate_final_with_best']:
        logger.info('Validate with best model')
        models_path=os.path.join(experiment_home_dir,'models')
        load_models(models_path)
        final_validate()
    Runtime['step']='finish'
    run_experiment_callback("exit_fn")
