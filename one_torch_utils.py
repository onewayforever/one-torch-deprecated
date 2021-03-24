import torch
import numpy as np
import logging
import os
from torchvision.utils import save_image
import cv2
import shutil
import matplotlib.pyplot as plt
import random
import csv
import cmd

Runtime=None
Experiment=None
logger=None

def init(runtime,experiment):
    global Experiment
    global Runtime 
    Runtime = runtime
    Experiment = experiment

def epoch_insight_2class(result_list):
    threshold = 0.5
    pred_list=list(map(lambda x:x[0],result_list))
    target_list=list(map(lambda x:x[1],result_list))
    pred=torch.cat(pred_list,dim=0).squeeze(1)
    pred = (pred >= threshold).long()
    target=torch.cat(target_list,dim=0)
    target = (target >= threshold).long()
    #print('',pred.shape,target.shape)
    TP = ((pred== 1) & (target.data == 1)).sum().float().item()
    # TN    predict 和 label 同时为0
    TN = ((pred== 0) & (target.data == 0)).sum().float().item()
    # FN    predict 0 label 1
    FN = ((pred== 0) & (target.data == 1)).sum().float().item()
    # FP    predict 1 label 0
    FP = ((pred== 1) & (target.data == 0)).sum().float().item()
    #print(pred,target)
    #print(TP,TN,FN,FP)
    
    p = TP / (TP + FP + 1e-04)
    r = TP / (TP + FN + 1e-04)
    F1 = 2 * r * p / (r + p + 1e-04)
    acc = (TP + TN) / (TP + TN + FP + FN)
    return 'TP:{},TN:{},FN:{},FP:{},ACC:{:.3f},Precise:{:.3f},Recall:{:.3f},F1:{:.3f}'.format(TP,TN,FN,FP,acc,p,r,F1)

def make_confusion_matrix(pred,target,nclass):
    confusion_matrix=np.zeros((nclass,nclass))
    for p,t in zip(list(pred),list(target)):
        confusion_matrix[p, t] += 1
    TP = np.diag(confusion_matrix)
    statistics=[]
    #print(confusion_matrix)
    for c in range(nclass):
        #idx = np.ones(nclass).astype(int)
        #idx[c] = 0
        idx = list(range(nclass))
        FP = confusion_matrix[c, idx].sum() - TP[c]
        FN = confusion_matrix[idx, c].sum() - TP[c]
        TN = confusion_matrix.sum()-FP-FN-TP[c]
        statistics.append({'TP':int(TP[c]),'TN':int(TN),'FN':int(FN),'FP':int(FP),'Precise':TP[c]/(TP[c]+FP+1e-04),'Recall':TP[c]/(TP[c]+FN+1e-04)})

    return confusion_matrix,statistics

def epoch_insight_Nclass(N_class,result_list):
    if len(result_list)==0:
        return None
    pred_list=list(map(lambda x:x['output'],result_list))
    target_list=list(map(lambda x:x['target'],result_list))
    pred=torch.cat(pred_list,dim=0)
    target=torch.cat(target_list,dim=0)
    pred = torch.argmax(pred, dim=1)
    #print('',pred.shape,target.shape)
    #print(pred,target)
    matrix,statistics = make_confusion_matrix(pred,target,N_class)
    return matrix, statistics
    #print(matrix)
    TP = np.diag(matrix)
    acc = TP.sum()/matrix.sum()
    def view_statistics(info):
        return 'ACC:{:.3f},TP:{},TN:{},FN:{},FP:{},Precise:{:.3f},Recall:{:.3f} sum:{}\n{}'.format(acc,info['TP'],info['TN'],info['FN'],info['FP'],info['Precise'],info['Recall'],matrix.sum(),matrix.astype(int))
    return view_statistics(statistics[N_class-1]) 

def create_default_epoch_insight_fn(N_class):  
    def fn(runtime):
    #def fn(epoch,result_list,is_train):
        matrix,statistics = epoch_insight_Nclass(N_class,runtime.get('epoch_results'))
        TP = np.diag(matrix)
        acc = TP.sum()/matrix.sum()
        def view_statistics(info):
            return 'ACC:{:.3f},TP:{},TN:{},FN:{},FP:{},Precise:{:.3f},Recall:{:.3f} sum:{}\n{}'.format(acc,info['TP'],info['TN'],info['FN'],info['FP'],info['Precise'],info['Recall'],matrix.sum(),matrix.astype(int))
        return view_statistics(statistics[N_class-1]) 
    return fn

def epoch_insight_classification(runtime,experiment,ret,nclass=1):
    assert nclass > 1
    if runtime.get('epoch_results') is None:
        return ret 
    matrix,statistics = epoch_insight_Nclass(nclass,runtime.get('epoch_results'))
    TP = np.diag(matrix)
    acc = TP.sum()/matrix.sum()
    def view_statistics(info):
        return 'ACC:{:.3f},TP:{},TN:{},FN:{},FP:{},Precise:{:.3f},Recall:{:.3f} sum:{}\n{}'.format(acc,info['TP'],info['TN'],info['FN'],info['FP'],info['Precise'],info['Recall'],matrix.sum(),matrix.astype(int))
    ret['display']=view_statistics(statistics[nclass-1]) 
    return ret

def batch_result_extract(runtime,experiment,ret):
    #print(runtime['output'])
    #print(runtime['target'])
    return { 'output':runtime['output'].clone().detach().cpu(),
             'target':runtime['target'].clone().detach().cpu()
            }

class MyLogger():
    def __init__(self,log_file=None,sub_log=None):
        self.id = -1
        self.logger = logging.getLogger()
        self.logger.setLevel('INFO')
        chlr = logging.StreamHandler() 
        self.logger.addHandler(chlr)
        if log_file:
            fhlr = logging.FileHandler(log_file) 
            self.logger.addHandler(fhlr)
        if sub_log:
            sfhlr = logging.FileHandler(sub_log)
            self.logger.addHandler(sfhlr)
    def info(self,record):
        if self.id==0:
            self.logger.info(record)
 
    

def logger_init(log_file=None,sub_log=None):
    global logger
    logger = MyLogger(log_file,sub_log)
    rank = Runtime['__rank']
    if rank<0:
        logger.id = 0
    else:
        logger.id = rank
    return logger
    '''
    logger = logging.getLogger()
    logger.setLevel('INFO')
    chlr = logging.StreamHandler() 
    logger.addHandler(chlr)
    if log_file:
        fhlr = logging.FileHandler(log_file) 
        logger.addHandler(fhlr)
    if sub_log:
        sfhlr = logging.FileHandler(sub_log)
        logger.addHandler(sfhlr)
    return logger
    '''

def save_results_to_csv(runtime,experiment,tag,results):
    results_csv=runtime['results']['csv']
    if results_csv.get(tag) is None:
        runtime_id=runtime.get('runtime_id')
        experiment_home=Experiment.get('home')
        os.makedirs(os.path.join(experiment_home,'results',runtime_id),exist_ok=True) 
        path=os.path.join(get_home_path('results'),runtime_id,'{}.csv'.format(tag))
        f = open(path,'a',encoding='utf-8')
        csv_writer = csv.writer(f)
        runtime['results']['csv'][tag]={'writer':csv_writer,'path':path}
    else:
        csv_writer=results_csv[tag]['writer']

    for row in results:
        csv_writer.writerow(row)

result_id=0
def save_one_img_result(path,item):
    global result_id
    result_id+=1
    #print(item['numpy'])
    label=item.get('label')
    #print(item['numpy'].shape)
    cv2.imwrite(os.path.join(path,'{}_{}.jpg'.format(result_id,label)),item['numpy'])

def get_results_path(runtime,file_type,tag):
    results_type=runtime['results'][file_type]
    if results_type.get(tag) is None:
        runtime_id=runtime.get('runtime_id')
        experiment_home=Experiment.get('home')
        os.makedirs(os.path.join(experiment_home,'results',runtime_id),exist_ok=True) 
        os.makedirs(os.path.join(experiment_home,'results',runtime_id,tag),exist_ok=True) 
        os.makedirs(os.path.join(experiment_home,'results',runtime_id,tag,file_type),exist_ok=True) 
        dir_path=os.path.join(experiment_home,'results',runtime_id,tag,file_type)
        runtime['results'][file_type][tag]={'path':dir_path}
    else:
        dir_path=results_type[tag]['path']
    return dir_path
    
def save_results_to_image(runtime,experiment,tag,results):
    '''
    results_img=runtime['results']['img']
    if results_img.get(tag) is None:
        runtime_id=runtime.get('runtime_id')
        experiment_home=Experiment.get('home')
        os.makedirs(os.path.join(experiment_home,'results',runtime_id),exist_ok=True) 
        os.makedirs(os.path.join(experiment_home,'results',runtime_id,tag),exist_ok=True) 
        os.makedirs(os.path.join(experiment_home,'results',runtime_id,tag,'img'),exist_ok=True) 
        img_dir_path=os.path.join(experiment_home,'results',runtime_id,tag,'img')
        runtime['results']['img'][tag]={'path':img_dir_path}
    else:
        img_dir_path=results_img[tag]['path']
    '''
    img_dir_path=get_results_path(runtime,'img',tag)

    for item in results:
        save_one_img_result(img_dir_path,item)

def create_dirs_at_home(runtime,experiment,ret,dirs):
    experiment_home=Experiment.get('home')
    for d in dirs:
        os.makedirs(os.path.join(experiment_home,d),exist_ok=True)
    return ret

def get_home_path(dir_name):
    experiment_home=Experiment.get('home')
    return os.path.join(experiment_home,dir_name)

def batch_save_image(runtime,experiment,ret,dir='images',combine=(5,5),format_batch=None,interval=0,start=0):
    output = runtime.get('output')
    #epoch = runtime.get('epoch')
    #total = runtime.get('total_train_batches')
    #batch_idx = runtime.get('batch_idx')
    #batches_done = epoch * total + batch_idx
    batches_done = runtime['batches_done']
    if batches_done<start:
        return ret
    if interval>0 and batches_done % interval != 0:
        return ret
    if runtime['step']=='train':
        filename = os.path.join(get_home_path(dir),'{:0>6d}.jpg'.format(batches_done))
    else:
        filename = os.path.join(get_results_path(runtime,'img',dir),'{:0>6d}.jpg'.format(batches_done))
    r,w=combine
    if format_batch:
        img_data = format_batch(runtime)
    else:
        img_data = output
    #print('minibatch',img_data.shape,img_data[:(r*w)].shape)
    if len(img_data.shape)==4: #combine a group of images to one
        save_image(img_data[:(r*w)], filename, nrow=r, normalize=True)
    if len(img_data.shape)==3: #save the only one image 
        save_image(img_data, filename, nrow=r, normalize=True)
    return ret


def list_all_files(base,ext=None):
    file_list=[]
    for root, ds, fs in os.walk(base):
        for f in fs:
            if ext is None:
                file_list.append(os.path.join(root, f))
            elif f.endswith(ext):
                file_list.append(os.path.join(root, f))
    return list(sorted(file_list))


def highlight_latest_result_file(runtime,experiment,ret,input_dir,output_dir='.',filename='final.log',ext=None):
    file_list=list_all_files(get_home_path(input_dir),ext)
    if len(file_list)==0:
        return ret
    latest_file=sorted(file_list,key=lambda x:os.stat(x).st_mtime,reverse=True)[0]
    target_path=os.path.join(get_home_path(output_dir),filename)
    shutil.copy(latest_file,target_path)
    logger.info('Latest result is at {}'.format(target_path))
    return ret

def convert_images_to_video(runtime,experiment,ret,images_dir='images',output_dir='.',video_file='clip.mp4'):
    #print(get_home_path(images_dir))
    images_file_list=list_all_files(get_home_path(images_dir),ext='.jpg')
    #print(images_file_list)
    total =  len(images_file_list)
    if total == 0:
        return ret
    logger.info('Starting convert {} images to video..'.format(total))
    img = cv2.imread(images_file_list[0])
    #print(dir(img))
    #print(img.shape)
    size=(img.shape[1],img.shape[0])
    #print(size)
    video_path='.clip.mp4'
    #print(video_path)
    #fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #fourcc = cv2.VideoWriter_fourcc(*'I420')
    #fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    #fourcc = cv2.VideoWriter_fourcc('A', 'V', 'C', '1')
    #fourcc = cv2.VideoWriter_fourcc('F', 'L', 'V', '1')
    videowriter = cv2.VideoWriter(video_path,fourcc,20,size)
    #videowriter = cv2.VideoWriter(video_path,fourcc,20,size)
    for filename in images_file_list:
        #print(filename)
        img = cv2.imread(filename)
        if img is None:
            #print(filename + " is error!")
            continue
        #print('write img {}'.format(filename))
        for i in range(2):
            videowriter.write(img)
    videowriter.release()
    target_path = os.path.join(get_home_path(output_dir),video_file)
    shutil.move(video_path,target_path)
    logger.info('Convert images to video at file: {}'.format(target_path))
    return ret


class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)

def preview_tensor(data):
    shape_list=[]
    if isinstance(data,tuple) or isinstance(data,list):
        for item in data:
            preview_tensor(item)
    else:
        print(data)
        if hasattr(data,'shape'):
            print(data.shape)

class CLI(cmd.Cmd):
    def __init__(self,prompt):
        cmd.Cmd.__init__(self)
        self.prompt = prompt  # define command prompt

    def do_quit(self, arg):
        return True

    def help_quit(self):
        print("syntax: quit -- terminatesthe application")

    # define the shortcuts
    do_q = do_quit

