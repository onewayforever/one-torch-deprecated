import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import one_torch_utils as otu
import one_torch_models
from torchvision import datasets, transforms

N_class=10

nChannel=3
nWidth=32
nHeight=32
nClass=10

class CustomCNNModel(nn.Module):
    def __init__(self):
        super(CustomCNNModel, self).__init__()
        self.nChannel = nChannel
        self.nClass = N_class 
        self.nWidth = nWidth 
        self.nHeight= nHeight 
        self.num_layers=2
        self.hidden_size=256
        self.dropout_rate=0.5
        self.resnet = one_torch_models.ResModelNd(self.nChannel,[self.nWidth,self.nHeight] , [('conv',32,3,1,1,'relu'),[('conv',32,3,1,1,'relu'),('conv',32,3,1,1,'relu')],[('conv',32,3,1,1,'relu')],('maxpool',2,2),('conv',64,3,1,1,'relu'),[('conv',64,3,1,1,'relu'),('conv',64,3,1,1,'relu')],[('conv',64,3,1,1,'relu')],('maxpool',2,2),('conv',64,3,1,1,'relu'),[('conv',64,3,1,1,'relu'),('conv',64,3,1,1,'relu')],('maxpool',2,2),('conv',64,3,1,1,'relu'),[('conv',64,3,1,1,'relu'),('conv',64,3,1,1,'relu')],[('conv',64,3,1,1,'relu')],('maxpool',2,2)] )
        outputshape=self.resnet.cnn_output_shape
        self.flatten_dim = np.product(outputshape)
        self.dropout = nn.Dropout(self.dropout_rate,inplace=True)
        self.fc = nn.Linear(self.flatten_dim, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.nClass)
    def forward(self,input):
        x = input
        out=self.resnet(x)
        out = out.view(-1, self.flatten_dim)
        out = torch.relu(self.fc(self.dropout(out)))
        out = F.log_softmax(self.fc2(out),dim=1)
        return out


transform = transforms.Compose(
            [transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 

def create_train_dataset_fn(path):
    if N_class==10:
        return datasets.CIFAR10(path, train=True, transform=transform,download=True)
    if N_class==100:
        return datasets.CIFAR100(path, train=True, transform=transform,download=True)

def create_val_dataset_fn(path):
    if N_class==10:
        return datasets.CIFAR10(path, train=False, transform=transform)
    if N_class==100:
        return datasets.CIFAR100(path, train=False, transform=transform)

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
    #for i in range(10):
    for i in range(len(care)):
        if care[i] == True:
            #otu.save_results_to_image(runtime,experiment,'errors',[{'numpy':input[i].cpu().permute(1,2,0).numpy(),'label':'{}->{}'.format(pred[i].item(),target[i].item())}])
            otu.save_results_to_image(runtime,experiment,'errors',[{'numpy':input[i].cpu().permute(1,2,0).numpy()*256,'label':'{}->{}'.format(pred[i].item(),target[i].item())}])
    return ret

Experiment={
    # Define Experiment Model
    "custom_models":[CustomCNNModel()],
    # Define function to create train dataset
    "create_train_dataset_fn":create_train_dataset_fn,
    # Define function to create validate dataset
    "create_val_dataset_fn":create_val_dataset_fn,    
    # Define callback function to collate dataset in dataloader, can be None
    "collate_fn_by_dataset":None,
    # Define callback function to preprocess data in each iteration, can be None
    "data_preprocess_fn":None,
    # Define Loss function
    "loss_criterions":[nn.NLLLoss()],
    # Define function to deep insight result in each iteration, can be None
    "post_epoch_val_fn":(otu.epoch_insight_classification,{'nclass':N_class}),
    "post_batch_val_fn":otu.batch_result_extract,
    "post_epoch_train_fn":(otu.epoch_insight_classification,{'nclass':N_class}),
    "post_batch_train_fn":otu.batch_result_extract,
    "validate_batch_val_result_fn":validate_batch_val_result_fn
}


