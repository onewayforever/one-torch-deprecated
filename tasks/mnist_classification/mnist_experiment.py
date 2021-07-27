import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import one_torch_utils as otu
from torchvision import datasets, transforms

N_class=10

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def create_train_dataset_fn(path):
    return datasets.MNIST(path, download=False,train=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                          #transforms.Normalize((0.1307,), (0.3081,))
                       ]))

def create_val_dataset_fn(path):
    return datasets.MNIST(path, download=False,train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           #transforms.Normalize((0.1307,), (0.3081,))
                       ]))

loss_criterion=nn.NLLLoss()

def loss_evaluation_fn(runtime,experiment):
    output = runtime.get('output')
    target = runtime.get('target')
    loss = loss_criterion(output,target) 
    return loss,loss.item()

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
            otu.save_results_to_image(runtime,experiment,'errors',[{'numpy':input[i].cpu().permute(1,2,0).numpy()*256,'label':'{}->{}'.format(pred[i].item(),target[i].item())}])
    return ret

Experiment={
    "hparams":{'optim':'Adam',
               'lr':2e-4,
               'loader_n_worker':4,
               'Adam':{'betas':(0.5,0.999)},
               'early_stop':5,
              },
    # Define Experiment Model
    "custom_models":[Net()],
    # Define function to create train dataset
    "create_train_dataset_fn":create_train_dataset_fn,
    # Define function to create validate dataset
    "create_val_dataset_fn":create_val_dataset_fn,    
    # Define callback function to collate dataset in dataloader, can be None
    "collate_fn_by_dataset":None,
    # Define callback function to preprocess data in each iteration, can be None
    "data_preprocess_fn":None,
    # Define Loss function
    "loss_criterions":[loss_criterion],
    "loss_evaluation_fn":loss_evaluation_fn,
    # Define function to deep insight result in each iteration, can be None
    "post_epoch_val_fn":[(otu.epoch_insight_classification,{'nclass':N_class}),(otu.to_plot,{'keys':['ACC']})],
    "post_batch_val_fn":otu.batch_result_extract,
    "post_epoch_train_fn":[(otu.epoch_insight_classification,{'nclass':N_class}),(otu.to_plot,{'keys':['ACC']})],
    "post_batch_train_fn":otu.batch_result_extract,
    "validate_batch_val_result_fn":validate_batch_val_result_fn,
    "validate_n_batch_in_train":0,
    "train_validate_each_n_epoch":1,
    "train_validate_final_with_best":True
}
