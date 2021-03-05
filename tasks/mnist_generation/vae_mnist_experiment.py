import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, transforms
import one_torch_utils as otu

N_class=10
nWidth=28
nHeight=28
z_dim=2

# VAE model
class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=2):
        super(VAE, self).__init__()
        self.fc1 = torch.nn.Linear(image_size, h_dim)
        self.fc2 = torch.nn.Linear(h_dim, z_dim) 
        self.fc3 = torch.nn.Linear(h_dim, z_dim)
        self.fc4 = torch.nn.Linear(z_dim, h_dim)
        self.fc5 = torch.nn.Linear(h_dim, image_size)
        
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc4(z))
        return F.sigmoid(self.fc5(h))
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var

class VAE_LOSS(nn.Module):
    def __init__(self):
        super(VAE_LOSS, self).__init__()
    def forward(self,input,target,output):
        x_reconst, mu, log_var = output
        #print(x_reconst.shape,mu.shape,log_var.shape)
        
        reconst_loss = F.binary_cross_entropy(x_reconst, input, size_average=False)
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        loss = reconst_loss + kl_div
        return loss


def create_train_dataset_fn(path):
    return datasets.MNIST(path, train=True, transform=transforms.ToTensor())

def create_val_dataset_fn(path):
    return datasets.MNIST(path, train=False, transform=transforms.ToTensor())

def flatten_image_fn(runtime,experiment,original_data):
    x,y=original_data
    return  x.view(-1, nWidth*nHeight),y

def epoch_insight(result_list):
    print(result_list)

def loss_evaluation_fn(runtime,experiment):
    #loss = loss_criterion(output,target) 
    output = runtime.get('output')
    data = runtime.get('input')
    x_reconst, mu, log_var = output
    #print(x_reconst.shape,mu.shape,log_var.shape)
        
    reconst_loss = F.binary_cross_entropy(x_reconst, data, size_average=False)
    kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
    loss = reconst_loss + kl_div
    #print(reconst_loss.item(),kl_div.item())
    return loss,loss.item()


Experiment={
    # Define Experiment Model
    "init_fn":(otu.create_dirs_at_home,{'dirs':['images']}),
    "exit_fn":[(otu.convert_images_to_video,{'images_dir':'images','video_file':'evoluton.mp4'}),
               (otu.highlight_latest_result_file,{'input_dir':'images','filename':'final.jpg','ext':'.jpg'})],
    "hparams":{'optim':'Adam',
               'lr':2e-4,
               'loader_n_worker':4,
               'Adam':{'betas':(0.5,0.999)}
              },
    "custom_models":[VAE()],
    # Define function to create train dataset
    "create_train_dataset_fn":create_train_dataset_fn,
    # Define function to create validate dataset
    "create_val_dataset_fn":create_val_dataset_fn,    
    # Define callback function to collate dataset in dataloader, can be None
    "collate_fn_by_dataset":None,
    # Define callback function to preprocess data in each iteration, can be None
    "data_preprocess_fn":flatten_image_fn,
    # Define Loss function
    "loss_criterions":[],
    "loss_evaluation_fn":loss_evaluation_fn,
    # Define function to deep insight result in each iteration, can be None
    #"epoch_insight_fn":None, #epoch_insight#None #one_torch_utils.create_default_epoch_insight_fn(N_class)
    "post_batch_train_fn":(otu.batch_save_image,{'format_batch':lambda x:x['output'][0].view(-1,1,28,28)})
}

