import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import one_torch_utils as otu
from torchvision import datasets, transforms
import os
import cv2


N_class=10

DATASET='MNIST'

latent_dim=100
nChannel=1
nWidth=28
nHeight=28

HPARAMS={
    'optim':'Adam',
    'lr':2e-4,
    'batch_size':64,
    'n_epochs':200,
    'loader_n_worker':8,
    'Adam':{'betas':(0.5,0.999)}#,'weight_decay':0.01}
    }
soft_label=0
flip_label=0
n_critic=0

otu.update_vars_by_conf(globals())

img_shape = (nChannel, nWidth, nHeight)

def create_train_dataset_fn(path):
    return getattr(datasets,DATASET)(path, train=True, transform=transforms.Compose([
                           transforms.Resize(nWidth),
                           transforms.ToTensor(),
                           transforms.Normalize([0.5], [0.5])
                       ]))



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        block_list=[nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()]

        self.model = nn.Sequential(*block_list)

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        #print('img_flat',img_flat.shape)
        validity = self.model(img_flat)

        return validity

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()



def gan_train_fn(runtime,experiment):
#def gan_train_fn(models,criterions,optimizers,data,target,device):
    models = experiment['custom_models']
    criterions = experiment['loss_criterions']
    optimizers = experiment['custom_optimizers']
    device = experiment['device']
    data = runtime['input']
    assert len(models)==2 and len(optimizers)==2
    g_loss = None

    real_imgs = data
    generator,discriminator = models
    optimizer_G,optimizer_D = optimizers
    adversarial_loss = criterions[0]
   
    batch_size = real_imgs.shape[0]

    # Adversarial ground truths
    valid = torch.ones(batch_size,1,device=device,requires_grad=False)
    fake = torch.zeros(batch_size,1,device=device,requires_grad=False)
    if soft_label>0:
        valid-=soft_label*torch.rand(batch_size,1).to(device)
        fake+=soft_label*torch.rand(batch_size,1).to(device)
    z = torch.randn(batch_size, latent_dim,device=device)

    # Generate a batch of images
    gen_imgs = generator(z)

    # -----------------
    #  Train Generator
    # -----------------
    if n_critic==0 or runtime['batch_idx']%n_critic==0:
        optimizer_G.zero_grad()

	    # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

    # ---------------------
    #  Train Discriminator
    # ---------------------

    optimizer_D.zero_grad()

    # Measure discriminator's ability to classify real from generated samples
    if flip_label>0:
        flip_num=int(flip_label*batch_size)
        select = torch.randperm(batch_size)[:flip_num]
        valid[select]=0
        select = torch.randperm(batch_size)[:flip_num]
        fake[select]=1
    real_loss = adversarial_loss(discriminator(real_imgs), valid)
    fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
    d_loss = (real_loss + fake_loss) / 2

    d_loss.backward()
    optimizer_D.step()


    if g_loss is None:
        return gen_imgs, {'d_loss':d_loss.item()} 
    else:
        return gen_imgs, {'g_loss':g_loss.item(),'d_loss':d_loss.item(),'loss':0.01*g_loss.item()+d_loss.item()} 



Experiment={
    "init_fn":(otu.create_dirs_at_home,{'dirs':['images']}),
    "exit_fn":[(otu.convert_images_to_video,{'images_dir':'images','video_file':'evoluton.mp4'}),
               (otu.highlight_latest_result_file,{'input_dir':'images','filename':'final.jpg','ext':'.jpg'})],
    "hparams":HPARAMS,
    # Define Experiment Model
    "custom_models":[generator,discriminator],
    # Define function to create train dataset
    "create_train_dataset_fn":create_train_dataset_fn,
    "loss_criterions":[adversarial_loss],
    "loss_evaluation_fn":None,
    # Define function to deep insight result in each iteration, can be None
    "custom_train_fn":gan_train_fn,
    #"learning_curve_batch_interval":400,
    "post_batch_train_fn":(otu.batch_save_image,{'interval':600})
}
