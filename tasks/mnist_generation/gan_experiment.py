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


latent_dim=100
nChannel=1
nWidth=28
nHeight=28
img_shape = (nChannel, nWidth, nHeight)

def create_train_dataset_fn(path):
    return datasets.MNIST(path, train=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ]))

def create_val_dataset_fn(path):
    return datasets.MNIST(path, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
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

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
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

    real_imgs = data
    generator,discriminator = models
    optimizer_G,optimizer_D = optimizers
    adversarial_loss = criterions[0]

    # Adversarial ground truths
    #valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
    #fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
    valid = torch.ones(real_imgs.shape[0],1,device=device)
    fake = torch.zeros(real_imgs.shape[0],1,device=device)


    # -----------------
    #  Train Generator
    # -----------------

    optimizer_G.zero_grad()

    # Sample noise as generator input
    #z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
    z = torch.randn(real_imgs.shape[0], latent_dim,device=device)

    # Generate a batch of images
    gen_imgs = generator(z)

    # Loss measures generator's ability to fool the discriminator
    g_loss = adversarial_loss(discriminator(gen_imgs), valid)

    g_loss.backward()
    optimizer_G.step()

    # ---------------------
    #  Train Discriminator
    # ---------------------

    optimizer_D.zero_grad()

    # Measure discriminator's ability to classify real from generated samples
    real_loss = adversarial_loss(discriminator(real_imgs), valid)
    fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
    d_loss = (real_loss + fake_loss) / 2

    d_loss.backward()
    optimizer_D.step()
    return gen_imgs, {'g_loss':g_loss.item(),'d_loss':d_loss.item(),'loss':(g_loss.item()+d_loss.item())/2} 



Experiment={
    "init_fn":(otu.create_dirs_at_home,{'dirs':['images']}),
    "exit_fn":[(otu.convert_images_to_video,{'images_dir':'images','video_file':'evoluton.mp4'}),
               (otu.highlight_latest_result_file,{'input_dir':'images','filename':'final.jpg','ext':'.jpg'})],
    "hparams":{'optim':'Adam',
               'lr':2e-4,
               'loader_n_worker':4,
               'Adam':{'betas':(0.5,0.999)}
              },
    # Define Experiment Model
    "custom_models":[generator,discriminator],
    # Define function to create train dataset
    "create_train_dataset_fn":create_train_dataset_fn,
    # Define function to create validate dataset
    "create_val_dataset_fn":create_val_dataset_fn,    
    # Define callback function to collate dataset in dataloader, can be None
    "collate_fn_by_dataset":None,
    # Define callback function to preprocess data in each iteration, can be None
    "data_preprocess_fn":None,
    # Define Loss function
    "loss_criterions":[adversarial_loss],
    "loss_evaluation_fn":None,
    # Define function to deep insight result in each iteration, can be None
    "custom_train_fn":gan_train_fn,
    "post_batch_train_fn":(otu.batch_save_image,{'interval':600})
}
