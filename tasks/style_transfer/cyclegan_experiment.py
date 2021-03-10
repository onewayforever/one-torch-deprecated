import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import one_torch_utils as otu
from torchvision import datasets, transforms
import os
import cv2
from PIL import Image
import itertools
import glob
import random
from torchvision.utils import save_image, make_grid



HPARAMS={}
#IMAGE_SIZE = 784
#HPARAMS['LATENT_DIM']=100

HPARAMS['RES_BLOCK']=9



nWidth=64
nHeight=64
nChannel=3
img_shape = (nChannel, nWidth, nHeight)

BATCH_SIZE=56
NROW=5
def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image
class ImageDataset2Sets(Dataset):
    def __init__(self, root, transforms_=None, unaligned=True, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, "%sA" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%sB" % mode) + "/*.*"))
        #print(self.files_A)
        #print(self.files_B)

    def __getitem__(self, index):
        #print(index)
        #print(self.files_A[index % len(self.files_A)])
        image_A = Image.open(self.files_A[index % len(self.files_A)])
        #print(image_A)

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])
        #print(image_B)
        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        item ={"A": item_A, "B": item_B}
        #print(item)
        return item

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

transforms_ = [
    transforms.Resize(int(nHeight * 1.12), Image.BICUBIC),
    #transforms.Resize((nWidth,nHeight), Image.BICUBIC),
    transforms.RandomCrop((nHeight, nWidth)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

#DATASET_NAME="horse2zebra"
#DATASET_NAME="apple2orange"
#DATASET_NAME="vangogh2photo"
def create_train_dataset_fn(path):
    #print(os.path.join(path,DATASET_NAME))
    #return ImageDataset2Sets(os.path.join(path,DATASET_NAME),transforms_=transforms_,mode="train")
    return ImageDataset2Sets(path,transforms_=transforms_,mode="train")

def create_val_dataset_fn(path):
    return ImageDataset2Sets(path,transforms_=transforms_,mode="test")
    #return ImageDataset2Sets(os.path.join(path,DATASET_NAME),transforms_=transforms_,mode="test")

G_NAME='Resnet'

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, param=9):
        super(GeneratorResNet, self).__init__()
        num_residual_blocks=param

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

MyGenerator=GeneratorResNet

##############################
#        Discriminator
##############################
D_NAME='Single_1'

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)
MyDiscriminator = Discriminator


criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

All_Criterions={
    'criterion_GAN':criterion_GAN,
    'criterion_cycle':criterion_cycle,
    'criterion_identity':criterion_identity
}

input_shape = (nChannel, nHeight, nWidth)

G_AB = MyGenerator(input_shape, param=HPARAMS['RES_BLOCK'])
G_BA = MyGenerator(input_shape, param=HPARAMS['RES_BLOCK'])
D_A = MyDiscriminator(input_shape)
D_B = MyDiscriminator(input_shape)


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

criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()


def data_preprocess_fn(runtime,experiment,data):
    return (data['A'],data['B']),None


fake_A_buffer = otu.ReplayBuffer()
fake_B_buffer = otu.ReplayBuffer()

def cyclegan_train_fn(runtime,experiment):
#def cyclegan_train_fn(models,criterions,optimizers,data,target,device):
    models = experiment['custom_models']
    criterions = experiment['loss_criterions']
    optimizers = experiment['custom_optimizers']
    device = experiment['device']
    data = runtime['input']
    assert len(models)==4 and len(optimizers)==3

    real_A , real_B = data
    #print('data',real_A.shape,real_B.shape)
    G_AB,G_BA,D_A,D_B= models
    criterion_GAN,criterion_cycle,criterion_identity = criterions
    optimizer_G,optimizer_D_A,optimizer_D_B = optimizers


    # Adversarial ground truths
    valid = torch.autograd.Variable(torch.from_numpy(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False).float().to(device)
    fake = torch.autograd.Variable(torch.from_numpy(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False).float().to(device)


    optimizer_G.zero_grad()

    # Identity loss
    loss_id_A = criterion_identity(G_BA(real_A), real_A)
    loss_id_B = criterion_identity(G_AB(real_B), real_B)

    loss_identity = (loss_id_A + loss_id_B) / 2

    # GAN loss
    fake_B = G_AB(real_A)
    loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
    fake_A = G_BA(real_B)
    loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

    loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

    # Cycle loss
    recov_A = G_BA(fake_B)
    loss_cycle_A = criterion_cycle(recov_A, real_A)
    recov_B = G_AB(fake_A)
    loss_cycle_B = criterion_cycle(recov_B, real_B)

    loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

    # Total loss
    lambda_cyc=10.0
    lambda_id=5.0
    loss_G = loss_GAN + lambda_cyc * loss_cycle + lambda_id * loss_identity

    loss_G.backward()
    optimizer_G.step()

    # -----------------------
    #  Train Discriminator A
    # -----------------------

    optimizer_D_A.zero_grad()

    # Real loss
    loss_real = criterion_GAN(D_A(real_A), valid)
    # Fake loss (on batch of previously generated samples)
    #fake_A_ = fake_A_buffer.push_and_pop(fake_A)
    #fake_A_ = fake_A
    loss_fake = criterion_GAN(D_A(fake_A.detach()), fake)
    # Total loss
    loss_D_A = (loss_real + loss_fake) / 2

    loss_D_A.backward()
    optimizer_D_A.step()

    # -----------------------
    #  Train Discriminator B
    # -----------------------

    optimizer_D_B.zero_grad()

    # Real loss
    loss_real = criterion_GAN(D_B(real_B), valid)
    # Fake loss (on batch of previously generated samples)
    #fake_B_ = fake_B_buffer.push_and_pop(fake_B)
    #fake_B_ = fake_B
    loss_fake = criterion_GAN(D_B(fake_B.detach()), fake)
    # Total loss
    loss_D_B = (loss_real + loss_fake) / 2

    loss_D_B.backward()
    optimizer_D_B.step()

    loss_D = (loss_D_A + loss_D_B) / 2
    loss_D = (loss_D_A + loss_D_B) / 2
    return (fake_B,fake_A),{'loss_D':loss_D.item(),
            'loss_G':loss_G.item(),
            'loss_GAN':loss_GAN.item(),
            'loss_cycle':loss_cycle.item(),
            'loss_identity':loss_identity.item()}


def cyclegan_val_fn(runtime,experiment):
    device = experiment['device']
    data = runtime['input']
    models = experiment['custom_models']
    criterions = experiment['loss_criterions']
    real_A , real_B = data
    #print('data',real_A.shape,real_B.shape)
    G_AB,G_BA,D_A,D_B= models
    criterion_GAN,criterion_cycle,criterion_identity = criterions
    valid = torch.autograd.Variable(torch.from_numpy(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False).float().to(device)
    fake = torch.autograd.Variable(torch.from_numpy(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False).float().to(device)

    fake_B = G_AB(real_A)
    fake_A = G_BA(real_B)
    
    loss_id_A = criterion_identity(G_BA(real_A), real_A)
    loss_id_B = criterion_identity(G_AB(real_B), real_B)

    loss_identity = (loss_id_A + loss_id_B) / 2
    
    loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
    loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
    loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
    recov_A = G_BA(fake_B)
    loss_cycle_A = criterion_cycle(recov_A, real_A)
    recov_B = G_AB(fake_A)
    loss_cycle_B = criterion_cycle(recov_B, real_B)

    loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
    lambda_cyc=10.0
    lambda_id=5.0
    loss_G = loss_GAN + lambda_cyc * loss_cycle + lambda_id * loss_identity
    loss_real = criterion_GAN(D_A(real_A), valid)
    # Fake loss (on batch of previously generated samples)
    fake_A_ = torch.autograd.Variable(fake_A_buffer.push_and_pop(fake_A))
    #fake_A_ = fake_A
    loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
    # Total loss
    loss_D_A = (loss_real + loss_fake) / 2
    loss_real = criterion_GAN(D_B(real_B), valid)
    # Fake loss (on batch of previously generated samples)
    fake_B_ = torch.autograd.Variable(fake_B_buffer.push_and_pop(fake_B))
    #fake_B_ = fake_B
    loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
    # Total loss
    loss_D_B = (loss_real + loss_fake) / 2
    loss_D = (loss_D_A + loss_D_B) / 2
    loss_D = (loss_D_A + loss_D_B) / 2
    # Arange images along x-axis
    #real_A = make_grid(real_A, nrow=5, normalize=True)
    #real_B = make_grid(real_B, nrow=5, normalize=True)
    #fake_A = make_grid(fake_A, nrow=5, normalize=True)
    #fake_B = make_grid(fake_B, nrow=5, normalize=True)
    # Arange images along y-axis
    #image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    #save_image(image_grid, "images/%s/%s.png" % (opt.dataset_name, batches_done), normalize=False)
    #return (fake_B,fake_A),{}
    return (fake_B,fake_A),{'loss_D':loss_D.item(),
            'loss_G':loss_G.item(),
            'loss_GAN':loss_GAN.item(),
            'loss_cycle':loss_cycle.item(),
            'loss_identity':loss_identity.item()}

def epoch_val_fn(runtime,experiment,ret):
    val_loader = experiment['val_loader'] 
    assert val_loader is not None
    data,_ = data_preprocess_fn(runtime,experiment,next(iter(val_loader)))
    models = experiment['custom_models']
    criterions = experiment['loss_criterions']
    device = experiment['device']
    real_A , real_B = data
    real_A=real_A.to(device)
    real_B=real_B.to(device)
    runtime['input']=(real_A,real_B)
    
    G_AB,G_BA,D_A,D_B= models
    criterion_GAN,criterion_cycle,criterion_identity = criterions
    fake_B = G_AB(real_A)
    fake_A = G_BA(real_B)
    runtime['output']=(fake_B,fake_A)
    return ret
  
            
def combine_2sets(runtime):
    
    real_A, real_B = runtime['input']
    fake_B, fake_A = runtime['output']
    real_A = make_grid(real_A[:NROW], nrow=NROW, normalize=True)
    real_B = make_grid(real_B[:NROW], nrow=NROW, normalize=True)
    fake_A = make_grid(fake_A[:NROW], nrow=NROW, normalize=True)
    fake_B = make_grid(fake_B[:NROW], nrow=NROW, normalize=True)
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    #print(image_grid.shape)
    return image_grid 





Experiment={
    "id":None,
    "home":None, #path for the experiment to save data
    "init_fn":(otu.create_dirs_at_home,{'dirs':['images']}),
    "exit_fn":[(otu.convert_images_to_video,{'images_dir':'images','video_file':'evolution.mp4'}),
               (otu.highlight_latest_result_file,{'input_dir':'images','filename':'final.jpg','ext':'.jpg'})],
    "hparams":{'optim':'Adam',
               'lr':2e-4,
               'loader_n_worker':8,
               'batch_size':BATCH_SIZE,
               #'train_batch_size':1,
               'lr_scheduler':'StepLR',
               'StepLR':{'step_size':3,'gamma':0.8},
               'Adam':{'betas':(0.5,0.999)},
              },
    # Define Experiment Model
    "custom_models":[G_AB,G_BA,D_A,D_B],
    "custom_parameters":[itertools.chain(G_AB.parameters(), G_BA.parameters()),D_A.parameters(),D_B.parameters()],
    # Define function to create train dataset
    "create_train_dataset_fn":create_train_dataset_fn,
    # Define function to create validate dataset
    "create_val_dataset_fn":create_val_dataset_fn,    
    # Define callback function to collate dataset in dataloader, can be None
    "collate_fn_by_dataset":None,
    # Define callback function to preprocess data in each iteration, can be None
    "data_preprocess_fn":data_preprocess_fn,
    # Define Loss function
    "loss_criterions":[criterion_GAN,criterion_cycle,criterion_identity],
    "loss_evaluation_fn":None,
    # Define function to deep insight result in each iteration, can be None
    "epoch_insight_fn":None,
    "epoch_insight_record_fn":None,
    "custom_train_fn":cyclegan_train_fn,
    "custom_val_fn":cyclegan_val_fn,
    #"post_batch_train_fn":[epoch_val_fn,(otu.batch_save_image,{'format_batch':combine_2sets,'interval':5})],
    "post_batch_val_fn":(otu.batch_save_image,{'format_batch':combine_2sets,'interval':1}),
    "train_validate_each_n_batch":400,
}
