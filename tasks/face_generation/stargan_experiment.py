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
import pandas as pd



#MAX_TRAIN=10000
MAX_TRAIN=0
class CelebADataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train", attributes=None):
        self.transform = transforms.Compose(transforms_)

        self.selected_attrs = attributes
        path="%s/img_align_celeba/*.jpg" % root
        #print('img',path)
        self.files = sorted(glob.glob(path))
        max_to_train=MAX_TRAIN if MAX_TRAIN>0 else -2000
        self.files = self.files[:max_to_train] if mode == "train" else self.files[-2000:]
        self.attr_path = os.path.join(root,'list_attr_celeba.csv')
        self.bbox_path = os.path.join(root,'list_bbox_celeba.csv')
        self.eval_path = os.path.join(root,'list_eval_celeba.csv')
        self.landmarks_path = os.path.join(root,'list_landmarks_align_celeba.csv')
        self.annotations = self.get_annotations()

    def get_annotations(self):
        """Extracts annotations for CelebA"""
        annotations = {}
        df = pd.read_csv(self.attr_path)
        df=df[['image_id']+self.selected_attrs]
        df[self.selected_attrs]=(df[self.selected_attrs]>0).astype(int)
        print(df)
        return df
        lines = [line.rstrip() for line in open(self.label_path, "r")]
        self.label_names = lines[1].split()
        print(self.label_names)
        for _, line in enumerate(lines[2:]):
            filename, *values = line.split()
            labels = []
            for attr in self.selected_attrs:
                idx = self.label_names.index(attr)
                labels.append(1 * (values[idx] == "1"))
            annotations[filename] = labels
        return annotations

    def __getitem__(self, index):
        filepath = self.files[index % len(self.files)]
        filename = filepath.split("/")[-1]
        img = self.transform(Image.open(filepath))
        label = self.annotations[self.annotations.image_id==filename][self.selected_attrs].values[0]
        #label = self.annotations[filename]
        label = torch.FloatTensor(np.array(label))
        #print(img.shape,label.shape)

        return img, label

    def __len__(self):
        return len(self.files)


DATASET_NAME='celeba'
nWidth=128
nHeight=128
nChannel=3
BATCH_SIZE=16
SELECTED_ATTRS=["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"]
c_dim=len(SELECTED_ATTRS)


# Configure dataloaders
train_transforms = [
    transforms.Resize(int(1.12 * nHeight), Image.BICUBIC),
    transforms.RandomCrop(nHeight),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

val_transforms = [
    transforms.Resize((nHeight, nWidth), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]





""# 2 模型

# 2.1 模型参数

HPARAMS={}
#IMAGE_SIZE = 784
#HPARAMS['LATENT_DIM']=100

HPARAMS['RES_BLOCK']=9
HPARAMS['lr']=2e-4
HPARAMS['optim']='Adam'
HPARAMS['WIDTH']=nWidth
HPARAMS['HEIGHT']=nHeight
HPARAMS['CHANNEL']=nChannel
HPARAMS['N_CRITIC']=5


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)




def create_train_dataset_fn(path):
    return CelebADataset(path, transforms_=train_transforms, mode="train", attributes=SELECTED_ATTRS)

def create_val_dataset_fn(path):
    return CelebADataset(path, transforms_=val_transforms, mode="test", attributes=SELECTED_ATTRS)

G_NAME='Resnet'

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(in_features, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(in_features, affine=True, track_running_stats=True),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class GeneratorResNet(nn.Module):
    def __init__(self, img_shape=(3, 128, 128), res_blocks=9, c_dim=5):
        super(GeneratorResNet, self).__init__()
        channels, img_size, _ = img_shape

        # Initial convolution block
        model = [
            nn.Conv2d(channels + c_dim, 64, 7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        curr_dim = 64
        for _ in range(2):
            model += [
                nn.Conv2d(curr_dim, curr_dim * 2, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ]
            curr_dim *= 2

        # Residual blocks
        for _ in range(res_blocks):
            model += [ResidualBlock(curr_dim)]

        # Upsampling
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ]
            curr_dim = curr_dim // 2

        # Output layer
        model += [nn.Conv2d(curr_dim, channels, 7, stride=1, padding=3), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat((x, c), 1)
        return self.model(x)



##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, img_shape=(3, 128, 128), c_dim=5, n_strided=6):
        super(Discriminator, self).__init__()
        channels, img_size, _ = img_shape

        def discriminator_block(in_filters, out_filters):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1), nn.LeakyReLU(0.01)]
            return layers

        layers = discriminator_block(channels, 64)
        curr_dim = 64
        for _ in range(n_strided - 1):
            layers.extend(discriminator_block(curr_dim, curr_dim * 2))
            curr_dim *= 2

        self.model = nn.Sequential(*layers)

        # Output 1: PatchGAN
        self.out1 = nn.Conv2d(curr_dim, 1, 3, padding=1, bias=False)
        # Output 2: Class prediction
        kernel_size = img_size // 2 ** n_strided
        self.out2 = nn.Conv2d(curr_dim, c_dim, kernel_size, bias=False)

    def forward(self, img):
        feature_repr = self.model(img)
        out_adv = self.out1(feature_repr)
        out_cls = self.out2(feature_repr)
        return out_adv, out_cls.view(out_cls.size(0), -1)




G= GeneratorResNet(img_shape=(3, 128, 128), res_blocks=6, c_dim=5)
D= Discriminator(img_shape=(3, 128, 128), c_dim=5)

#定义训练函数，返回每次训练待loss情况
# Loss weights
lambda_cls = 1
lambda_rec = 10
lambda_gp = 10

def criterion_cls(logit, target):
    return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)

def compute_gradient_penalty(D, real_samples, fake_samples,device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates, _ = D(interpolates)
    fake = torch.autograd.Variable(torch.FloatTensor(np.ones(d_interpolates.shape)), requires_grad=False).to(device)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty




def data_preprocess_fn(runtime,experiment,data):
    print(data)
    return data,None



def stargan_train_fn(runtime,experiment):
#def cyclegan_train_fn(models,criterions,optimizers,data,target,device):
    models = experiment['custom_models']
    criterions = experiment['loss_criterions']
    optimizers = experiment['custom_optimizers']
    device = experiment['device']
    imgs = runtime['input']
    labels = runtime['target']
    batch_idx = runtime['batch_idx']
    assert len(models)==2 and len(optimizers)==2

    generator, discriminator = models
    optimizer_G,optimizer_D = optimizers
    criterion_cycle = criterions[0]


    # Sample labels as generator inputs
    
    sampled_c = torch.autograd.Variable(torch.FloatTensor(np.random.randint(0, 2, (imgs.size(0), c_dim)))).to(device)
    # Generate fake batch of images
    fake_imgs = generator(imgs, sampled_c)

    # ---------------------
    #  Train Discriminator
    # ---------------------

    optimizer_D.zero_grad()

    # Real images
    real_validity, pred_cls = discriminator(imgs)
    # Fake images
    fake_validity, _ = discriminator(fake_imgs.detach())
    # Gradient penalty
    gradient_penalty = compute_gradient_penalty(discriminator, imgs.data, fake_imgs.data,device)
    # Adversarial loss
    loss_D_adv = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
    # Classification loss
    loss_D_cls = criterion_cls(pred_cls, labels)
    # Total loss
    loss_D = loss_D_adv + lambda_cls * loss_D_cls

    loss_D.backward()
    optimizer_D.step()

    optimizer_G.zero_grad()

    # Every n_critic times update generator
    if batch_idx % HPARAMS['N_CRITIC'] == 0:

        # -----------------
        #  Train Generator
        # -----------------

        # Translate and reconstruct image
        gen_imgs = generator(imgs, sampled_c)
        recov_imgs = generator(gen_imgs, labels)
        # Discriminator evaluates translated image
        fake_validity, pred_cls = discriminator(gen_imgs)
        # Adversarial loss
        loss_G_adv = -torch.mean(fake_validity)
        # Classification loss
        loss_G_cls = criterion_cls(pred_cls, sampled_c)
        # Reconstruction loss
        loss_G_rec = criterion_cycle(recov_imgs, imgs)
        # Total loss
        loss_G = loss_G_adv + lambda_cls * loss_G_cls + lambda_rec * loss_G_rec

        loss_G.backward()
        optimizer_G.step()
        return gen_imgs,{'loss_D':loss_D.item(),
                #'loss_D_adv':loss_D_adv.item(),
                #'loss_D_cls':loss_D_cls.item(),
                'loss_G':loss_G.item(),
                #'loss_G_adv':loss_G_adv.item(),
                #'loss_G_cls':loss_G_cls.item(),
                #'loss_G_rec':loss_G_rec.item()
                }
    return fake_imgs,{'loss_D':loss_D.item(),
            #'loss_D_adv':loss_D_adv.item(),
            #'loss_D_cls':loss_D_cls.item()
            }

label_changes = [
    ((0, 1), (1, 0), (2, 0)),  # Set to black hair
    ((0, 0), (1, 1), (2, 0)),  # Set to blonde hair
    ((0, 0), (1, 0), (2, 1)),  # Set to brown hair
    ((3, -1),),  # Flip gender
    ((4, -1),),  # Age flip
]

def stargan_val_fn(runtime,experiment):
    #print('call val')
    device = experiment['device']
    val_imgs = runtime['input']
    models = experiment['custom_models']
    criterions = experiment['loss_criterions']
    val_labels = runtime['target']
    generator=models[0]
    #print(val_imgs.shape,val_labels.shape)
    img_samples = None
    for i in range(10):
        img,label=val_imgs[i],val_labels[i]
        #print(img.shape,label.shape)
        # Repeat for number of label changes
        imgs = img.repeat(c_dim, 1, 1, 1)
        labels = label.repeat(c_dim, 1)
        # Make changes to labels
        for sample_i, changes in enumerate(label_changes):
            for col, val in changes:
                labels[sample_i, col] = 1 - labels[sample_i, col] if val == -1 else val

        # Generate translations
        gen_imgs = generator(imgs, labels)
        # Concatenate images by width
        gen_imgs = torch.cat([x for x in gen_imgs.data], -1)
        img_sample = torch.cat((img.data, gen_imgs), -1)
        # Add as row to generated samples
        img_samples = img_sample if img_samples is None else torch.cat((img_samples, img_sample), -2)
    #print('val output',img_samples.shape)
    return img_samples,{}




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
               #'lr_scheduler':'StepLR',
               #'StepLR':{'step_size':3,'gamma':0.8},
               'Adam':{'betas':(0.5,0.999)},
               "minibatch_insight_interval":400
              },
    # Define Experiment Model
    "custom_models":[G,D],
    #"custom_parameters":[itertools.chain(G_AB.parameters(), G_BA.parameters()),D_A.parameters(),D_B.parameters()],
    # Define function to create train dataset
    "create_train_dataset_fn":create_train_dataset_fn,
    # Define function to create validate dataset
    "create_val_dataset_fn":create_val_dataset_fn,    
    # Define callback function to collate dataset in dataloader, can be None
    "collate_fn_by_dataset":None,
    # Define callback function to preprocess data in each iteration, can be None
    #"data_preprocess_fn":data_preprocess_fn,
    # Define Loss function
    "loss_criterions":[torch.nn.L1Loss()],
    #"loss_evaluation_fn":None,
    # Define function to deep insight result in each iteration, can be None
    "model_init_fn":weights_init_normal,
    "epoch_insight_fn":None,
    "epoch_insight_record_fn":None,
    "custom_train_fn":stargan_train_fn,
    "custom_val_fn":stargan_val_fn,
    #"post_batch_train_fn":[(otu.batch_save_image,{'interval':100})],
    "post_batch_val_fn":(otu.batch_save_image,{'combine':(10,5)}),
    "train_validate_each_n_batch":400,
    "checkpoint_n_epoch":2
}
