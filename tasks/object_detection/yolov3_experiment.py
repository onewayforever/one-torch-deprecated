import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import one_torch_utils as otu
from torchvision import datasets, transforms

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.augmentations import *
from utils.transforms import *
from utils.parse_config import *
from torchvision.utils import save_image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from functools import reduce

infer_imgs=[]
infer_detections=[]

model_def='yolov3.cfg'

img_size=416

class_names = load_classes('/dev/shm/yolo_2014/coco.names')

def create_train_dataset_fn(path):
    return  ListDataset(path, img_size=img_size,multiscale=Experiment['hparams']['multiscale'], transform=AUGMENTATION_TRANSFORMS)

def create_val_dataset_fn(path):
    return  ListDataset(path, img_size=img_size,multiscale=Experiment['hparams']['multiscale'], transform=DEFAULT_TRANSFORMS)

def create_infer_dataset_fn(path):
    return ImageFolder(path, transform=transforms.Compose([DEFAULT_TRANSFORMS, Resize(img_size)]))

def data_preprocess_fn(runtime,experiment,data):
    paths,imgs,targets=data
    runtime['img_paths']=paths
    return imgs,targets

def infer_data_preprocess_fn(runtime,experiment,data):
    #print(data)
    paths,imgs=data
    runtime['img_paths']=paths
    return imgs

model = Darknet(model_def)
#model.apply(weights_init_normal)



def yolo_train_fn(runtime,experiment):
    models = experiment.get('custom_models')
    device = experiment.get('device')
    optimizers = experiment.get('custom_optimizers')
    imgs = runtime['input']
    targets = runtime['target']
    model = models[0]
    #optimizer = optimizers[0]
    #optimizer.zero_grad()
    loss,output = model(imgs,targets)
    runtime['output'] = output
    #print('train')
    #print(imgs.shape,targets.shape)
    #print(output.shape)
    loss.backward()
    #batches_done=runtime.get('train_batches')*runtime.get('epoch')+runtime.get('batch_idx')
    batches_done = runtime['batches_done']
    if batches_done % experiment['hparams'].get('gradient_accumulations') == 0:
        # Accumulates gradient before each step
        optimizer = optimizers[0]
        optimizer.step()
        optimizer.zero_grad()
    #optimizer.step()
    return output,{'loss':loss.item()}

def yolo_val_fn(runtime,experiment):
    models = experiment.get('custom_models')
    device = experiment.get('device')
    imgs = runtime['input']
    targets = runtime['target']
    model = models[0]
    hparams=experiment.get('hparams')
    iou_thres=hparams.get('iou_thres')
    nms_thres=hparams.get('nms_thres')
    conf_thres=hparams.get('conf_thres')
    # Extract labels
    #labels += targets[:, 1].tolist()
    # Rescale target
    #targets[:, 2:] = xywh2xyxy(targets[:, 2:])
    #targets[:, 2:] *= img_size

    #imgs = Variable(imgs.type(torch.Tensor), requires_grad=False)

    with torch.no_grad():
        loss,output = model(imgs,targets)
        #print('in val')
        #print(imgs.shape,targets.shape)
        #print(output.shape)
        output = non_max_suppression(output, conf_thres=conf_thres, nms_thres=nms_thres)

    return output,{'loss':loss.item()}



def epoch_insight_fn(runtime,experiment,ret):
    #print('epoch')
    res=''
    result_list=reduce(lambda x,y:x+y,runtime.get('epoch_results'))
    #print('epoch_insight_fn')
    #print(result_list)
    #print(len(result_list))
    if len(result_list)==0:
        return ret
    labels=runtime.get('labels')
   
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*result_list))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    evaluation_metrics = [
        ("validation/precision", precision.mean()),
        ("validation/recall", recall.mean()),
        ("validation/mAP", AP.mean()),
        ("validation/f1", f1.mean()),
    ]
    #logger.list_of_scalars_summary(evaluation_metrics, epoch)

    # Print class APs and mAP
    ap_table = [["Index", "Class name", "AP"]]
    for i, c in enumerate(ap_class):
        ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
    #print(AsciiTable(ap_table).table)
    print(ap_table)
    print(f"---- mAP {AP.mean()}")
    #res="Average Precisions:\n"
    #for i, c in enumerate(ap_class):
    #    res+=f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}\n"

    #res+=f"mAP: {AP.mean()}"
    runtime['labels']=None
    return ret 

def epoch_insight_record(runtime,experiment,ret):
    #print('called?')
    #print(runtime['output'])
    #print(runtime['target'])
    imgs = runtime['input']
    output = runtime['output']
    targets = runtime['target']
    targets = targets.clone().detach().cpu()
    #print(imgs.shape)
    #print(targets.shape)
    #print(output)
    labels = runtime.get('labels')
    batch_idx = runtime.get('batch_idx')
    if labels is None:
        labels=[]
        runtime['labels']=labels
    #print(batch_idx) 
    if batch_idx==1:
        #epoch = runtime.get('epoch')
        #total = runtime.get('total_train_batches')
        batches_done = runtime.get('batches_done')
        #print(imgs.shape)
        filename = os.path.join(otu.get_home_path('images'),'{:0>6d}.jpg'.format(batches_done))
        #save_image(imgs[:(2*2)], filename, nrow=2, normalize=True)
        paths=runtime.get('img_paths')
        save_detect(paths,imgs,output,filename)

        

    labels += targets[:, 1].tolist()
    # Rescale target
    targets[:, 2:] = xywh2xyxy(targets[:, 2:])
    targets[:, 2:] *= img_size
    labels += targets[:, 1].tolist()
    #print(output)
    #print(targets)
    batch_info = get_batch_statistics(output, targets, iou_threshold=0.5)
    #print(batch_info)
    #print(len(batch_info))
    #print(type(batch_info))
    #result_list=runtime.get('epoch_results')
    #result_list+=batch_info
    return batch_info 

def save_detect(paths,input_imgs,all_detections,filename):
    #detections = non_max_suppression(detections, 0.5, 0.5)
    #print(input_imgs.shape,detections.shape,filename)
    #print(input_imgs.shape)
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 80)]
    #print('all colors')
    #print(colors)
    #for item in detections:
    #    print(item.shape)
    
    #detections = non_max_suppression(detections, 0.5, 0.5)
    #for item in detections:
    #    print(item.shape)
    #print(filename,input_imgs.shape[0],len(all_detections))
    #for i in range(input_imgs.shape)
    #img_list=[]
    #input_imgs = input_imgs.clone().detach().cpu()
    #print(paths)
    img_list=[]
    for i in range(len(all_detections)):
        #print('\n\ncv write\n\n')
        #img=input_imgs[i].numpy()
        path=paths[i]
        #print(path)
        #img=input_imgs[i].permute(1,2,0).numpy()*255
        detections = all_detections[i]
        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                if cls_conf.item()<0.9:
                    continue

                #print("\t+ Label: %s, Conf: %.5f" % (class_names[int(cls_pred)], cls_conf.item()))
                #print(x1,y1,x2,y2)
                x1=min(max(0,x1),img_size)
                y1=min(max(0,y1),img_size)
                x2=min(max(0,x2),img_size)
                y2=min(max(0,y2),img_size)

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=class_names[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        #filename = os.path.basename(path).split(".")[0]
        output_path = os.path.join("./", f".{i}.png")
        #output_path = os.path.join("output", f"{filename}.png")
        #print(output_path)
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
        plt.close()
        #print('save image')
        tmp=cv2.imread(output_path)
        tmp=cv2.resize(tmp,(img_size,img_size),interpolation=cv2.INTER_CUBIC)
        img_list.append(torch.Tensor(tmp).permute(2,0,1))
    imgs=torch.stack(img_list)
    #print(imgs.shape)
    save_image(imgs, filename, nrow=2, normalize=True)
    return

def gather_infer_detections(runtime,experiment):
    global infer_imgs
    global infer_detections
    img_paths = runtime.get('img_paths')
    detections = runtime.get('output')
    with torch.no_grad():
        detections = non_max_suppression(detections, 0.5, 0.5)
    infer_imgs.extend(img_paths)
    infer_detections.extend(detections)
    
    
def save_infer_detections(runtime,experiment):
    global infer_imgs
    global infer_detections
    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 80)]

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(infer_imgs, infer_detections)):

        print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print("\t+ Label: %s, Conf: %.5f" % (class_names[int(cls_pred)], cls_conf.item()))
                x1=min(max(0,x1),img_size)
                y1=min(max(0,y1),img_size)
                x2=min(max(0,x2),img_size)
                y2=min(max(0,y2),img_size)

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=class_names[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = os.path.basename(path).split(".")[0]
        output_path = os.path.join('.', f"{filename}.png")
        print(output_path)
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
        plt.close()

#
# Porting from https://github.com/eriklindernoren/PyTorch-YOLOv3
#

Experiment={
    "hparams":{'optim':'Adam',
               'lr':2e-4,
               'batch_size':8,
               'infer_batch_size':1,
               'loader_n_worker':8,
               'Adam':{'betas':(0.5,0.999)},
               'gradient_accumulations':2,
               #'train_dataset_path':'/dev/shm/yolo_2014/train2014.txt',
               #'val_dataset_path':'/dev/shm/yolo_2014/val2014.txt',
               #'train_dataset_path':'/dev/shm/yolo_2014/train2014_tiny.txt',
               #'val_dataset_path':'/dev/shm/yolo_2014/val2014_tiny.txt',
               #'infer_dataset_path':'/dev/shm/yolo_2014/samples',
               ###custom options
               'model_def':model_def,
               'multiscale':True,
               'nms_thres':0.5,
               'conf_thres':0.5,
               'iou_thres':0.5
              },
    #'train_dataset_path':'/dev/shm/yolo_2014/train2014.txt',
    #'val_dataset_path':'/dev/shm/yolo_2014/val2014.txt',
    #'infer_dataset_path':'/dev/shm/yolo_2014/samples',
    "codes_to_backup":['yolov3.cfg','models.py','utils'],
    "init_fn":(otu.create_dirs_at_home,{'dirs':['images','infers']}),
    "exit_fn":[(otu.convert_images_to_video,{'images_dir':'images','video_file':'evoluton.mp4'}),
               (otu.highlight_latest_result_file,{'input_dir':'images','filename':'final.jpg','ext':'.jpg'})],
    # Define Experiment Model
    "custom_models":[model],
    # Define function to create train dataset
    "create_train_dataset_fn":create_train_dataset_fn,
    # Define function to create validate dataset
    "create_val_dataset_fn":create_val_dataset_fn,    
    "create_infer_dataset_fn":create_infer_dataset_fn,    
    # Define callback function to collate dataset in dataloader, can be None
    "collate_fn_by_dataset":lambda x:x.collate_fn,
    # Define callback function to preprocess data in each iteration, can be None
    "data_preprocess_fn":data_preprocess_fn,
    "infer_data_preprocess_fn":infer_data_preprocess_fn,
    # Define Loss function
    "loss_criterions":[],
    "custom_train_fn":yolo_train_fn,
    "custom_val_fn":yolo_val_fn,
    #"loss_evaluation_fn":loss_evaluation_fn,
    # Define function to deep insight result in each iteration, can be None
    "post_epoch_val_fn":epoch_insight_fn,
    "post_batch_val_fn":epoch_insight_record,
    #"val_epoch_insight_fn":epoch_insight_fn,
    #"val_epoch_insight_record_fn":epoch_insight_record,
    "post_batch_infer_fn":gather_infer_detections,
    "post_infer_fn":save_infer_detections,
    "train_validate_each_n_epoch":1,
    "validate_n_batch_in_train":0,
    "train_validate_final_with_best":True
}
