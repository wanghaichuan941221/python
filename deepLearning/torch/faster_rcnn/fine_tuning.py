import os
import torch
import numpy as np
import torch.utils.data
from PIL import Image
import utils
import transforms as T
from engine import train_one_epoch, evaluate

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
 
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import matplotlib.pyplot as plt
import matplotlib.patches as patches

index = 199


# img = Image.open("data/test_image/css ("+str(index)+").bmp")

# with open("data/lable/css ("+str(index)+").txt") as f: 
#     lines = f.read().split("\n")

# num_objs = int(len(lines)/4)
# boxes = []
# labels = []

# fig,ax = plt.subplots(1)
# ax.imshow(img)

# for i in range(num_objs):
#     box = lines[4*i].split(',')
#     box = list(map(int, box))
#     label = int(lines[4*i+1])
#     rect = patches.Rectangle((box[0],box[1]),box[4]-box[0],box[5]-box[1],linewidth=1,edgecolor='r',facecolor='none')
#     ax.add_patch(rect) 
#     boxes.append(box) 
#     labels.append(label)
# plt.show()


# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# num_classes = 2
# in_features = model.roi_heads.box_predictor.cls_score.in_features

# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)



class TsignDet(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "css_image"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "css_label"))))
        
 
    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "css_image","css ("+str(idx+1)+").bmp")
        mask_path = os.path.join(self.root, "css_label","css ("+str(idx+1)+").txt")
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance with 0 being background
        # mask = Image.open(mask_path)
        with open(mask_path) as f: 
            lines = f.read().split("\n")

        num_objs = int(len(lines)/4)
        boxes = []
        labels = []

        for i in range(num_objs):
            box = lines[4*i].split(',')
            box = list(map(int, box))
            label = int(lines[4*i+1])
            # rect = patches.Rectangle((box[0],box[1]),box[4]-box[0],box[5]-box[1],linewidth=1,edgecolor='r',facecolor='none')
            boxes.append([box[0],box[1],box[4],box[5]]) 
            labels.append(label)
  
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        # labels = torch.ones((num_objs,), dtype=torch.int64)
        labels = torch.as_tensor(labels,dtype=torch.int64)
        # masks = torch.as_tensor(masks, dtype=torch.uint8)
 
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
 
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
 
        if self.transforms is not None:
            img, target = self.transforms(img, target)
 
        return img, target
 
    def __len__(self):
        return len(self.imgs)


# dataset = TsignDet("data")

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
 
    return T.Compose(transforms)

dataset = TsignDet("data",get_transform(train=True))
dataset_test = TsignDet('data', get_transform(train=False))

torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])


data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=8,
    collate_fn=utils.collate_fn)
 
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=8,
    collate_fn=utils.collate_fn)
 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# model = get_instance_segmentation_model(num_classes)

model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
 
# the learning rate scheduler decreases the learning rate by 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)
 
# training
num_epochs = 1

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
 
    # update the learning rate
    lr_scheduler.step()
 
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)


torch.save(model.state_dict(),"model_traffic")

# model.load_state_dict(torch.load("model"))
