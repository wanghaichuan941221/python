import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
from torch import optim
from layers.modules import MultiBoxLoss
from data import *
from utils.augmentations import SSDAugmentation
import time

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
from ssd import build_ssd

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = 1e-3 * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

device = ("cuda" if torch.cuda.is_available() else "cpu")
cfg = coco
num_classes = 21
cfg["num_classes"] = num_classes
# net = build_ssd('train', 300, 21)    # initialize SSD
net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
net.load_weights('weights/ssd300_mAP_77.43_v2.pth')

net.num_classes = num_classes

vgg_weights = torch.load("weights/vgg16_reducedfc.pth") 
net.vgg.load_state_dict(vgg_weights)

# net.extras.apply(weights_init)
# net.loc.apply(weights_init)
# net.conf.apply(weights_init)

params = [p for p in net.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5,False)

net.train()

# loss counters
loc_loss = 0
conf_loss = 0
epoch = 0



dataset = COCODetection(root="/home/haichuan/data/coco",transform=SSDAugmentation(cfg['min_dim'],MEANS))

data_loader = data.DataLoader(dataset, batch_size=1,num_workers=1,shuffle=True, collate_fn=detection_collate,pin_memory=True)

epoch_size = len(dataset) // 2
step_index = 0

batch_iterator = iter(data_loader)
for iteration in range(0, cfg['max_iter']):
    if  iteration != 0 and (iteration % epoch_size == 0):
        # reset epoch loss counters
        loc_loss = 0
        conf_loss = 0
        epoch += 1

    if iteration in cfg['lr_steps']:
        step_index += 1
        adjust_learning_rate(optimizer, 0.1, step_index)

    # load train data
    images, targets = next(batch_iterator)
    
    images = Variable(images.cuda())
    targets = [Variable(ann.cuda(), volatile=True) for ann in targets]

    # forward
    t0 = time.time()
    out = net(images)
    # backprop
    optimizer.zero_grad()
    
    loss_l, loss_c = criterion(out, targets)
    loss = loss_l + loss_c
    loss.backward()
    optimizer.step()
    t1 = time.time()
    loc_loss += loss_l.item()
    conf_loss += loss_c.item()

    if iteration % 10 == 0:
        print('timer: %.4f sec.' % (t1 - t0))
        print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')

    

    if iteration != 0 and iteration % 5000 == 0:
        print('Saving state, iter:', iteration)
        torch.save(net.state_dict(), 'weights/ssd300_COCO_' +
                    repr(iteration) + '.pth')
    
    if iteration>2:
        break
# torch.save(net.state_dict(),"demo/res.pth")


# image = cv2.imread("demo/me.jpg")
# rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# x = cv2.resize(image, (300, 300)).astype(np.float32)
# x = torch.from_numpy(x).permute(2, 0, 1)

# xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
# xx = xx.to(device)

# y = net(xx)

# import matplotlib.pyplot as plt
# from data import VOC_CLASSES as labels

# top_k=10
# plt.figure(figsize=(10,10))
# colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
# plt.imshow(rgb_image)  # plot the image for matplotlib
# currentAxis = plt.gca()

# detections = y.data
# # scale each detection back up to the image
# scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
# for i in range(detections.size(1)):
#     j = 0
#     while detections[0,i,j,0] >= 0.6:
#         score = detections[0,i,j,0]
#         label_name = labels[i-1]
#         display_txt = '%s: %.2f'%(label_name, score)
#         pt = (detections[0,i,j,1:]*scale).cpu().numpy()
#         coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
#         color = colors[i]
#         currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
#         currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
#         j+=1
        
# plt.show()