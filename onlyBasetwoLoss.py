import os
import numpy as np
import argparse
import torch
import torch.optim as optim
from tqdm import *
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision
# import matplotlib.pyplot as plt
from option import Options
from datasets import oneShotBaseCls
from datasets import oneShotUnsuperviseCls

from torch.optim import lr_scheduler
import copy
import time
rootdir = os.getcwd()

args = Options().parse()

from logger import Logger
logger = Logger('./logs/'+args.tensorname)

image_datasets = {}

print('sample from base!')
image_datasets = {x: oneShotBaseCls.miniImagenetOneshotDataset(type=x,ways= (args.trainways if x=='train' else args.ways),shots=args.shots,test_num=args.test_num,epoch=args.epoch,galleryNum=args.galleryNum)
                  for x in ['train', 'val','test']}

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                             shuffle=(x=='train'), num_workers=args.nthreads,worker_init_fn=worker_init_fn)
              for x in ['train', 'val','test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}

######################################################################
# Weight matrix pre-process

patch_xl = []
patch_xr = []
patch_yl = []
patch_yr = []

if args.Fang == 3:
    point = [0,74,148,224]
elif args.Fang == 5:
    point = [0,44,88,132,176,224]
elif args.Fang == 7:
    point = [0,32,64,96,128,160,192,224]



for i in range(args.Fang):
    for j in range(args.Fang):
        patch_xl.append(point[i])
        patch_xr.append(point[i+1])
        patch_yl.append(point[j])
        patch_yr.append(point[j+1])

fixSquare = torch.zeros(1,args.Fang*args.Fang,3,224,224).float()
for i in range(args.Fang*args.Fang):
    fixSquare[:,i,:,patch_xl[i]:patch_xr[i],patch_yl[i]:patch_yr[i]] = 1.00
fixSquare = fixSquare.cuda()

oneSquare = torch.ones(1,3,224,224).float()
oneSquare = oneSquare.cuda()
######################################################################
#plot related
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
#################################################3

mu = [0.485, 0.456, 0.406]
sigma = [0.229, 0.224, 0.225]
class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class Clip(object):
    def __init__(self):
        return

    def __call__(self, tensor):
        t = tensor.clone()
        t[t>1] = 1
        t[t<0] = 0
        return t

detransform = transforms.Compose([
        Denormalize(mu, sigma),
        Clip(),
        transforms.ToPILImage(),
    ])


def plotPicture(image,name):
    fig = plt.figure()
    ax = fig.add_subplot(111)  
    A = image.clone()
    ax.imshow(detransform(A))
    fig.savefig('picture/'+str(name)+'.png')
    plt.close(fig)

######################################################################
# Define the Embedding Network
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
class ClassificationNetwork(nn.Module):
    def __init__(self):
        super(ClassificationNetwork, self).__init__()
        self.convnet = torchvision.models.resnet18(pretrained=False)
        num_ftrs = self.convnet.fc.in_features
        self.convnet.fc = nn.Linear(num_ftrs,64)
        #print(self.convnet)

    def forward(self,inputs):
        outputs = self.convnet(inputs)
        
        return outputs

# resnet18 without fc layer
class weightNet(nn.Module):
    def __init__(self):
        super(weightNet, self).__init__()
        self.resnet = ClassificationNetwork()
        self.resnet.load_state_dict(torch.load('models/'+str(args.network)+'.t7', map_location=lambda storage, loc: storage))
        print('loading ',str(args.network))

        self.conv1 = self.resnet.convnet.conv1
        self.conv1.load_state_dict(self.resnet.convnet.conv1.state_dict())
        self.bn1 = self.resnet.convnet.bn1
        self.bn1.load_state_dict(self.resnet.convnet.bn1.state_dict())
        self.relu = self.resnet.convnet.relu
        self.maxpool = self.resnet.convnet.maxpool
        self.layer1 = self.resnet.convnet.layer1
        self.layer1.load_state_dict(self.resnet.convnet.layer1.state_dict())
        self.layer2 = self.resnet.convnet.layer2
        self.layer2.load_state_dict(self.resnet.convnet.layer2.state_dict())
        self.layer3 = self.resnet.convnet.layer3
        self.layer3.load_state_dict(self.resnet.convnet.layer3.state_dict())
        self.layer4 = self.resnet.convnet.layer4
        self.layer4.load_state_dict(self.resnet.convnet.layer4.state_dict())
        self.layer4 = self.resnet.convnet.layer4
        self.layer4.load_state_dict(self.resnet.convnet.layer4.state_dict())
        self.avgpool = self.resnet.convnet.avgpool

    def forward(self,x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        layer1 = self.layer1(x) # (, 64L, 56L, 56L)
        layer2 = self.layer2(layer1) # (, 128L, 28L, 28L)
        layer3 = self.layer3(layer2) # (, 256L, 14L, 14L)
        layer4 = self.layer4(layer3) # (,512,7,7)
        x = self.avgpool(layer4) # (,512,1,1)
        x = x.view(x.size(0), -1)
        return x

class smallNet(nn.Module):
    def __init__(self):
        super(smallNet, self).__init__()
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

        self.encoder = nn.Sequential( # 6*224*224
            conv_block(6, 32), # 64*112*112
            conv_block(32, 64), # 64*56*56
            conv_block(64, 64), # 64*28*28
            conv_block(64, 32), # 64*14*14
            conv_block(32, 16), # 32*7*7
            Flatten() # 784
        )
        print(self.encoder)

    def forward(self,inputs):

        """                 
        inputs: Batchsize*3*224*224
        outputs: Batchsize*100
        """
        outputs = self.encoder(inputs)
        
        return outputs


class GNet(nn.Module):
    '''
        Two branch's performance are similar one branch's
        So we use one branch here
        Deeper attention network do not bring in benifits
        So we use small network here
    '''
    def __init__(self):
        super(GNet, self).__init__()
        # self.ANet = weightNet()
        # self.BNet = weightNet()
        self.attentionNet = smallNet()

        self.toWeight = nn.Sequential(
                nn.Linear(784,args.Fang*args.Fang),
                # nn.ReLU(),
                # nn.Linear(100,args.Fang*args.Fang),
                # nn.Linear(1024,9),
                # nn.Tanh(),
                # nn.ReLU(),
            )

        self.CNet = weightNet()
        self.fc = nn.Linear(512,64)

        resnet = ClassificationNetwork()
        resnet.load_state_dict(torch.load('models/'+str(args.network)+'.t7', map_location=lambda storage, loc: storage))

        self.fc.load_state_dict(resnet.convnet.fc.state_dict())

        self.scale = nn.Parameter(torch.FloatTensor(1).fill_(1.0), requires_grad=True)
    
    def forward(self,A,B=1,fixSquare=1,oneSquare=1,mode='one'):
        # A,B :[batch,3,224,224] fixSquare:[batch,9,3,224,224] oneSquare:[batch,3,224,224]
        if mode == 'two':
            # Calculate 3*3 weight matrix
            batchSize = A.size(0)
            feature = self.attentionNet(torch.cat((A,B),1))
            weight = self.toWeight(feature) # [batch,3*3]
            
            weightSquare = weight.view(batchSize,args.Fang*args.Fang,1,1,1)
            weightSquare = weightSquare.expand(batchSize,args.Fang*args.Fang,3,224,224)
            weightSquare = weightSquare * fixSquare # [batch,9,3,224,224]
            weightSquare = torch.sum(weightSquare,dim=1) # [batch,3,224,224]

            C = weightSquare*A + (oneSquare - weightSquare) * B
            Cfeature = self.CNet(C)
            return Cfeature, weight, feature

        elif mode == 'one':
            # Calculate feature
            Cfeature = self.CNet(A)
            return Cfeature

        elif mode == 'fc':
            # Go through fc layer, just for debug
            Cfeature = self.fc(A)
            return Cfeature

GNet = GNet()


if args.GNet!='none':
    GNet.load_state_dict(torch.load('models/'+args.GNet+'.t7', map_location=lambda storage, loc: storage))
    print('loading ',args.GNet)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    GNet = nn.DataParallel(GNet)

GNet = GNet.cuda()

#############################################
#Define the optimizer

if torch.cuda.device_count() > 1:
    if args.scratch == 0:
        optimizer_attention = torch.optim.Adam([
                    {'params': GNet.module.attentionNet.parameters()},
                    {'params': GNet.module.toWeight.parameters(), 'lr':  args.LR}
                ], lr=args.LR) # 0.001
        optimizer_classifier = torch.optim.Adam([
                    {'params': GNet.module.CNet.parameters(),'lr': args.clsLR*0.1},
                    {'params': GNet.module.fc.parameters(), 'lr':  args.clsLR}
                ]) # 0.00003
        optimizer_scale = torch.optim.Adam([
                    {'params': GNet.module.scale}
                ], lr=args.LR) # 0.001
    else:
        optimizer_attention = torch.optim.Adam([
                    {'params': GNet.module.ANet.parameters()},
                    {'params': GNet.module.BNet.parameters()},
                    {'params': GNet.module.toWeight.parameters()}
                ], lr=args.LR)
        optimizer_classifier = torch.optim.Adam([
                    {'params': GNet.module.CNet.parameters()},
                    {'params': GNet.module.fc.parameters()}
                ], lr=args.LR)
else:
    optimizer_GNet = torch.optim.Adam([
                {'params': base_params},
                {'params': GNet.toWeight.parameters(), 'lr':  args.LR}
            ], lr=args.LR*0.1)

Attention_lr_scheduler = lr_scheduler.StepLR(optimizer_attention, step_size=40, gamma=0.5)
Classifier_lr_scheduler = lr_scheduler.StepLR(optimizer_classifier, step_size=40, gamma=0.5)
clsCriterion = nn.CrossEntropyLoss()

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^

# Gallery 
Gallery = image_datasets['test'].Gallery
galleryFeature = image_datasets['test'].acquireFeature(GNet,args.batchSize).cpu()


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    # To accelerate training, but observe little effect
    A = GNet.module.scale

    return (torch.pow(x - y, 2)*A).sum(2)

def iterateMix(supportImages,supportFeatures,supportBelongs,supportReals,ways):
    '''
        Inputs:
            supportImages ways,shots,3,224,224
        Outputs:
            AImages [ways*shots*(1+augnum),3,224,224]
            BImages [ways*shots*(1+augnum),3,224,224]
            ABelongs: The label in [0,way-1]
            Reals: The label in [0,63] # Just for debug
    '''
    center = supportFeatures.view(ways,args.shots,-1).mean(1)

    # dists = euclidean_dist(galleryFeature,center) # [ways*unNum,ways]
    Num = galleryFeature.size(0)/10
    with torch.no_grad():
        dists = euclidean_dist(galleryFeature[:Num].cuda(),center)
        for i in range(1,10):
            _end = (i+1)*Num
            if i==9:
                _end = galleryFeature.size(0)
            dist = euclidean_dist(galleryFeature[i*Num:_end].cuda(),center)
            dists = torch.cat((dists,dist),dim=0)

    dists = dists.transpose(1,0) # [ways,ways*unNum]

    AImages = torch.FloatTensor(ways*args.shots*(1+args.augnum),3,224,224)
    ABelongs = torch.LongTensor(ways*args.shots*(1+args.augnum),1)
    Reals = torch.LongTensor(ways*args.shots*(1+args.augnum),1)

    BImages = torch.FloatTensor(ways*args.shots*(1+args.augnum),3,224,224)

    _, bh = torch.topk(dists,args.chooseNum,dim=1,largest=False)

    for i in range(ways):
        for j in range(args.shots):

            AImages[i*args.shots*(1+args.augnum)+j*(args.augnum+1)+0] = supportImages[i*args.shots+j]
            ABelongs[i*args.shots*(1+args.augnum)+j*(args.augnum+1)+0] = supportBelongs[i*args.shots+j]
            Reals[i*args.shots*(1+args.augnum)+j*(args.augnum+1)+0] = supportReals[i*args.shots+j]

            BImages[i*args.shots*(1+args.augnum)+j*(args.augnum+1)+0] = supportImages[i*args.shots+j]

            for k in range(args.augnum):

                p = np.random.randint(0,2)
                if p==0:
                    AImages[i*args.shots*(1+args.augnum)+j*(args.augnum+1)+1+k] = torch.flip(supportImages[i*args.shots+j],[2])
                else:
                    AImages[i*args.shots*(1+args.augnum)+j*(args.augnum+1)+1+k] = supportImages[i*args.shots+j]
                ABelongs[i*args.shots*(1+args.augnum)+j*(args.augnum+1)+1+k] = supportBelongs[i*args.shots+j]
                Reals[i*args.shots*(1+args.augnum)+j*(args.augnum+1)+1+k] = supportReals[i*args.shots+j]

                choose = np.random.randint(0,args.chooseNum)
                BImages[i*args.shots*(1+args.augnum)+j*(args.augnum+1)+1+k] = image_datasets['test'].get_image(Gallery[bh[i][choose]])
                # BImages[i*args.shots*(1+args.augnum)+j*(args.augnum+1)+1+k] = unImages[bh[i][choose]]
                
    return AImages,BImages,ABelongs,Reals

def batchModel(model,AInputs,requireGrad):
    Batch = (AInputs.size(0)+args.batchSize-1)//args.batchSize
    First = True
    Cfeatures = 1


    for b in range(Batch):
        if b<Batch-1:
            midFeature = model(Variable(AInputs[b*args.batchSize:(b+1)*args.batchSize].cuda(),requires_grad=requireGrad))
        else:
            midFeature = model(Variable(AInputs[b*args.batchSize:AInputs.size(0)].cuda(),requires_grad=requireGrad))

        if First:
            First = False
            Cfeatures = midFeature
        else:
            Cfeatures = torch.cat((Cfeatures,midFeature),dim=0)

    return Cfeatures

def train_model(model,num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1000000000.0

    

    print(type(galleryFeature))
    print('Gallery size: ',galleryFeature.size())

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase

        for phase in [ 'test','train']: ####@@@
        # for phase in [ 'test']: ###

            if phase == 'train':
                Attention_lr_scheduler.step()
                Classifier_lr_scheduler.step()

            model.train(False) # To ban batchnorm

            running_loss = 0.0 
            running_accuracy = 0
            running_cls_loss = 0
            running_cls_accuracy= 0

            Times = 0

            # Iterate over data.
            allWeight = {}
            for k in range(args.Fang*args.Fang):
                allWeight[str(k)] = []

            np.random.seed()

            for i,(supportInputs,supportLabels,supportReals,testInputs,testLabels,testReals) in tqdm(enumerate(dataloaders[phase])):

                if epoch ==0 and i>4000:
                    break
                
                Times = Times + 1

                supportInputs = supportInputs.squeeze(0)
                supportLabels = supportLabels.squeeze(0)
                supportReals = supportReals.squeeze(0)

                testInputs = testInputs.squeeze(0)
                testLabels = testLabels.squeeze(0).cuda()

                ways = supportInputs.size(0)/args.shots

                supportFeatures = batchModel(model,supportInputs,requireGrad=False)
                testFeatures = batchModel(model,testInputs,requireGrad=True)

                AInputs, BInputs, ABLabels, ABReals = iterateMix(supportInputs,supportFeatures,supportLabels,supportReals,ways=ways)
                

                Batch = (AInputs.size(0)+args.batchSize-1)//args.batchSize

                First = True
                Cfeatures = 1
                Ccls = 1
                Weights = 0

                '''
                    Pytorch has a bug.
                    Per input's size has to be divisble by the number of GPU
                    So make sure each input's size can be devisble by the number of available GPU
                '''

                for b in range(Batch):
                    if b<Batch-1:
                        _cfeature, weight, middleFeature = model(Variable(AInputs[b*args.batchSize:(b+1)*args.batchSize].cuda(),requires_grad=True),
                            Variable(BInputs[b*args.batchSize:(b+1)*args.batchSize].cuda(),requires_grad=True),
                            Variable(fixSquare.expand(args.batchSize,args.Fang*args.Fang,3,224,224).cuda(),requires_grad=False),
                            Variable(oneSquare.expand(args.batchSize,3,224,224).cuda(),requires_grad=False),
                            mode='two'
                            )
                        _cls = model(_cfeature,B=1,fixSquare=1,oneSquare=1,mode='fc')
                    else:
                        _len = AInputs.size(0)-(b*args.batchSize)
                        _cfeature, weight, middleFeature = model(Variable(AInputs[b*args.batchSize:].cuda(),requires_grad=True),
                            B=Variable(BInputs[b*args.batchSize:].cuda(),requires_grad=True),
                            fixSquare=Variable(fixSquare.expand(_len,args.Fang*args.Fang,3,224,224).cuda(),requires_grad=False),
                            oneSquare=Variable(oneSquare.expand(_len,3,224,224).cuda(),requires_grad=False),
                            mode='two'
                            )
                        _cls = model(_cfeature,B=1,fixSquare=1,oneSquare=1,mode='fc')

                    if First:
                        First = False
                        Cfeatures = _cfeature
                        Weights = weight
                        Ccls = _cls
                    else:
                        Cfeatures = torch.cat((Cfeatures,_cfeature),dim=0)
                        Weights = torch.cat((Weights,weight),dim=0)
                        Ccls = torch.cat((Ccls,_cls),dim=0)

                Weights = Weights.transpose(1,0) # 9*Batch

                for k in range(args.Fang*args.Fang):
                    allWeight[str(k)] = allWeight[str(k)] + Weights[k].view(-1).tolist()

                center = Cfeatures.view(ways,args.shots*(1+args.augnum),-1).mean(1) # [ways,512]
                dists = euclidean_dist(testFeatures,center) # [ways*test_num,ways]

                log_p_y = F.log_softmax(-dists,dim=1).view(ways, args.test_num, -1) # [ways,test_num,ways]

                loss_val = -log_p_y.gather(2, testLabels.view(ways,args.test_num,1)).squeeze().view(-1).mean()
                
                _,y_hat = log_p_y.max(2)

                acc_val = torch.eq(y_hat, testLabels.view(ways,args.test_num)).float().mean()

                # statistics
                running_loss += loss_val.item()
                running_accuracy += acc_val.item()

                # backward + optimize only if in training phase

                if phase == 'train':
                    if (args.fixAttention==0):
                        optimizer_attention.zero_grad()
                        loss_val.backward(retain_graph=True)
                        optimizer_attention.step()
                    if args.fixScale == 0:
                        optimizer_scale.zero_grad()
                        loss_val.backward(retain_graph=True)
                        optimizer_scale.step()
                    _, preds = torch.max(Ccls, 1)
                    ABReals = ABReals.view(ABReals.size(0)).cuda()
                    loss_cls = clsCriterion(Ccls, ABReals)
                    if epoch!=0 and (args.fixCls==0):
                        optimizer_classifier.zero_grad()
                        loss_cls.backward()
                        optimizer_classifier.step()

                    running_cls_loss += loss_cls.item()
                    running_cls_accuracy += torch.eq(preds,ABReals).float().mean()

            epoch_loss = running_loss / (Times*1.0)
            epoch_accuracy = running_accuracy / (Times*1.0)
            epoch_cls_loss = running_cls_loss / (Times*1.0)
            epoch_cls_accuracy = running_cls_accuracy / (Times*1.0)


            info = {
                phase+'loss': epoch_loss,
                phase+'accuracy': epoch_accuracy,
                phase+'_cls_loss': epoch_cls_loss,
                phase+'_cls_accuracy': epoch_cls_accuracy,
            }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, epoch+1)

            print('{} Loss: {:.4f} Accuracy: {:.4f}'.format(
                phase, epoch_loss,epoch_accuracy))

            # print('Classify Loss: {:.4f} Accuracy: {:.4f}'.format(
            #     epoch_cls_loss,epoch_cls_accuracy))

            # deep copy the model
            if phase == 'test' and epoch_loss < best_loss:
                best_loss = epoch_loss

                if torch.cuda.device_count() > 1:
                    best_model_wts = copy.deepcopy(model.module.state_dict())
                else:
                    best_model_wts = copy.deepcopy(model.state_dict())

        
        print()
        if epoch%2 == 0 :
            
            torch.save(best_model_wts,os.path.join(rootdir,'models/'+str(args.tensorname)+'.t7'))
            print('save!')
        

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Loss: {:4f}'.format(best_loss))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


GNet = train_model(GNet, num_epochs=120)
##

# ... after training, save your model 

if torch.cuda.device_count() > 1:
    torch.save(GNet.module.state_dict(),os.path.join(rootdir,'models/'+str(args.tensorname)+'.t7'))
else:
    torch.save(GNet.state_dict(),os.path.join(rootdir,'models/'+str(args.tensorname)+'.t7'))

# .. to load your previously training model:
#model.load_state_dict(torch.load('mytraining.pt'))

