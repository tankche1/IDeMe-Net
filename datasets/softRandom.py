import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os.path
import csv
import math
import collections
from tqdm import tqdm

import numpy as np
import getpass  
userName = getpass.getuser()


pathminiImageNet = '/home/'+userName+'/data/miniImagenet/'
pathImages = os.path.join(pathminiImageNet,'images/')
# LAMBDA FUNCTIONS
filenameToPILImage = lambda x: Image.open(x)

Fang = 3

patch_xl = []
patch_xr = []
patch_yl = []
patch_yr = []
point = [0,74,148,224]

for i in range(Fang):
    for j in range(Fang):
        patch_xl.append(point[i])
        patch_xr.append(point[i+1])
        patch_yl.append(point[j])
        patch_yr.append(point[j+1])

class miniImagenetEmbeddingDataset(data.Dataset):
    def __init__(self, dataroot = '/home/'+userName+'/data/miniImagenet', type = 'train'):
        if type == 'specialtest':
            type = 'test'
        # Transformations to the image
        if type=='train':
            self.transform = transforms.Compose([filenameToPILImage,
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                                ])
        else:
            self.transform = transforms.Compose([filenameToPILImage,
                                                transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                ])

        def loadSplit(splitFile):
            dictLabels = {}
            with open(splitFile) as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                next(csvreader, None)
                for i,row in enumerate(csvreader):
                    filename = row[0]
                    label = row[1]

                    if label in dictLabels.keys():
                        dictLabels[label].append(filename)
                    else:
                        dictLabels[label] = [filename]
            return dictLabels

        self.miniImagenetImagesDir = os.path.join(dataroot,'images')

        self.data = loadSplit(splitFile = os.path.join(dataroot,type + '.csv'))

        self.type = type
        self.data = collections.OrderedDict(sorted(self.data.items()))
        self.classes_dict = {self.data.keys()[i]:i  for i in range(len(self.data.keys()))} # map NLabel to id(0-99)
        self.bhToClass = {i:self.data.keys()[i]  for i in range(len(self.data.keys()))}

        self.Files = []
        self.belong = []

        for c in range(len(self.data.keys())):
            self.data[self.data.keys()[c]] = self.data[self.data.keys()[c]][:500]
            for file in self.data[self.data.keys()[c]]:
                self.Files.append(file)
                self.belong.append(c)
        
        self.Files = self.Files + self.Files
        self.belong = self.belong + self.belong
        self.__size = len(self.Files)

        print(type,self.__size,len(self.data.keys()))

    def __getitem__(self, index):

        c = self.belong[index]
        File = self.Files[index]

        path = os.path.join(pathImages,str(File))
        images = self.transform(path)

        p = np.random.randint(0,3)
        if p<2:
            # if p==0:
            #     BFile = self.Files[np.random.randint(0,len(self.Files))]
            # else:
            className = self.bhToClass[c]
            BFile = self.data[className][np.random.randint(0,len(self.data[className]))]


            

            Bimages = self.transform(os.path.join(pathImages,str(BFile)))

            for k in range(Fang*Fang):
                weight = np.random.uniform(0,1)
                images[:,patch_xl[k]:patch_xr[k],patch_yl[k]:patch_yr[k]] = weight*images[:,patch_xl[k]:patch_xr[k],patch_yl[k]:patch_yr[k]] + (1-weight) * Bimages[:,patch_xl[k]:patch_xr[k],patch_yl[k]:patch_yr[k]]

        return images,torch.LongTensor([c])

    def __len__(self):
        return self.__size

# ######################################################################
# #plot related
# import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import imshow
# #################################################3

# mu = [0.485, 0.456, 0.406]
# sigma = [0.229, 0.224, 0.225]
# class Denormalize(object):
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std
    
#     def __call__(self, tensor):
#         for t, m, s in zip(tensor, self.mean, self.std):
#             t.mul_(s).add_(m)
#         return tensor


# class Clip(object):
#     def __init__(self):
#         return

#     def __call__(self, tensor):
#         t = tensor.clone()
#         t[t>1] = 1
#         t[t<0] = 0
#         return t

# detransform = transforms.Compose([
#         Denormalize(mu, sigma),
#         Clip(),
#         transforms.ToPILImage(),
#     ])


# def plotPicture(image,name):
#     fig = plt.figure()
#     ax = fig.add_subplot(111)  
#     A = image.clone()
#     ax.imshow(detransform(A))
#     fig.savefig('picture/'+str(name)+'.png')
#     print('picture/'+str(name)+'.png')
#     plt.close(fig)

# if __name__ == '__main__':
#     dataTrain = miniImagenetEmbeddingDataset(type='train')
#     print(len(dataTrain))

#     C,_ = dataTrain.__getitem__(2)
#     print('Size: ',C.size())
#     plotPicture(C,'origin')
#     C = torch.flip(C,[2])
#     plotPicture(C,'flip')

