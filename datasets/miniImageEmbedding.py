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
#from watch import NlabelTovector
import getpass  
userName = getpass.getuser()

np.random.seed(2191)  # for reproducibility

pathminiImageNet = '/home/'+userName+'/data/miniImagenet/'
pathImages = os.path.join(pathminiImageNet,'images/')
# LAMBDA FUNCTIONS
filenameToPILImage = lambda x: Image.open(x)
PiLImageResize = lambda x: x.resize((84,84))

forbidden = {}

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

                    if forbidden.has_key(str(label)):
                        continue
                    if label in dictLabels.keys():
                        dictLabels[label].append(filename)
                    else:
                        dictLabels[label] = [filename]
            return dictLabels

        #requiredFiles = ['train','val','test']
        self.miniImagenetImagesDir = os.path.join(dataroot,'images')

        self.data = loadSplit(splitFile = os.path.join(dataroot,type + '.csv'))

        self.type = type
        self.data = collections.OrderedDict(sorted(self.data.items()))
        self.classes_dict = {self.data.keys()[i]:i  for i in range(len(self.data.keys()))} # map NLabel to id(0-99)
        

        self.Files = []
        self.belong = []

        self.keyTobh = {}
        for c in range(len(self.data.keys())):
            self.keyTobh[self.data.keys()[c]] = c

        torch.save(self.keyTobh,os.path.join(pathminiImageNet,'classTobh.t7'))


        for c in range(len(self.data.keys())):
            for file in self.data[self.data.keys()[c]]:
                self.Files.append(file)
                self.belong.append(c)

        self.__size = len(self.Files)
        print(self.keyTobh)

    def __getitem__(self, index):

        c = self.belong[index]
        File = self.Files[index]

        path = os.path.join(pathImages,str(File))
        images = self.transform(path)
        return images,torch.Tensor(NlabelTovector[self.data.keys()[c]]),torch.LongTensor([c])

    def __len__(self):
        return self.__size


dataTrain = miniImagenetEmbeddingDataset(type='train')
print(len(dataTrain))


