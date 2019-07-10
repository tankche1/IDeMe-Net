import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os.path
import csv
import math
import collections
from tqdm import tqdm
import datetime

import numpy as np
import numpy
#from watch import NlabelTovector
import getpass  
userName = getpass.getuser()

pathminiImageNet = '/home/'+userName+'/data/miniImagenet/'
pathImages = os.path.join(pathminiImageNet,'images/')
# LAMBDA FUNCTIONS
filenameToPILImage = lambda x: Image.open(x)

np.random.seed(2191)

patch_xl = [0,0,0,74,74,74,148,148,148]
patch_xr = [74,74,74,148,148,148,224,224,224]
patch_yl = [0,74,148,0,74,148,0,74,148]
patch_yr = [74,148,224,74,148,224,74,148,224]

class miniImagenetOneshotDataset(data.Dataset):
    def __init__(self, dataroot = '/home/'+userName+'/data/miniImagenet', type = 'train',ways=5,shots=1,test_num=1,epoch=100,eraseNum=0,unNum=15):
        # oneShot setting
        self.ways = ways
        self.shots = shots
        self.test_num = test_num # indicate test number of each class
        self.__size = epoch

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

        self.unData = loadSplit(splitFile = os.path.join(dataroot,'train' + '.csv'))
        self.data = loadSplit(splitFile = os.path.join(dataroot,type + '.csv'))
        

        # for c in range(len(self.data.keys())):
        #     for file in self.data[self.data.keys()[c]]:
        #         self.Files.append(file)
        #         self.belong.append(c)

        
        '''
        for key,value in self.data.items():
            if self.keyTobh.has_key(key):
                continue
            else:
                self.keyTobh[key] = Tot
                Tot = Tot + 1

        '''

        self.type = type
        self.data = collections.OrderedDict(sorted(self.data.items()))
        self.unData = collections.OrderedDict(sorted(self.unData.items()))
        self.eraseNum = eraseNum
        self.unNum = unNum

        self.keyTobh = {}
        for c in range(len(self.data.keys())):
            self.keyTobh[self.data.keys()[c]] = c

        # self.keyTobh = {}
        for c in range(len(self.unData.keys())):
            self.keyTobh[self.unData.keys()[c]] = c

        #print(self.keyTobh)

    def __getitem__(self, index):
        # ways,shots,3,224,224
        #numpy.random.seed(index+datetime.datetime.now().second + datetime.datetime.now().microsecond)
        supportFirst = True
        supportImages = 1
        supportBelongs = torch.LongTensor(self.ways*self.shots,1)
        supportReal = torch.LongTensor(self.ways*self.shots,1)

        testFirst = True
        testImages = 1
        testBelongs = torch.LongTensor(self.ways*self.test_num,1)
        testReal = torch.LongTensor(self.ways*self.test_num,1)
        testFiles = []

        unFirst = True
        unImages = 1
        unBelongs = torch.LongTensor(self.ways*self.unNum,1)
        unReal = torch.LongTensor(self.ways*self.unNum,1)

        selected_classes = np.random.choice(self.data.keys(), self.ways, False)
        for i in range(self.ways):
            files = np.random.choice(self.data[selected_classes[i]], self.shots, False)
            for j in range(self.shots):
                image = self.transform(os.path.join(pathImages,str(files[j])))
                image = image.unsqueeze(0)
                if supportFirst:
                    supportFirst=False
                    supportImages = image
                else:
                    supportImages = torch.cat((supportImages,image),0)
                supportBelongs[i*self.shots+j,0] = i
                supportReal[i*self.shots+j,0] = self.keyTobh[selected_classes[i]]


            files = np.random.choice(self.data[selected_classes[i]], self.test_num, False)
            for j in range(self.test_num):
                # image = self.transform(os.path.join(pathImages,str(files[j])))
                # image = image.unsqueeze(0)

                testFiles.append(os.path.join(pathImages,str(files[j])))
                # if testFirst:
                #     testFirst = False
                #     testImages = image
                # else:
                #     testImages = torch.cat((testImages,image),0)
                testBelongs[i*self.test_num+j,0] = i
                testReal[i*self.test_num+j,0] = self.keyTobh[selected_classes[i]]

            selected_class = np.random.choice(self.unData.keys(), self.unNum, True)
            # files = np.random.choice(self.data[selected_class[i]], self.unNum, False)
            for j in range(self.unNum):
                file = np.random.choice(self.unData[selected_class[j]], 1, False)
                image = self.transform(os.path.join(pathImages,str(file[0])))
                image = image.unsqueeze(0)
                if unFirst:
                    unFirst = False
                    unImages = image
                else:
                    unImages = torch.cat((unImages,image),0)
                unBelongs[i*self.unNum+j,0] = i
                unReal[i*self.unNum+j,0] = self.keyTobh[selected_class[j]]


        return supportImages,supportBelongs,supportReal,testFiles,testBelongs,testReal,unImages,unBelongs,unReal

    def __len__(self):
        return self.__size

if __name__ == '__main__':
    dataTrain = torch.utils.data.DataLoader(miniImagenetOneshotDataset(type='train',ways=5,shots=5,test_num=15,epoch=1000),batch_size=1,shuffle=False,num_workers=1)


    # for j in range(5):

    # for i,(supportInputs,supportBelongs,supportReals,testInputs,testBelongs,testReals,unInputs,unBelongs,unReals) in tqdm(enumerate(dataTrain)):
    #     pass
    #     #if i<3:
    #     #    print(supportInputs[0][0][0])
        
    #     haha = 1
    #     if i<=5:
    #         print(i,supportInputs.size(),supportBelongs.size(),testInputs.size(),testBelongs.size())
    #     #print(testLabels)
    #     if i<2:
    #         print(supportBelongs,supportReals)
        


