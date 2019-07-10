
import argparse
import os

class Options():
    def __init__(self):### 
        # Training settings
        parser = argparse.ArgumentParser(description='Tank Shot')
        parser.add_argument('--LR', default=0.001,type=float,
                            help='Learning rate of the Encoder Network')
        parser.add_argument('--clsLR', default=0.001,type=float,
                            help='Learning rate of the Encoder Network')
        parser.add_argument('--batchSize', default=128,type=int,
                            help='Batch Size')
        parser.add_argument('--nthreads', default=8,type=int,
                            help='threads num to load data')
        parser.add_argument('--tensorname',default='resnet18',type=str,
                            help='tensorboard curve name')
        parser.add_argument('--ways', default=5,type=int,
                            help='number of class for one test')
        parser.add_argument('--shots', default=1,type=int,
                            help='number of pictures of each class to support')
        parser.add_argument('--test_num', default=15,type=int,
                            help='number of pictures of each class for test')
        parser.add_argument('--augnum', default=0,type=int,
                            help='number of augnum')
        parser.add_argument('--data',default='miniImageEmbedding',type=str,
                            help='data loader type')
        parser.add_argument('--network',default='None',type=str,
                            help='load network.t7')
        parser.add_argument('--galleryNum', default=30,type=int,
                            help='number of gallery')
        parser.add_argument('--stepSize', default=10,type=int,
                            help='number of epoch to decay lr')
        parser.add_argument('--Fang', default=3,type=int,
                            help='number of block')
        parser.add_argument('--epoch', default=600,type=int,
                            help='train epoch')
        parser.add_argument('--trainways', default=5,type=int,
                            help='number of class for one episode in training')
        parser.add_argument('--fixScale', default=0,type=int,
                            help='1 means fix Scale ')
        parser.add_argument('--GNet',default='none',type=str,
                            help='load network.t7')
        parser.add_argument('--scratch', default=0,type=int,
                            help='whether to train from scratch')
        parser.add_argument('--fixAttention', default=0,type=int,
                            help='whether to fix attention part')
        parser.add_argument('--fixCls', default=0,type=int,
                            help='whether to fix cls part')
        parser.add_argument('--chooseNum', default=15,type=int,
                            help='number of choosing')

        self.parser = parser

    def parse(self):
        return self.parser.parse_args()
