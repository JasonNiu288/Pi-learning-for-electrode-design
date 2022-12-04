import torch
import torch.nn as nn
import torch.nn.functional as F


def ConvNet(nc, cnn_dim):


    class Dis_relu(nn.Module):  # it proves this network work excellently
        def __init__(self):
            super(Dis_relu, self).__init__()
            self.main = nn.Sequential(
                nn.Conv3d(nc, 256, 4, 2, 1),
                nn.BatchNorm3d(256),
                nn.ReLU(),

                nn.Conv3d(256, 128, 4, 2, 1),
                nn.BatchNorm3d(128),
                nn.ReLU(),

                nn.Conv3d(128, 64, 4, 2, 1),
                nn.BatchNorm3d(64),
                nn.ReLU(),

                nn.Conv3d(64, 32, 4, 2, 1),
                nn.BatchNorm3d(32),
                nn.ReLU(),

                nn.Conv3d(32, 1, 4, 1, 0),
                #nn.Sigmoid(),
                #nn.Tanh(),
                nn.ReLU(),


            )
        def forward(self, input):
            x = self.main(input)
            x = torch.flatten(x,1)
            #x = 0.5*(torch.tanh(x) + 1)
            return x

    
    if(cnn_dim ==3):
        return Dis_relu

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') !=-1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') !=-1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data,0)
