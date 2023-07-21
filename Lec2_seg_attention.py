#!/usr/bin/env python     [>jupyter nbconvert --to script lec2_2023.ipynb]
# Liam_seg code + https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py
import torch, os
from skimage import io, transform
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
from torchmetrics.classification import Dice as c_dice
from torchmetrics.functional import dice as f_dice
import time
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import platform, psutil, datetime
from pynvml import *

use_cuda = torch.cuda.is_available();  start_time = time.time() 
device = torch.device('cuda' if use_cuda else 'cpu') #print(device) #
# ## UNET
# - UNet is a particular type of CNN, named after its U shape
# - [Original paper](https://arxiv.org/pdf/1505.04597.pdf)
# <p align="center">
# <img src="unet.png" width="1000" title="Image." >
# </p>
#=======================================================================================
class conv_block(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.up(x)
        return x

class single_conv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )       
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x*psi

class AttU_Net(nn.Module): # https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py
    def __init__(self,img_ch=3,output_ch=1):
        super(AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Conv1 = conv_block(in_channels=img_ch,out_channels=64)
        self.Conv2 = conv_block(in_channels=64,out_channels=128)
        self.Conv3 = conv_block(in_channels=128,out_channels=256)
        self.Conv4 = conv_block(in_channels=256,out_channels=512)
        self.Conv5 = conv_block(in_channels=512,out_channels=1024)

        self.Up5 = up_conv(in_channels=1024,out_channels=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(in_channels=1024, out_channels=512)

        self.Up4 = up_conv(in_channels=512,out_channels=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(in_channels=512, out_channels=256)
        
        self.Up3 = up_conv(in_channels=256,out_channels=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(in_channels=256, out_channels=128)
        
        self.Up2 = up_conv(in_channels=128,out_channels=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(in_channels=128, out_channels=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2) # Add sigmoid for loss later 
        #print(d1.shape,d1.dtype,torch.max(d1),'====================== IN ATT_UNET')
        ww = nn.Sigmoid()
        d1 = ww(d1)        #print(d1.shape,'======================')
        return d1
#=======================================================================================
    
def getmem(use_cuda=use_cuda):
    if use_cuda:
            nvmlInit()
            h = nvmlDeviceGetHandleByIndex(0);    info = nvmlDeviceGetMemoryInfo(h)
            per = info.free/info.total*100
            mem = f'{per:.0f}% Free{info.free/s24:.0f}MB (out of {info.total/s24:.0f})'
            mem0 = f'Free {info.free/s24:.0f}MB ({per:.0f}%)'
    else:
            free = int(psutil.virtual_memory().total - psutil.virtual_memory().available)
            tot = int(psutil.virtual_memory().total)
            per = free/tot*100
            mem = f'{per:.0f}% Free{free/s24:.0f}MB (out of {tot/s24:.0f})'   
            mem0 = f'Free {free/s24:.0f}MB ({per:.0f}%)'
    return mem0,mem   


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, padding = 1):
        super().__init__()

        self.activation_fn = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = padding)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, padding = padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.activation_fn(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation_fn(x)
        return x
    
class down_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = conv_block(in_channels, out_channels)
        self.pool = nn.MaxPool2d((2,2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x,p

class up_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 2, stride = 2, padding = 0)
        self.conv = conv_block(out_channels+out_channels, out_channels)
    
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


# ## Segmentation
# 
# - Image segmentation is the task of partitioning an image, or identifying an object in an image
# - Particular value in medical imaging, highlighting objects of interest
# 
# <p align="center">
# <img src="DRIVE/training/images/21_training.png" width="200" title="Image." >
# </p>
# <p align="center">
# <img src="DRIVE/training/1st_manual/21_manual1.png" width="200" title="Image." >
# </p>
# 
# - The target image is a binary image, where pixels = 0 are background and pixels = 1 are foreground. We want the network to output a binary image replicating this.

# In[3]:


def get_paths(base_path='DRIVE/'):
    train_im_paths = []
    train_gt_paths = []
    test_im_paths = []
    test_gt_paths = []
    
    for i in range(21, 41):
        train_im_paths.append(base_path + 'training/images/%d_training.tif'%(i))
        train_gt_paths.append(base_path + 'training/1st_manual/%d_manual1.gif'%(i))

    for i in range(1, 21):
        test_im_paths.append(base_path + 'test/images/%d_test.tif'%(i))
        test_gt_paths.append(base_path + 'test/1st_manual/%d_manual1.gif'%(i))
        
    train_paths = [train_im_paths,train_gt_paths]
    test_paths = [test_im_paths,test_gt_paths]
    return train_paths, test_paths

def read_and_resize(im_paths, gt_paths, resize=(256, 256, 1)): 
    imgs = io.imread(im_paths, 1)
    gts = io.imread(gt_paths, 1) 

    imgs = transform.resize(imgs, resize)
    gts = transform.resize(gts, resize) 
        
    return imgs, gts


# In[4]:


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        # TODO
        # 1. Initialize file paths or a list of file names. 
        self.im_paths = paths[0]
        self.gt_paths = paths[1]
        self.preprocesses = T.Compose([
            T.Resize((256,256)),
            T.ToTensor(),
        ])
        pass
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        image = Image.open(self.im_paths[index])
        label = Image.open(self.gt_paths[index])
        # 2. Preprocess the data (e.g. torchvision.Transform).
        
        image = self.preprocesses(image)
        label = self.preprocesses(label)
        
        #Ensure gt is binary
        label[label>.5] = 1
        label[label<=.5]=0
        # 3. Return a data pair (e.g. image and label).
        return image, label
        
    def __len__(self):
        return len(self.im_paths)

    def forward(self, x,plot=False):

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        
        if plot:
            print('Input shape', x.shape)
            print('After layer 1', x1.shape)
            print('After layer 2', x2.shape)
            print('After layer 3', x3.shape)
            print('After layer 4', x4.shape)
            print('After layer 5', x5.shape)
            print('After layer 6 CNN', x6.shape,torch.max(x6),'=============out')

        return x6     

def train2(num_epochs, model, loaders, total_step):  
    model.train()       
    # Train the model --        total_step1 = len(loaders)
    time0 = time.time()
    print('\ttrain2  [attention] for %d epochs, %d sets of data  each with batch size = %d' % 
          (num_epochs,total_step,loaders.batch_size))    
    for epoch in range(num_epochs):

        print ('Epoch %2d/%d Loss =' % (epoch + 1, num_epochs),end= " ")  #        print( list(loaders)  )
        for i, (images, labels) in enumerate(loaders): #['train']
            # images is of size [batch_size, 28, 28]

            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y

            output = model(b_x)#[0]               print(i, b_y.shape, output.shape)
            #print('\n---------',i,b_y.shape, output.shape,b_y.dtype, output.dtype,torch.max(output))
            loss = loss_func(output, b_y)
            
            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()                # apply gradients             
            optimizer.step()                
            
            if (i+1) % 100 == 0 or (  (i+1) % 2 == 0 and total_step<=99 ):
                print(f' {loss.item():.2f}@{i+1:d}', end="")   
        timea = time.time() - time0; timen=timea/(epoch+1)*(num_epochs-1-epoch)           
        if (epoch % 50 == 0) or (epoch+1)==num_epochs:     
            mem,_=getmem() 
            print(f"/{total_step:d} {timen/60:.2f} mins to go [{mem}]")
        else:
            print(f"/{total_step:d} {timen/60:.2f} mins to go ", end='\r') 

def dice_coeff(pred, target):
    smooth = 1.0e-6
    num = pred.size(0)
    m1 = pred.view(num, -1).float()  # Flatten
    m2 = target.view(num, -1).float()  # Flatten
    intersection = (m1 * m2).sum().float()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)              
 
if __name__ == '__main__':  
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print('\nRunning:',  __file__.split('\\')[-1],' on', device, '=', platform.node())
    if os.path.isdir('DRIVE/') :
        train_paths, test_paths = get_paths()
        print('Image data from DIRVE/  read ...')
    else:
        print("Error as DIR = DIRVE is not there ..."); exit()
    
    print('\n',  __file__.split('\\')[-1],'Choice of common UNet and Local CNN for retina segmentation')
    custom_dataset = CustomDataset(train_paths)
    train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                            batch_size=4, 
                                            shuffle=True)
    c = 0
    for i,(a1,a2) in enumerate(train_loader):
        c+=1        #print(a1.shape)
    print(f'Total Data Sets = {c} and batch = {train_loader.batch_size}')

    im,label = custom_dataset[0]
    print(im.size(), label.size())

    learning_rate = 0.001;  s24=1014**2;     loss_func = nn.BCELoss()   
    num_epochs = 12
       
    model = AttU_Net() # Case 3 - UNET from githup with Upsapling attention
        
    total_para = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)  
    sfile = 'model_sav_seg_%s.h5' % (str(model).split('(')[0])
    if os.path.isfile( sfile ):
            temp = torch.load( sfile, map_location=device ) ########### Previous trained results loaded
            model.load_state_dict(temp['model_state_dict']) # Pick up Weights and Opt state
            optimizer.load_state_dict(temp['optimizer_state_dict'])
            print('\tPrevious trained results in [%s] loaded (%d more epos)' % (sfile,num_epochs), ' Total paras = %d' % total_para)
    else:
            print('\tNew training for %d epochs' % num_epochs, ' Total paras = %d' % total_para)

    train2(num_epochs, model, train_loader, c) 
        
    end_time = time.time()
    _,mem = getmem() 

    now = datetime.datetime.now()
    torch.save({'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, sfile) ####### Current trained results saved
    print('\tINFO: saved = %s' % sfile, mem,now.strftime("%Y-%m-%d %H:%M"))   
    print('\tTry to predict next ...')    

    test_dataset = CustomDataset(test_paths)
    train_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=4, 
                                                shuffle=True)
    dataset = test_dataset

    with torch.no_grad():
            figure = plt.figure(figsize=(10, 8))
            rows = 2
            t_loss = 0; c_sys=0; dis=0; f_sys=0
            for i in range(0, rows):
                sample_idx = torch.randint(len(dataset), size=(1,)).item()
                img, label = dataset[sample_idx]
                b_x = img.unsqueeze(0)
                output = model(b_x)
                b_y = output.squeeze(0)
                loss = loss_func(b_y, label);      dis0=dice_coeff(b_y, label)
                dis += np.double(dis0)
                
                mydice = c_dice(average='micro')
                dis1 = mydice(b_y.type(torch.int32), label.type(torch.int32))
                dis2 = f_dice(b_y, label.type(torch.int32))
                #print(i, 'coeff',dis0, 'system', dis1, 'loss', loss,'seg dice', dis2)
                t_loss += loss;   
                c_sys += np.double( dis1 )
                f_sys += np.double( dis2 )
                figure.add_subplot(rows, 3, 3*i + 1) #________________________
                plt.axis("off")
                plt.imshow(img.permute(1,2,0))
                plt.title(f"image: {str(model).split('(')[0]}")
                figure.add_subplot(rows, 3, 3*i + 2)
                plt.axis("off")
                plt.imshow(torch.squeeze(output), cmap="gray")
                plt.title(f"model output dice {c_sys/(i+1)*100:.2f} {f_sys/(i+1)*100:.2f}")
                figure.add_subplot(rows, 3, 3*i + 3)
                plt.axis("off")
                plt.imshow(torch.squeeze(label), cmap="gray")
                plt.title("ground truth") #____________________________________
        
            file=__file__.split('\\')[-1].split('.')[0]
            plt.savefig('Pred_for_%s_by_model_%s.jpg' % (file, str(model).split('(')[0]) )
            print('One figure saved',end="")
            plt.pause(4)
            print(f'[avg loss BCE={t_loss/rows:.2f}, C Dice={c_sys/rows*100:.2f}, S Dice={f_sys/rows*100:.2f}]',
                  mem,'Time used %d\n' % (start_time-end_time))