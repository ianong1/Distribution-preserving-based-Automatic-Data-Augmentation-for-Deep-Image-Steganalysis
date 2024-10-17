import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import TES
from hpf import HPF
from ImageDataset import ImageDataset
import UNet

parser = argparse.ArgumentParser(description='AugmentationNet')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--init-learning-rate', type=float, default=1e-4,
                    help='Initial learning rate (default: 0.0001)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--amplitude', type=int, default=16, metavar='N',
                    help='Amplitude of noise (default: 16)')
parser.add_argument('--Enum', type=int, default=400, metavar='N',
                    help='Expectation of noise points (default: 400)')
parser.add_argument('--train-size', type=int, default=4000, metavar='N',
                    help='number of images to train (default: 4000)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--data-path', type=str, default = '/data-x/g15/zhangjiansong/cover',
                    help='Path of the training set')
parser.add_argument('--checkpoints-path', type=str, default = 'result',
                    help='Path of the checkpoints')
args = parser.parse_args()
kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
random.seed(2020)
                    
train_num = args.train_size
image_perm = random.sample(range(train_num), k=train_num)
image_train = image_perm[0:train_num]

data_path = args.data_path
lr = args.init_learning_rate
iter_num = args.epochs


class ToTensor():
  def __call__(self, sample):
    data = sample['data']
    data = data.astype(np.float32)
    new_sample = {
      'data': torch.from_numpy(data).unsqueeze(0),
    }
    return new_sample


train_transform = transforms.Compose([
ToTensor()
])

train_loader = torch.utils.data.DataLoader(
    ImageDataset(data_path, image_train, transform=train_transform),
    batch_size=args.batch_size, shuffle=False, **kwargs, drop_last=True)



def initWeights(module):
    if type(module) == nn.Conv2d:
        if module.weight.requires_grad:
            nn.init.normal_(module.weight.data, mean=0, std=0.02)

    if type(module) == nn.Linear:
        nn.init.normal_(module.weight.data, mean=0, std=0.02)
        nn.init.constant_(module.bias.data, val=0)



generator = UNet.UNet()
generator = nn.DataParallel(generator)
generator = generator.cuda()
generator.apply(initWeights)

def loss_noise(noise):                     #loss2
    loss = abs(noise).sum()/args.batch_size
    return abs(loss/args.amplitude-args.Enum)

# setup optimizer
optimizer = optim.Adam(generator.parameters(), lr)
critirion = nn.L1Loss()


hpf = HPF().cuda()
tes = TES.TES().cuda()

if not os.path.exists(args.checkpoints_path):
    os.makedirs(args.checkpoints_path)

for epoch in range(iter_num):
    for batch_idx, sample in enumerate(train_loader):
        cover = sample['data']
        cover = cover.cuda()     
        p = generator(cover)
        p1 = p/2
        m1 = p/2
        noise = tes(p1,m1)*args.amplitude                    
        new_cover = noise + cover
        
        residual = critirion(hpf(cover),hpf(new_cover))    #loss1
        loss1 = residual.cuda()
        loss2 = loss_noise(noise)
        if epoch == 0:
            loss = loss1
        else:
            loss = loss1 + 1.0 * loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print(epoch,'-',batch_idx,'residual:',loss1.item(),'number of nosie:',(abs(noise).sum()).item()/(args.batch_size*args.amplitude))
    # do checkpointing
    torch.save(generator.state_dict(), '%s/msePronetG_epoch_%d.pth' % (args.checkpoints_path,epoch))

