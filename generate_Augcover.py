import UNet
import torch
import cv2
import numpy as np
from collections import OrderedDict
import TES

tes = TES.TES().cuda()

def image2tensor(cover_path):
    image = cv2.imread(cover_path, -1)
    data = image.astype(np.float32)
    data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
    return data,image

generator = UNet.UNet().cuda()

pretrained_net_dict = torch.load('result/msePronetG_epoch_99.pth')
new_state_dict = OrderedDict()
for k, v in pretrained_net_dict.items():
    name = k[7:] # remove "module."
    new_state_dict[name] = v

generator.load_state_dict(new_state_dict)
generator.eval()


for i in range(1,10001):
    print(i)
    i = str(i)
    cover_path = '/data-x/g15/zhangjiansong/cover/' + i + '.pgm'                      # Path of the original cover
    Aug_cover_path = '/data-x/g15/zhangjiansong/Aug_cover/' + i + '.pgm'              # Path of the Augmented cover
    data,image = image2tensor(cover_path)
    y = generator(data.cuda())
    y = tes(y/2, y/2)
    y = y.reshape(256,256)
    y = y.detach().cpu().numpy()
    y = image + 16 * y                                                                # The amplitude of noise is set to 16
    y[y > 255] = 255
    y[y < 0] = 0
    y = np.uint8(y)
    cv2.imwrite(Aug_cover_path,y)
