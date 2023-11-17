import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as tfs
import glob
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image, make_grid
from zqypackage import read_image
import numpy as np


class SRSFGANDataset(Dataset):
    def __init__(self, root, mode="train"):
        self.lt0 = sorted(glob.glob(os.path.join(root,"%s/lt0_patch" % mode) + "/*.tif"))
        self.lt1 = sorted(glob.glob(os.path.join(root,"%s/lt1_patch" % mode) + "/*.tif"))
        self.mt1 = sorted(glob.glob(os.path.join(root,"%s/mt1_patch" % mode) + "/*.tif"))
        self.tfs = tfs.Compose([
            tfs.ToTensor(),

        ])

    def __len__(self):
        assert len(self.lt0)==len(self.lt1)
        return len(self.lt1)

    @staticmethod
    def transform(data):
        data[data < 0] = 0
        out = data.mul_(0.0001)
        return out

    def __getitem__(self, index):
        lt0,_,_,_ = read_image(self.lt0[index])
        lt1,_,_,_= read_image(self.lt1[index])
        mt1,_,_,_ = read_image(self.mt1[index])
        lt0s = torch.from_numpy(lt0.astype(np.float32))
        lt1s = torch.from_numpy(lt1.astype(np.float32))
        mt1s = torch.from_numpy(mt1.astype(np.float32))
        for i in range(len(lt0)):
            im0 = lt0s[i,:,:]
            lt0s[i,:,:] = self.transform(im0)
            im1 = lt1s[i,:,:]
            lt1s[i,:,:] = self.transform(im1)
            im3 = mt1s[i,:,:]
            mt1s[i,:,:] = self.transform(im3)
        return lt0s,mt1s,lt1s


if __name__ == '__main__':
    e = SRSFGANDataset(r"E:\traindata",mode='val')
    lt0s,mt1s,lt1s = e[0]
    print(type(lt0s))

    # plt.imshow(a)
    # plt.show()
s = torch.from_numpy(mt1.astype(np.float32))
        for i in range(len(lt0)):
            im0 = lt0s[i,:,:]
            lt0s[i,:,:] = self.transform(im0)
            im1 = lt1s[i,:,:]
            lt1s[i,:,:] = self.transform(im1)
            im3 = mt1s[i,:,:]
            mt1s[i,:,:] = self.transform(im3)
        return lt0s,mt1s,lt1s









if __name__ == '__main__':
    e = SRGANDataset(r"E:\traindata",mode='val')
    lt0s,mt1s,lt1s = e[0]
    print(type(lt0s))

    # plt.imshow(a)
    # plt.show()
