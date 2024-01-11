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
def transform_image(image):
    image = image.astype(np.float32)
    image = torch.from_numpy(image.copy())
    image = image/10000
    return image, image_mask

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
        images = [lt0,mt1,lt1]
        patches = [None] * len(images)

        for i in range(len(images)):
            im = images[i]
            im, _ = transform_image(im)
            patches[i] = im
        # gt_mask = masks[0] * masks[1] * masks[2] * masks[3]

        return patches[0], patches[1], patches[2]



