import torch
from torch.utils import data
import numpy as np
import tifffile


def read_data(folder_name):
    blur_data = np.zeros((5,128,128,128))
    gt_data = np.zeros((5,128,128,128))

    for i in range(1,6):
        x_nuc = tifffile.imread('data/'+folder_name+'/'+str(i)+'.tif', maxworkers=6)

        if (folder_name == 'nuc') or (folder_name == 'act'):
            x_gt = tifffile.imread('data/gt/0_'+str(i)+'.tif', maxworkers=6)
        else:
            x_gt = tifffile.imread('data/gt/'+str(i)+'.tif', maxworkers=6)

        blur_data[i-1] = x_nuc
        gt_data[i-1] = x_gt

    blur_data = (blur_data - np.min(blur_data))/(np.max(blur_data) - np.min(blur_data))
    gt_data = (gt_data - np.min(gt_data))/(np.max(gt_data) - np.min(gt_data))

    blur_data = torch.from_numpy(blur_data)
    gt_data = torch.from_numpy(gt_data)

    return blur_data, gt_data


class ImageLoader(data.Dataset):
    def __init__(self, gt, blur, depth):
        # gt and blur: torch tensors
        # depth: number of slices for forward pass

        self.crop_depth = depth
        self.crop_size = 64
        self.im_size = gt.shape[-1]

        self.gt = gt
        self.blur = blur

        self.len = 128

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        im_index = np.random.randint(0, self.gt.shape[0])

        z_index = np.random.randint(0, self.im_size - self.crop_depth)
        x_index = np.random.randint(0, self.im_size - self.crop_size)
        y_index = np.random.randint(0, self.im_size - self.crop_size)

        blur = self.blur[im_index][z_index:(z_index+self.crop_depth),x_index:(x_index+self.crop_size),y_index:(y_index+self.crop_size)]
        gt = self.gt[im_index][z_index:(z_index+self.crop_depth),x_index:(x_index+self.crop_size),y_index:(y_index+self.crop_size)]

        blur = torch.unsqueeze(blur, dim=0)
        gt = torch.unsqueeze(gt, dim=0)

        return blur, gt