from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from vedo import *
from scipy.spatial import distance_matrix



import torch.utils.data as data
from PIL import Image
import h5py
import numpy as np
import torch

def is_hdf5_file(filename):
    return filename.lower().endswith('.h5')


def get_keys(hdf5_path):
    with h5py.File(hdf5_path, 'r') as file:
        return list(file.keys())


def h5_loader(data, opt=None):
    ct_data = data['CT']
    fib_score = data['Fibrosis'][()]
    nas_stea_score = data['Steatosis'][()]
    nas_lob_score = data['Lobular'][()]
    nas_balloon_score = data['Ballooning'][()]

    ct_imgs = []
    for key in ct_data.keys():
        img = ct_data[key][()]
        if opt is not None and opt.model['use_resnet'] == 1:
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
                img = np.repeat(img, 3, axis=2)
            ct_imgs.append(Image.fromarray(img.astype('uint8'), 'RGB'))
        else:
            ct_imgs.append(Image.fromarray(img.astype('uint8'), 'L'))

    if fib_score == 0:  # 0: 0
        fib_label = 0
    elif fib_score < 3:  # 1: [1, 2, 2.5]
        fib_label = 1
    else:               # 2: [3, 3.5, 4]
        fib_label = 2

    nas_stea_label = 0 if nas_stea_score < 2 else 1
    nas_lob_label = nas_lob_score if nas_lob_score < 2 else 2
    # nas_lob_label = 0 if nas_lob_score < 2 else 1
    nas_balloon_label = nas_balloon_score

    return ct_imgs, fib_label, nas_stea_label, nas_lob_label, nas_balloon_label


class LiverDataset(data.Dataset):
    def __init__(self, hdf5_path, num_classes=15, patch_size=7000):
        super(LiverDataset, self).__init__()
        self.hdf5_path = hdf5_path
        self.keys = get_keys(self.hdf5_path)

    def __getitem__(self, index):
        #print(self.keys[index])
        hdf5_file = h5py.File(self.hdf5_path, "r")
        slide_data = hdf5_file[self.keys[index]]
        ct_imgs, fib_label, nas_stea_label, nas_lob_label, nas_balloon_label = h5_loader(slide_data, self.opt)
        ct_tensor = []
        for i in range(len(ct_imgs)):
            ct_tensor.append(self.data_transform(ct_imgs[i]).unsqueeze(0))

        return torch.cat(ct_tensor, dim=0), \
               torch.tensor(fib_label).unsqueeze(0).long(), torch.tensor(nas_stea_label).unsqueeze(0).long(), \
               torch.tensor(nas_lob_label).unsqueeze(0).long(), torch.tensor(nas_balloon_label).unsqueeze(0).long()

    def __len__(self):
        return len(self.keys)

class Mesh_Dataset(Dataset):
    def __init__(self, data_list_path, num_classes=15, patch_size=7000):
        """
        Args:
            h5_path (string): Path to the txt file with h5 files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.hdf5_path = data_list_path
        self.keys = get_keys(self.hdf5_path)

        self.num_classes = num_classes
        self.patch_size = patch_size

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        hdf5_file = h5py.File(self.hdf5_path, "r")
        slide_data = hdf5_file[self.keys[idx]]

        X = slide_data['data'][()]
        Y = slide_data['label'][()]
        labels = Y

        # initialize batch of input and label
        X_train = np.zeros([self.patch_size, X.shape[1]], dtype='float32')
        Y_train = np.zeros([self.patch_size, Y.shape[1]], dtype='int32')
        S1 = np.zeros([self.patch_size, self.patch_size], dtype='float32')
        S2 = np.zeros([self.patch_size, self.patch_size], dtype='float32')

        # calculate number of valid cells (tooth instead of gingiva)
        positive_idx = np.argwhere(labels>0)[:, 0] #tooth idx
        negative_idx = np.argwhere(labels==0)[:, 0] # gingiva idx

        num_positive = len(positive_idx) # number of selected tooth cells

        if num_positive > self.patch_size: # all positive_idx in this patch
            positive_selected_idx = np.random.choice(positive_idx, size=self.patch_size, replace=False)
            selected_idx = positive_selected_idx
        else:   # patch contains all positive_idx and some negative_idx
            num_negative = self.patch_size - num_positive # number of selected gingiva cells
            positive_selected_idx = np.random.choice(positive_idx, size=num_positive, replace=False)
            negative_selected_idx = np.random.choice(negative_idx, size=num_negative, replace=False)
            selected_idx = np.concatenate((positive_selected_idx, negative_selected_idx))

        selected_idx = np.sort(selected_idx, axis=None)

        X_train[:] = X[selected_idx, :]
        Y_train[:] = Y[selected_idx, :]


        X_train = X_train.transpose(1, 0)
        Y_train = Y_train.transpose(1, 0)


        sample = {'cells': torch.from_numpy(X_train), 'labels': torch.from_numpy(Y_train)}
        return sample
