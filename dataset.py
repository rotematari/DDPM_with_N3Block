import random

import torch
import torchvision.transforms
from PIL import Image
from torch.utils.data import Dataset
import os


def get_transforms(img_size):
    return torchvision.transforms.Compose([
                torchvision.transforms.Resize(img_size, torchvision.transforms.functional.InterpolationMode.BICUBIC),
                torchvision.transforms.CenterCrop(img_size),
                # torchvision.transforms.RandomCrop(img_size),
                # torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor()
            ])

def is_img(I):
    extension = I.split('.')[-1]
    return (extension.lower() == 'png') or (extension.lower() == 'jpg') or (extension.lower() == 'jpeg') or \
           (extension.lower() == 'bmp') or (extension.lower() == 'webp')


class unlabeled_data(Dataset):
    def __init__(self, data_dir, image_size, T=None):
        super(unlabeled_data, self).__init__()

        folders = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        self.samples = []

        for f in folders:
            self.samples += [os.path.join(f, I) for I in os.listdir(f) if is_img(I)]

        if T is None:
            self.T = get_transforms(image_size)
        else:
            self.T = T

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        I = Image.open(self.samples[item]).convert("RGB")
        x0 = self.T(I)
        return x0

class sr_data(Dataset):
    def __init__(self, data_dir, crop_size, sf=2, T=None, mix_HF=False):
        super(sr_data, self).__init__()

        self.sf = sf
        self.mix_HF = mix_HF

        folders = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        self.samples = []

        for f in folders:
            self.samples += [os.path.join(f, I) for I in os.listdir(f) if is_img(I)]

        if T is None:
            self.T = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(crop_size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor()
            ])
        else:
            self.T = T
        if mix_HF:
            self.A = lambda x: torch.nn.functional.interpolate(x, scale_factor=1.0/sf, mode='bicubic', antialias=True, align_corners=False)
            self.AT = lambda x: torch.nn.functional.interpolate(x, scale_factor=sf, mode='bicubic')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        I = Image.open(self.samples[item]).convert("RGB")
        x = self.T(I)
        if self.mix_HF:
            x = self.details_mix(x)
        return x

    def details_mix(self, x):
        I2 = Image.open(self.samples[random.randint(0, self.__len__() - 1)]).convert("RGB")
        z = self.T(I2)
        gamma = random.random()
        x = x.unsqueeze(0)
        mix = gamma * x + (1-gamma) * z.unsqueeze(0)
        return (mix - self.AT(self.A(mix)) + self.AT(self.A(x)))[0]

















# import torch
# import math
# from PIL import Image
# from torch.utils.data import Dataset
# import os


# class unlabeled_data(Dataset):
#     def __init__(self, sig_size):
#         super(unlabeled_data, self).__init__()
#
#         f_min = 0
#         f_max = torch.pi/8
#         N_freq_in_signal = 10
#         fs = 2*torch.pi
#         N_samples_per_signal = sig_size
#         T = N_samples_per_signal/fs
#
#         N_training_signals = 16000   # number of training signals, each spectrum is bounded by f_max and it is sampled at 4*f_max
#         t_vec = torch.linspace(0,T-1/fs,N_samples_per_signal).reshape(1,1,N_samples_per_signal)
#         freq_gt = f_max * torch.rand((N_training_signals,N_freq_in_signal,1)) + f_min
#         phase_gt = 2*torch.pi * torch.rand((N_training_signals,N_freq_in_signal,1))
#         mag_gt = 0.9+0.1*torch.rand((N_training_signals,N_freq_in_signal,2))
#         temp_cos = mag_gt[:,:,0:1] * torch.cos(freq_gt*t_vec + phase_gt)
#         temp_sin = mag_gt[:,:,1:2] * torch.sin(freq_gt*t_vec + phase_gt)
#         X_clean = torch.sum(temp_cos + temp_sin, dim=1)
#         X_clean = X_clean - torch.mean(X_clean, dim=1, keepdim=True)
#         X_clean_norm = torch.sqrt(torch.sum(X_clean**2,dim=1))
#         self.samples = X_clean / X_clean_norm[:,None] * math.sqrt(N_samples_per_signal)
#         self.samples = (self.samples - self.samples.min()) / (self.samples.max() - self.samples.min())
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, item):
#         return self.samples[item:item+1]
#
#
#
# if __name__ == '__main__':
#     import numpy as np
#     import matplotlib.pyplot as plt
#     data = unlabeled_data(sig_size=256)
#     for i, x_ in enumerate(data.samples[:16]):
#         plt.plot(np.arange(0, 256), np.array(x_.cpu()))
#         plt.grid()
#         plt.savefig(os.path.join('/disk5/Shady/ResDenoise/examples/', f'generated_{i}.png'))
#         plt.close()