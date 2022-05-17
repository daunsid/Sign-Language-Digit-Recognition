import os
import PIL
import path
#lllld
import numpy as np
import matplotlib.pyplot as plt

#import torch
#import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

sld_path = path.Path("../data")

transform = transforms.Compose([
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Resize(size=(128,128)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def normalize():
    mean=0
    std=0
    nb_samples=0
    for data in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    
    return transforms.Normalize(mean, std)


class sign_lang(Dataset):
    def __init__(self, path, transforms= None):
        self.transforms = transforms
        self.path = path
        self.data={cl:([(PIL.Image.open(self.path/cl/im),
                         int(cl)) for im in os.listdir(self.path/cl)]) 
                   for cl in os.listdir(self.path)}
        self.samp=[]
        for i in self.data:
            for e in self.data[i]:
                self.samp.append(e)
        self.targ = [self.samp[i][1] for i in range(len(self.samp))]
        
    def __len__(self):
        return len(self.samp)
    
    def targets(self):
        return self.targ
    
    def samples(self):
        return self.samp
    
    def __getitem__(self, index):
        sample = self.samp[index][0], self.samp[index][1]
        
        if self.transforms:    
            sample = self.transforms(sample[0]), sample[1]
        return sample
        
    
    def __repr__(self):
        return f"{__class__.__name__}:\nNum of datapoints: {len(self.samp)}\nRoot location: {self.path}"
    
    
def mkDataset():
    image_datasets = {x: sign_lang(sld_path/x, transform)
                      for x in ['train', 'valid', 'tests']}
    return image_datasets

def loadData():
    loaders = {x+'_loader': DataLoader(mkDataset()[x], batch_size=4,
                                              shuffle=False, num_workers=0)
                      for x in ['train', 'valid', 'tests']}
    return loaders

def input_layer():
    
    input_shape = next(iter(loadData()["train_loader"]))[0].shape
    return input_shape