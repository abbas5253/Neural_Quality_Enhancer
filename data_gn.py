import argparse
import os
import random
from skimage import io
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
import warnings

warnings.filterwarnings("ignore")

def build_DataLoader(name_list, root_dir, batch_size=4, transform=None):
    
    trainset = CustomDataset(name_list, root_dir, transform=transforms.Compose([
                                               ToTensor()
                                           ]))
    
    data_loader = DataLoader(trainset, batch_size=batch_size)
    
    return data_loader

def convert_to_lowq(image, scale_range=0.70):
    
    """
    Args:
        image : image to be converted into low quality
        scale_range : lower range upto which image can be rescaled
        random : random scaling range

    """
        
    image = rescale(image, (scale_range, scale_range, 1))
    image = rescale(image, (1/scale_range, 1/scale_range ,1))
    
    return image


    


class CustomDataset(Dataset):
    """Low Quality to High Quality Dataset."""

    def __init__(self, name_list, root_dir, transform=None):
        """
        Args:
            name_list (string): List of names of image in directory [ eg: os.listdir(train_dir)].
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.name_list  = name_list
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.name_list[0]) 
        image = io.imread(img_name)

        lowq_image = convert_to_lowq(image)
        
        sample = {'low': lowq_image, 'high':image}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        lowq, highq = sample['low'], sample['high']
        
        lowq = resize(lowq, (224,224,3))
        highq = resize(highq, (224,224,3))

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        lowq = lowq.transpose((2, 0, 1))
        highq = highq.transpose((2, 0, 1))
        return {'low': torch.from_numpy(lowq),
                'high': torch.from_numpy(highq)}
    
    
    
    

    


    





    

