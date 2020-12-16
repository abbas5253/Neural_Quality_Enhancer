import argparse
import os
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

def build_DataLoader(name_list, root_dir, transform=None):
    
    trainset = CustomDataset(name_list, root_dir, transform=transforms.Compose([
                                               ToTensor()
                                           ]))
    
    data_loader = DataLoader(trainset, batch_size=1)
    
    return data_loader
    


def convert_to_lowq(image):
    
    #scale to 0.25 then to 4x
    image = rescale(image, (0.75, 0.75, 1))
    image = rescale(image, (4/3,4/3,1))
    
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
#         landmarks = self.landmarks_frame.iloc[idx, 1:]
#         landmarks = np.array([landmarks])
#         landmarks = landmarks.astype('float').reshape(-1, 2)
        lowq_image = convert_to_lowq(image)
        sample = {'low': lowq_image, 'high':image}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        lowq, highq = sample['low'], sample['high']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        lowq = lowq.transpose((2, 0, 1))
        highq = highq.transpose((2, 0, 1))
        return {'low': torch.from_numpy(lowq),
                'high': torch.from_numpy(highq)}

    





    

