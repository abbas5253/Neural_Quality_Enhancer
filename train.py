import fire
import os
import matplotlib.pyplot as plt
from data_gn import *

def train(train_dir='./Train/Type1', epochs=10, batch_size=1):
    name_list = os.listdir(train_dir)
    train_loader = build_DataLoader(name_list, './Train/Type1', )
    
    #print(len(train_loader))
    for _ in train_loader:
        print(_['low'].shape, _['high'].shape)
        
        print_images(_)
        
        
        
        
def print_images(sample):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    ax = axes.ravel()
    
    lowq = sample['low'].numpy()[0].transpose((1,2,0))
    highq = sample['high'].numpy()[0].transpose((1,2,0))
    
    
    
    ax[0].imshow(lowq)
    ax[1].imshow(highq)
    
    plt.show()
    



if __name__ == '__main__':
    fire.Fire(train)