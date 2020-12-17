import fire
import os
import matplotlib.pyplot as plt
from data_gn import *
from model.model import *

model = UNet(3)

def train(train_dir='./data/Type1', epochs=10, batch_size=1):
    name_list = os.listdir(train_dir)
    train_loader = build_DataLoader(name_list, train_dir, )
    loss = nn.MSELoss()
    
    #print(len(train_loader))
    for _ in train_loader:
        model_output = model.forward(_['low'].float())
        loss  = loss(model_output.view(-1), _['high'].view(-1))
        print(loss.shape)
        print(loss)
        
        
        
            
def print_images(sample):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    ax = axes.ravel()
    
    lowq = sample['low'].detach().numpy()[0].transpose((1,2,0))

    highq = sample['high'].detach().numpy()[0].transpose((1,2,0))
    
    """
    print difference in pixels
    """
    
    ax[0].imshow(lowq)
    ax[1].imshow(highq)
    
    plt.show()
    



if __name__ == '__main__':
    fire.Fire(train)