import fire
import os
import matplotlib.pyplot as plt
from data_gn import *
from torch import optim
from model.model import *
from loss import ssim, ms_ssim, SSIM, MS_SSIM

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def train(train_dir='./data/Camera', epochs=50, batch_size=1):
    
    
    name_list = os.listdir(train_dir)[:1]
    
    train_loader = build_DataLoader(name_list, train_dir,)
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(device)
    print(len(train_loader))
    
    model = UNet(3)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-8)
    
    batch = next(iter(train_loader))
    model.train()
    batch['low'], batch['high'] = batch['low'].float().to(device), batch['high'].to(device)
    
    for i in range(20):
        pred = model(batch['low']).float()
        X = pred.float()
        Y = batch['high'].float()
        optimizer.zero_grad()
        ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=3)
        ms_ssim_loss = 1 - ms_ssim_module(X, Y)    
        ms_ssim_loss.backward()
        optimizer.step()
        
        print(ms_ssim_loss)

    
#     loss = 0;
#     for batch in train_loader:
        
#         #print_images(batch)
        
#         batch['low'], batch['high']  = batch['low'].float().to(device), batch['high'].to(device)
        
#         model_output = model.forward(batch['low'])
#         loss  =  loss_fn(model_output.view(-1), batch['high'].view(-1)).data
#         print(loss)

        
        
        
        
        
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
