#!/home/ewbell/miniforge3/envs/gpdiff/bin/python
upscaleFactor = 4

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from features import SingleImage
from network import MuyGP, NN
from torch.utils.data import DataLoader
from PIL import Image

def trainModel(loader, gp, device):
    gp.train()
    epoch = 0
    epochLoss = []
    gpopt = optim.AdamW(gp.parameters(), lr=1e-2, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(gpopt, patience=0, cooldown=4)
    
    while gpopt.param_groups[0]["lr"] > 1e-4 and epoch < 100000:
        print(gpopt.param_groups[0]["lr"])
        runningLoss = 0.
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            gpopt.zero_grad()
            output, var = gp(x)
            var = torch.clamp(var, min=1e-10)
            errors = (output - y) ** 2. / var.unsqueeze(1)
            loss = errors.sum() + y.size(1) * torch.log(var).sum()
            loss.backward()
            gpopt.step()
            runningLoss += loss.item()
        epochLoss.append(runningLoss)
        scheduler.step(runningLoss)
        epoch += 1
        print(epoch, epochLoss[-1])
        print(gp.a)
        print(gp.l)
    
def denoiseImage(img):
    batch = 8192
    loader = DataLoader(img, batch_size=batch, pin_memory=True)
    
    gp = MuyGP(2, 3).to(device)
    gp.trainX = img.x.to(device)
    gp.trainy = img.y.to(device)
    gp.ymean = gp.trainy.mean(dim=0, keepdim=True)
        
    #gp = NN(2, 1).to(device)
    trainModel(loader, gp, device)
    with torch.no_grad():
        gp.eval()
        hout = image.size(1) * upscaleFactor
        wout = image.size(2) * upscaleFactor
        yrange = torch.linspace(-1, 1, steps=hout)
        xrange = torch.linspace(-1, 1, steps=wout)
        pos = torch.cartesian_prod(yrange, xrange).to(device)
        chunks = torch.split(pos, batch, dim=0)
        us = []
        vs = []
        for chunk in chunks:
            u, v = gp(chunk)
            us.append(u)
            vs.append(v)
        upscale = torch.vstack(us)
        var = torch.cat(vs)
        upscale = np.clip((upscale.reshape(hout, wout, -1).cpu().numpy()+1)/2, a_min=0, a_max=1)
        var = var.view(hout, wout).cpu().numpy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow((image.permute(1,2,0)+1)/2)
        ax1.axis("off")
        ax2.imshow(upscale)
        ax2.axis("off")
        plt.tight_layout()
        plt.show()
        plt.imsave("/home/ewbell/denoise.png", upscale)
        
        plt.imshow(var)
        plt.colorbar()
        plt.show()

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])
    #imageset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    #imageset = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    #imageset = torchvision.datasets.SVHN(root="./data", split="train", download=True, transform=transform)
    
    '''
    for image, label in imageset:
        img = SingleImage(image)
        denoiseImage(img)
    '''
    
    from sys import argv
    image = Image.open(argv[1]).convert('RGB')
    image = transform(image)
    print(image.size())
    img = SingleImage(image)
    denoiseImage(img)
    
