import torch
from torch.utils.data import Dataset

class SingleImage(Dataset):
    def __init__(self, imgtensor):
        h = imgtensor.size(-2)
        w = imgtensor.size(-1)
        yrange = torch.linspace(-1, 1, steps=h)
        xrange = torch.linspace(-1, 1, steps=w)
        self.x = torch.cartesian_prod(yrange, xrange)
        self.y = imgtensor.permute(1,2,0).view(h*w,-1)
        
    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    
