from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np



class MSRADataset(Dataset):
  def __init__(self, files):
    self.files = files

  def __len__(self):
    return len(self.files)

  def __getitem__(self, id):
    label = Image.open(self.files[id]+".png").resize((128,128))
    label = torch.tensor(np.array(label)).float()
    img = Image.open(self.files[id]+".jpg").resize((128,128))
    img = torch.tensor(np.array(img)/255.).float().permute(2,0,1)
    return img, (torch.flatten(label, start_dim=0)>0).float()