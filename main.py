from model import BiStreamModel
from dataloader import MSRADataset
from deeplearning import train, evaluate
from torch.utils.data import DataLoader
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_results, get_conf
import sys, getopt, os
from sklearn.model_selection import train_test_split


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf = get_conf("conf.yml")
# config
epochs = conf["epochs"]
num = conf["num"]
batch_size = conf["batch_size"]
path = conf["path"]

# dataset

file_lists = []
dataset_path = "./dataset/MSRA10K_Imgs_GT/Imgs/"
for f in os.listdir(dataset_path):
    file_lists.append(dataset_path + f[:-4])

file_lists = np.random.choice(file_lists, num)
trainfiles, testfiles = train_test_split(file_lists, test_size=0.15)
trainfiles, evalfiles = train_test_split(trainfiles, test_size=0.15)


trainds = MSRADataset(trainfiles)
traindl = DataLoader(trainds, batch_size, shuffle= True)

evalds = MSRADataset(evalfiles)
evaldl = DataLoader(evalds, batch_size, shuffle= True)

testds = MSRADataset(testfiles)
testdl = DataLoader(testds, batch_size, shuffle= True)

# creating model
model = BiStreamModel()
model = model.to(device)

# optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(np.array([2]*128*128)).to(device))

# model output before training
for Xn, yn in traindl:
    Xn = Xn.to(device)
    yn = yn.to(device)
    break
pred = model(Xn)
fig , ax = plt.subplots(nrows=batch_size, ncols=4, figsize=(10,10))
for i in range(batch_size):
    ax[i][0].imshow(yn[i].cpu().detach().numpy().reshape((128, 128)))
    ax[i][1].imshow((nn.functional.sigmoid(pred[i])>0.7).float().cpu().detach().numpy().reshape((128,128)))
    ax[i][2].imshow((nn.functional.sigmoid(pred[i])).float().cpu().detach().numpy().reshape((128,128)))
    ax[i][3].imshow(Xn[i].permute(1,2,0).cpu().detach().numpy())
    ax[i][0].axis('off')
    ax[i][1].axis('off')
    ax[i][2].axis('off')
    ax[i][3].axis('off')
ax[0][0].set_title("Ground Truth")
ax[0][1].set_title("pred with thrsh:0.7")
ax[0][2].set_title("Pred")
ax[0][3].set_title("Original pic")
plt.savefig(path+"/pic_"+str(0)+".png")

# training model
history = train(model, traindl, evaldl, loss_fn, optimizer, epochs, path, Xn, yn, batch_size)

plt.plot(history["loss"], label="loss")
plt.plot(history["eval_loss"], label="eval")
plt.legend()
plt.savefig(path+"/loss.png")

evaluate(model, testdl, loss_fn, torch.nn.L1Loss())
