from model import BiStreamModel
from dataloader import MSRADataset
from deeplearning import train, evaluate
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_results
import sys, getopt, os
from sklearn.model_selection import train_test_split


argv = sys.argv[1:]
try:
    options, args = getopt.getopt(argv, "b:e:n:",
                               ["batch=",
                                "epochs=",
                                "number="])
except:
    print("Error")
epochs = 50000
num = 0
batch_size = 8
for name, value in options:
    if name in ['-b', '--batch']:
        batch_size = (int)(value)
    if name in ['-e', '--epochs']:
        epochs = (int)(value)
    if name in ['-n', '--number']:
        num = (int)(value)

# loading data
file_lists = []
dataset_path = "./dataset/MSRA10K_Imgs_GT/Imgs/"
for f in os.listdir(dataset_path):
    file_lists.append(dataset_path + f[:-4])

file_lists = np.random.choice(file_lists, num)
trainfiles, testfiles = train_test_split(file_lists, test_size=0.2)
trainfiles, evalfiles = train_test_split(trainfiles, test_size=0.2)

trainds = MSRADataset(trainfiles)
traindl = DataLoader(trainds, batch_size, shuffle= True)

evalds = MSRADataset(evalfiles)
evaldl = DataLoader(evalds, batch_size, shuffle= True)

testds = MSRADataset(testfiles)
testdl = DataLoader(testds, batch_size, shuffle= True)

# creating model
model = BiStreamModel()
model = model.cuda()

# definiation of optimizer and oss function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(np.array([2]*128*128)).cuda())

history = train(model, traindl, evaldl, loss_fn, optimizer, epochs, "./results/model.pth")

evaluate(model, testdl, loss_fn)

plt.plot(history["loss"], label="train")
plt.plot(history["eval_loss"], label="eval")
plt.legend()
plt.savefig("./results/epoch_loss.png")

# showing examples
for X, y in traindl:
    X = X.cuda()
    y = y.cuda()
    pred = model(X)
plot_results(X, y, pred, batch_size, "./results/pic_result.png")
