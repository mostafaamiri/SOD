from tqdm import tqdm
import torch
from torch import nn

def evaluate(model, dataloader, loss_fn1, loss_fn2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    iter = 0
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn1(pred, y)
        test_loss = (test_loss * iter + loss.detach().item())/(iter+1)
        iter += 1
    print("test BCE loss is : {}".format(test_loss))
    
    test_loss = 0
    iter = 0
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn2((nn.functional.sigmoid(pred)>0.7).float(), y)
        test_loss = (test_loss * iter + loss.detach().item())/(iter+1)
        iter += 1
    print("test MAE loss is : {}".format(test_loss))
