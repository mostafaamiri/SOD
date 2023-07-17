from tqdm import tqdm
import torch
from torch import nn
import matplotlib.pyplot as plt

def train(model, dataloader, evalloader, loss_fn, optimizer, epochs, path, Xn, yn):

    history = {"loss":[], "eval_loss": []}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(epochs):
        model.train()
        print("epoch {}".format(epoch))
        epoch_loss = 0
        iter = 0
        for X, y in tqdm(dataloader):
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss = (epoch_loss * iter + loss.detach().item())/(iter+1)
            iter += 1
            # if iter%100 == 0:
            #     print(epoch_loss)
        print("epoch loss is: {}".format(epoch_loss))
        history["loss"].append(epoch_loss)
    
        # if epoch%10 == 0:
        torch.save(model.state_dict(), path+"/model.pth")
        print("model saved")
    
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
        plt.savefig(path+"/pic_"+str(epoch+1)+".png")
    
    
        model.eval()
        eval_loss = 0
        iter = 0
        for X, y in evalloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            eval_loss = (eval_loss * iter + loss.detach().item())/(iter+1)
            iter += 1
        print("eval loss is : {}".format(eval_loss))
        history["eval_loss"].append(eval_loss)
    return history
