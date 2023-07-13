import matplotlib.pyplot as plt
from torch import nn

def plot_results(X, y, pred, batch_size, path):
    fig , ax = plt.subplots(nrows=batch_size, ncols=4, figsize=(10,20))
    for i in range(batch_size):
        ax[i][0].imshow(y[i].cpu().detach().numpy().reshape((128, 128)))
        ax[i][1].imshow((nn.functional.sigmoid(pred[i])>0.7).float().cpu().detach().numpy().reshape((128,128)))
        ax[i][2].imshow((nn.functional.sigmoid(pred[i])).float().cpu().detach().numpy().reshape((128,128)))
        ax[i][3].imshow(X[i].permute(1,2,0).cpu().detach().numpy())
        ax[i][0].axis('off')
        ax[i][1].axis('off')
        ax[i][2].axis('off')
        ax[i][3].axis('off')
    plt.savefig(path)