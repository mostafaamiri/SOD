from tqdm import tqdm

def evaluate(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    iter = 0
    for X, y in tqdm(dataloader):
        X = X.cuda()
        y = y.cuda()
        pred = model(X)
        loss = loss_fn(pred, y)
        total_loss = (total_loss * iter + loss.detach().item())/(iter+1)
        iter+=1
    print("test loss is : {}".format(total_loss))
    return total_loss