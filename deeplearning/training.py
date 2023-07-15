from tqdm import tqdm
import torch

def train(model, dataloader, evalloader, loss_fn, optimizer, epochs, path):
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
            if iter%100 == 0:
                print(epoch_loss)
        print("epoch loss is: {}".format(epoch_loss))
        history["loss"].append(epoch_loss)

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
    torch.save(model.state_dict(), path)
    return history
