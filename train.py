import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from torch.utils.data import DataLoader

from loss_CenterIR import CenterIR
from model import *
torch.use_deterministic_algorithms(True)

random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

class Model(object):
    def __init__(self, model=None, lr=0.0001, device=None, weight=None, k=None, loss_lambda=None):
        super(Model, self).__init__()
        self.model = model
        self.mse = nn.MSELoss()
        self.CenterIR = CenterIR(weight=weight, k=k)
        self.loss_lambda = loss_lambda
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.device = device

    def fit(self, trainloader=None, validloader=None, epochs=None):
        history = {"train_loss": [], "train_mse": [], "train_mae": [], "train_rmse": [], "train_r2": [],
                   "val_loss": [], "val_mse": [], "val_mae":[], "val_rmse":[], "val_r2":[]}
        for ep in range(1, epochs + 1):
            if (not (ep % 10)) or (ep == 1):
                #if not only_print_finish_ep_num:
                print(f"Epoch {ep}/{epochs}")

            step = 1
            self.model.train()    # Train mode
            for x_batch, y_batch in trainloader:
                x_batch, y_batch = x_batch.to(self.device, dtype=torch.float), y_batch.to(self.device)

                self.optimizer.zero_grad()
                pred_t, feature_t = self.model(x_batch)
                loss_t = (self.mse(pred_t, y_batch) + self.loss_lambda*self.CenterIR(feature_t, y_batch))
                loss_t.backward()
                self.optimizer.step()

                if (not (ep % 10)) or (ep == 1):
                    pbar = int(step * 30 / len(trainloader))
                    print("\r{}/{} [{}{}]".format(
                            step, len(trainloader), ">" * pbar, " " * (30 - pbar)),
                            end="")
                step += 1
            if (not (ep % 10)) or (ep == 1):
                print()
            loss, mse, mae, rmse, r2 = self.evaluate(trainloader)
            val_loss, val_mse, val_mae, val_rmse, val_r2 = self.evaluate(validloader)
            history["train_loss"] = np.append(history["train_loss"], loss)
            history["train_mse"] = np.append(history["train_mse"], mse)
            history["train_mae"] = np.append(history["train_mae"], mae)
            history["train_rmse"] = np.append(history["train_rmse"], r2)
            history["train_r2"] = np.append(history["train_r2"], mse)
            history["val_loss"] = np.append(history["val_loss"], val_loss)
            history["val_mse"] = np.append(history["val_mse"], val_mse)
            history["val_mae"] = np.append(history["val_mae"], val_mae)
            history["val_rmse"] = np.append(history["val_rmse"], val_rmse)
            history["val_r2"] = np.append(history["val_r2"], val_r2)

        return history

    def evaluate(self, dataloader):
        self.model.eval()    # Eval mode
        real=[]
        pred=[]
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(self.device, dtype=torch.float), y_batch.to(self.device)
                out, feature = self.model(x_batch)
                loss = (self.mse(out, y_batch) + self.loss_lambda*self.CenterIR(feature, y_batch))
                pred.append(out.view(-1).detach().cpu().numpy())
                real.append(y_batch.view(-1).cpu().numpy())

        real = np.concatenate(real).tolist()
        pred = np.concatenate(pred).tolist()

        mse = mean_squared_error(real, pred)
        mae = mean_absolute_error(real, pred)
        rmse = root_mean_squared_error(real, pred)
        r2=r2_score(real, pred)

        return (loss.item(), mse, mae, rmse, r2)

    def predict(self, dataset):
        dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
        prediction = []
        truth = []
        self.model.eval()
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(self.device, dtype=torch.float), y_batch.to(self.device)
            pred = (self.model(x_batch))[0].cpu()
            prediction = np.append(prediction, pred.view(-1).detach().numpy())#prediction = np.append(prediction, pred.argmax(dim=1).numpy())
            truth = np.append(truth, y_batch.cpu().numpy())
        return prediction, truth
