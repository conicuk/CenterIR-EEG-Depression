import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from train import *
from loss_CenterIR import loss_weight_dict
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
torch.use_deterministic_algorithms(True)

random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

BATCH_SIZE = 32
Learning_Rate = 0.0001
EPOCHS = 100

boundaries = 15
k = [1, 3, 5, 15]
CenterIR_lambda = 5e-8
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# dummy data
input_np = np.random.randn(4000, 1, 19, 2500)
target_np = np.random.randn(4000,1)

mse, mae, rmse, r2 = [], [], [], []

kf = KFold(n_splits=10, shuffle=True, random_state=random_seed)

for fold, (train_idx, test_idx) in enumerate(kf.split(input_np, target_np), start=1):
    train_data = input_np[train_idx]
    test_data = input_np[test_idx]
    train_label = target_np[train_idx]
    test_label = target_np[test_idx]

    print("-------fold", fold)

    loss_weight = loss_weight_dict(train_label, boundaries)

    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_label = torch.tensor(train_label, dtype=torch.float32)

    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_label = torch.tensor(test_label, dtype=torch.float32)

    trainset = TensorDataset(train_data, train_label)
    testset = TensorDataset(test_data, test_label)
    trainloader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)
    testloader = DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)

    cnn_lstm = CNN_BiLSTM().to(device)

    model = Model(cnn_lstm, lr=Learning_Rate, device=device, weight=loss_weight, k=k, loss_lambda=CenterIR_lambda)
    history = model.fit(trainloader=trainloader, validloader=testloader, epochs=EPOCHS)
    best_epoch = np.argmin(history["val_mse"])
    print(f"val_MSE  : {history['val_mse'][best_epoch]}",
          f"val_MAE  : {history['val_mae'][best_epoch]}",
          f"val_RMSE : {history['val_rmse'][best_epoch]}",
          f"val_R2  : {history['val_r2'][best_epoch]}")

    mse.append(history['val_mse'][best_epoch])
    mae.append(history['val_mae'][best_epoch])
    rmse.append(history['val_rmse'][best_epoch])
    r2.append(history['val_r2'][best_epoch])

print("mean MSE :", np.mean(mse),np.std(mse))
print("mean MAE :", np.mean(mae),np.std(mae))
print("mean RMSE :", np.mean(rmse),np.std(rmse))
print("mean R2 :", np.mean(r2),np.std(r2))

