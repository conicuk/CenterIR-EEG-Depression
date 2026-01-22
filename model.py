import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import torch.nn as nn
torch.use_deterministic_algorithms(True)

random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

class CNN_BiLSTM(nn.Module):
    def __init__(self):
        super(CNN_BiLSTM, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (1, 16), bias=False)
        self.conv2 = nn.Conv2d(32, 32, (19, 1), bias=False)
        self.Bn1 = nn.BatchNorm2d(32)
        self.AvgPool1 = nn.AvgPool2d((1, 35), stride=(1, 7))
        self.Drop = nn.Dropout(0.3)

        self.lstm_hidden_dim = 32
        self.lstm_layers = 1
        self.bidirectional = True

        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.lstm_layers,
            batch_first=True,
            bidirectional=self.bidirectional
        )

        self.classifier = nn.Sequential(
            nn.Linear(22464, 512),
            nn.ReLU(),
            self.Drop,
            nn.Linear(512, 128),
            nn.ReLU(),
            self.Drop,
            nn.Linear(128, 1)
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.Bn1(x)
        x = x ** 2
        x = self.AvgPool1(x)
        x = torch.log(x)

        x = x.squeeze(2)
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)
        flattened = lstm_out.contiguous().view(x.size(0), -1)

        x = self.classifier(flattened)

        return x, flattened