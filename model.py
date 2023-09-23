import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedCNNGRUAgePredictor_adult(nn.Module):
    def __init__(self):
        super().__init__()

        # 1D CNN layers
        self.conv1 = nn.Conv1d(12, 64, kernel_size=15, stride=1, padding=7)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=15, stride=1, padding=7)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=15, stride=1, padding=7)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=15, stride=1, padding=7)
        self.bn4 = nn.BatchNorm1d(512)

        # LSTM layer
        self.gru = nn.GRU(input_size=512, hidden_size=256, num_layers=4, batch_first=True, dropout=0.5)

        # Feature Network
        self.linear = nn.Linear(12*35, 128)

        # Fully connected layers
        self.fc1 = nn.Linear(256 + 2 + 128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x, gender, age_group):
        feature = x[:, :, 5000:]
        x = x[:, :, :5000]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool1d(x, 2)

        x = x.permute(0, 2, 1)
        _, h_n = self.gru(x)
        x = h_n[-1]

        feature = torch.log1p(torch.abs(feature))
        feature = feature.reshape(-1, 12 * 35)
        feature = F.relu(self.linear(feature))

        x = torch.cat([x, gender.unsqueeze(1), age_group.unsqueeze(1), feature], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class EnhancedCNNGRUAgePredictor_child(nn.Module):
    def __init__(self):
        super().__init__()

        # 1D CNN layers
        self.conv1 = nn.Conv1d(12, 64, kernel_size=15, stride=1, padding=7)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=15, stride=1, padding=7)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=15, stride=1, padding=7)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=15, stride=1, padding=7)
        self.bn4 = nn.BatchNorm1d(512)

        # LSTM layer
        self.gru = nn.GRU(input_size=512, hidden_size=256, num_layers=4, batch_first=True, dropout=0.5)

        self.linear = nn.Linear(12*35, 64)

        # Fully connected layers
        self.fc1 = nn.Linear(256 + 2 + 64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x, gender, age_group):
        feature = x[:, :, 5000:]
        x = x[:, :, :5000]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool1d(x, 2)

        x = x.permute(0, 2, 1)
        _, h_n = self.gru(x)
        x = h_n[-1]

        feature = torch.log1p(torch.abs(feature))
        feature = feature.reshape(-1, 12 * 35)
        feature = F.relu(self.linear(feature))


        x = torch.cat([x, gender.unsqueeze(1), age_group.unsqueeze(1), feature], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

