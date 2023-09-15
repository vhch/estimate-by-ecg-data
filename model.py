import torch
import torch.nn as nn


# Model Definition
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5000 * 12, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.reshape(-1, 5000 * 12)
        return self.fc(x)


class Model2(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, num_series=12, output_size=1):
        super().__init__()

        self.lstms = nn.ModuleList([nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) for _ in range(num_series)])
        self.fc = nn.Linear(hidden_size * num_series, output_size)

    def forward(self, x):
        outputs = []
        x = x.unsqueeze(-1)
        for i, lstm in enumerate(self.lstms):
            out, _ = lstm(x[:, i, :, :])
            outputs.append(out[:, -1, :])

        outputs_combined = torch.cat(outputs, dim=1)
        final_output = self.fc(outputs_combined)
        return final_output
