import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig


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


class Lstm(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_series=12, output_size=1):
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


class Lstm2(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_series=12, output_size=1):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * num_series, output_size)
        self.num_series = num_series

    def forward(self, x):
        outputs = []
        x = x.unsqueeze(-1)
        for i in range(self.num_series):
            out, _ = self.lstm(x[:, i, :, :])
            outputs.append(out[:, -1, :])

        outputs_combined = torch.cat(outputs, dim=1)
        final_output = self.fc(outputs_combined)
        return final_output


# Model with BERT
class BERTforECG(nn.Module):
    def __init__(self, bert_name="bert-base-uncased"):
        super().__init__()

        self.config = BertConfig(
            hidden_size=500,
            num_hidden_layers=6,
            num_attention_heads=10,
            intermediate_size=2048
        )
        self.bert = BertModel(self.config)

        self.fc = nn.Linear(self.config.hidden_size, 1)

    def forward(self, x):
        # BERT expects inputs in the shape (batch_size, seq_length, hidden_size)
        # Since we're not using any token type ids or attention masks, we can pass None for them
        outputs = self.bert(inputs_embeds=x, token_type_ids=None, attention_mask=None)

        # Only using the [CLS] token's output, denoting the aggregated representation
        pooled_output = outputs[1]
        final_output = self.fc(pooled_output)
        return final_output


class C(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=out_channels, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(512)  # Avg pooling to make the sequence length 512
        )
    
    def forward(self, x):
        return self.cnn(x)


class CNNtoBERT(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.cnn = C(in_channels)

        self.config = BertConfig(
            hidden_size=512,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=2048
        )
        self.bert = BertModel(self.config)

        self.fc = nn.Linear(self.config.hidden_size, 1)

    def forward(self, x):
        x = self.cnn(x)  # Pass through 1D CNN. Shape should now be (batch_num, 12, 512)
        
        outputs = self.bert(input_ids=None, inputs_embeds=x)
        pooled_output = outputs.pooler_output  # Taking pooled output. Change this depending on your requirement.
        
        return self.fc(pooled_output)

class Cnn1d(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # 추가적인 층을 더 쌓을 수 있습니다
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * (5000 // 4), 128),  # 2번의 MaxPool1d에 의해 길이가 1/4로 줄었다고 가정
            nn.ReLU(),
            nn.Linear(128, 1)  # Age 예측을 위한 출력. 회귀 문제로 간주.
        )
    
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


class CNNTOLSTM(nn.Module):
    def __init__(self, input_size=32, hidden_size=100, num_layers=1, dropout=0):
        super().__init__()

        # CNN layers
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # After the CNN layers, output channels = 32, and after pooling, seq_len = 1250
        # Hence, before passing to LSTM, we reshape it to have a seq_len of 1250 and feature_dim of 32

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # Fully connected layer
        self.fc = nn.Linear(in_features=100, out_features=1)

    def forward(self, x):
        # CNN
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # LSTM expects input of shape (batch, seq_len, input_size)
        # So, we reshape the tensor such that seq_len is 1250 and input_size is 32
        x = x.permute(0, 2, 1)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Take the output from the last time-step
        x = lstm_out[:, -1, :]

        # Fully connected layer
        x = self.fc(x)

        return x


class LSTMtoBERT(nn.Module):
    def __init__(self, input_size=1, hidden_size=256, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.config = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=1024
        )
        self.bert = BertModel(self.config)

        # Adjust the output dimension based on the BERT config hidden size to your task
        self.fc = nn.Linear(self.config.hidden_size, 1)  # Predicting age

    def forward(self, x):
        # x shape: (batch_size, 12, 5000, 1)
        x = x.unsqueeze(-1)
        batch_size, num_cases, seq_len, _ = x.size()

        lstm_outputs = []
        for i in range(num_cases):
            # Passing each case through the LSTM
            lstm_out, _ = self.lstm(x[:, i])
            # Getting the last hidden state of the LSTM
            lstm_outputs.append(lstm_out[:, -1, :])

        # Stacking all the LSTM outputs
        lstm_output = torch.stack(lstm_outputs, dim=1)  # shape: (batch_size, 12, hidden_dim)

        # Passing through BERT
        bert_output = self.bert(input_ids=None, inputs_embeds=lstm_output)['pooler_output']

        # Final fully connected layer
        out = self.fc(bert_output)

        return out
