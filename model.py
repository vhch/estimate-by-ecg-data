import torch
import torch.nn as nn
import torch.nn.functional as F
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


#######################################################################################################################################
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


class CNNtoB(nn.Module):
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
#######################################################################################################################################


class Cnntobert_adult(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=12, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # BERT model
        self.config = BertConfig(
            hidden_size=256,
            num_hidden_layers=8,
            num_attention_heads=4,
            intermediate_size=1024
        )
        self.bert = BertModel(self.config)
        self.dropout = nn.Dropout(0.1)

        # Fully connected layer after BERT
        # BERT base has an output size of 768
        self.fc = nn.Linear(in_features=self.config.hidden_size, out_features=104)

    def forward(self, x):
        # CNN
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv5(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten the CNN output to have a sequence length compatible with BERT
        # Here, we're assuming the sequence length is compatible with the BERT variant being used.
        # Adjust the reshaping as required.
        # x = x.permute(0, 2, 1).flatten(1, -2)
        x = x.permute(0, 2, 1)

        # BERT expects input of shape (batch, seq_len, feature_dim)
        outputs = self.bert(inputs_embeds=x)
        x = outputs['last_hidden_state'][:, 0, :]  # CLS token
        # x = self.dropout(x)

        # Fully connected layer
        x = self.fc(x)

        return x

# class Cnntobert(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.conv1 = nn.Conv1d(in_channels=12, out_channels=16, kernel_size=5, stride=1, padding=2)
#         self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
#         self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
#         self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
#         self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
#
#         self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.relu = nn.ReLU()
#
#         # BERT model
#         self.config = BertConfig(
#             hidden_size=256,
#             num_hidden_layers=8,
#             num_attention_heads=4,
#             intermediate_size=1024
#         )
#         self.bert = BertModel(self.config)
#         self.dropout = nn.Dropout(0.1)
#
#         # Fully connected layer after BERT
#         # BERT base has an output size of 768
#         self.fc = nn.Linear(in_features=self.config.hidden_size, out_features=1)
#
#     def forward(self, x):
#         # CNN
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.pool(x)
#
#         x = self.conv2(x)
#         x = self.relu(x)
#         x = self.pool(x)
#
#         x = self.conv3(x)
#         x = self.relu(x)
#         x = self.pool(x)
#         
#         x = self.conv4(x)
#         x = self.relu(x)
#         x = self.pool(x)
#         
#         x = self.conv5(x)
#         x = self.relu(x)
#         x = self.pool(x)
#
#         # Flatten the CNN output to have a sequence length compatible with BERT
#         # Here, we're assuming the sequence length is compatible with the BERT variant being used.
#         # Adjust the reshaping as required.
#         # x = x.permute(0, 2, 1).flatten(1, -2)
#         x = x.permute(0, 2, 1)
#
#         # BERT expects input of shape (batch, seq_len, feature_dim)
#         outputs = self.bert(inputs_embeds=x)
#         x = outputs['last_hidden_state'][:, 0, :]  # CLS token
#         x = self.dropout(x)
#
#         # Fully connected layer
#         x = self.fc(x)
#
#         return x

# class Cnntobert2(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.conv1 = nn.Conv1d(in_channels=12, out_channels=16, kernel_size=5, stride=1, padding=2)
#         self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
#         self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
#         self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
#         self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
#         self.conv6 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=5, stride=1, padding=2)
#
#         self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.relu = nn.ReLU()
#
#         # BERT model
#         self.config = BertConfig(
#             hidden_size=512,
#             num_hidden_layers=12,
#             num_attention_heads=8,
#             intermediate_size=2048
#         )
#         self.bert = BertModel(self.config)
#         self.dropout = nn.Dropout(0.1)
#
#         # Fully connected layer after BERT
#         # BERT base has an output size of 768
#         self.fc = nn.Linear(in_features=self.config.hidden_size, out_features=1)
#
#     def forward(self, x):
#         # CNN
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.pool(x)
#
#         x = self.conv2(x)
#         x = self.relu(x)
#         x = self.pool(x)
#
#         x = self.conv3(x)
#         x = self.relu(x)
#         x = self.pool(x)
#
#         x = self.conv4(x)
#         x = self.relu(x)
#         x = self.pool(x)
#
#         x = self.conv5(x)
#         x = self.relu(x)
#         x = self.pool(x)
#
#         x = self.conv6(x)
#         x = self.relu(x)
#         x = self.pool(x)
#
#         # Flatten the CNN output to have a sequence length compatible with BERT
#         # Here, we're assuming the sequence length is compatible with the BERT variant being used.
#         # Adjust the reshaping as required.
#         # x = x.permute(0, 2, 1).flatten(1, -2)
#         x = x.permute(0, 2, 1)
#
#         # BERT expects input of shape (batch, seq_len, feature_dim)
#         outputs = self.bert(inputs_embeds=x)
#         x = outputs['last_hidden_state'][:, 0, :]  # CLS token
#         x = self.dropout(x)
#
#         # Fully connected layer
#         x = self.fc(x)
#
#         return x
        



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

###########################################################################################


class EnhancedCnntobert2(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.config = BertConfig(
            hidden_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            intermediate_size=2048,
            max_position_embeddings=1024
        )
        self.bert = BertModel(self.config)
        self.dropout = nn.Dropout(0.2)

        # self.fc1 = nn.Linear(512 + 2 + 24, 256)  # Added two more feature inputs: rr_means, rr_stds
        self.fc1 = nn.Linear(512 + 1, 256)  # Added two more feature inputs: rr_means, rr_stds
        self.fc2 = nn.Linear(256, 1)
        # self.fc3 = nn.Linear(128, 64)
        # self.fc4 = nn.Linear(64, 1)


    def forward(self, x, gender, age_group):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.layer5(x)

        x = x.permute(0, 2, 1)


        outputs = self.bert(inputs_embeds=x)
        x = outputs['last_hidden_state'][:, 0, :]  # CLS token
        # x = torch.cat([x, gender.unsqueeze(1), age_group.unsqueeze(1)], dim=1)
        x = torch.cat([x, gender.unsqueeze(1)], dim=1)

        # x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        # x = F.relu(self.fc3(x))
        # x = self.dropout(x)
        # x = self.fc4(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class Cnntobert2(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # BERT model
        self.config = BertConfig(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=1024,
            max_position_embeddings=1024
        )
        self.bert = BertModel(self.config)
        self.dropout = nn.Dropout(0.1)

        # Fully connected layers
        self.fc1 = nn.Linear(256 + 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x, gender, age_group):
        # CNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        # Flatten the CNN output to have a sequence length compatible with BERT
        # Here, we're assuming the sequence length is compatible with the BERT variant being used.
        # Adjust the reshaping as required.
        # x = x.permute(0, 2, 1).flatten(1, -2)
        x = x.permute(0, 2, 1)

        # Concatenate with gender

        # BERT expects input of shape (batch, seq_len, feature_dim)
        outputs = self.bert(inputs_embeds=x)
        x = outputs['last_hidden_state'][:, 0, :]  # CLS token
        # x = torch.cat([x, gender.unsqueeze(1), age_group.unsqueeze(1)], dim=1)
        x = torch.cat([x, gender.unsqueeze(1), age_group.unsqueeze(1)], dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class Cnntobert3(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # BERT model
        self.config = BertConfig(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=1024,
            max_position_embeddings=1024
        )
        self.bert = BertModel(self.config)
        self.dropout = nn.Dropout(0.1)

        # Fully connected layers
        self.fc1 = nn.Linear(256 + 1, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x, gender, age_group):
        # CNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        # Flatten the CNN output to have a sequence length compatible with BERT
        # Here, we're assuming the sequence length is compatible with the BERT variant being used.
        # Adjust the reshaping as required.
        # x = x.permute(0, 2, 1).flatten(1, -2)
        x = x.permute(0, 2, 1)

        # Concatenate with gender

        # BERT expects input of shape (batch, seq_len, feature_dim)
        outputs = self.bert(inputs_embeds=x)
        x = outputs['last_hidden_state'][:, 0, :]  # CLS token
        # x = torch.cat([x, gender.unsqueeze(1), age_group.unsqueeze(1)], dim=1)
        x = torch.cat([x, gender.unsqueeze(1)], dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class Cnntobert4(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # BERT model
        self.config = BertConfig(
            hidden_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            intermediate_size=2048,
            max_position_embeddings=1024
        )
        self.bert = BertModel(self.config)
        self.dropout = nn.Dropout(0.1)

        # Fully connected layers
        self.fc1 = nn.Linear(512 + 1, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x, gender, age_group):
        # CNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Flatten the CNN output to have a sequence length compatible with BERT
        # Here, we're assuming the sequence length is compatible with the BERT variant being used.
        # Adjust the reshaping as required.
        # x = x.permute(0, 2, 1).flatten(1, -2)
        x = x.permute(0, 2, 1)

        # Concatenate with gender

        # BERT expects input of shape (batch, seq_len, feature_dim)
        outputs = self.bert(inputs_embeds=x)
        x = outputs['last_hidden_state'][:, 0, :]  # CLS token
        x = torch.cat([x, gender.unsqueeze(1)], dim=1)
        # x = torch.cat([x, gender.unsqueeze(1), age_group.unsqueeze(1)], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Cnn1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.fc1 = nn.Linear(512 * 312 + 2, 1024)  # Assuming that after 4 maxpooling layers the sequence length is 312. Adjust if necessary.
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, gender, age_group):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = out.view(out.size(0), -1)

        out = torch.cat([out, gender.unsqueeze(1), age_group.unsqueeze(1)], dim=1)


        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out


class CNNGRUAgePredictor(nn.Module):
    def __init__(self):
        super().__init__()

        # 1D CNN layers
        self.conv1 = nn.Conv1d(12, 64, kernel_size=15, stride=1, padding=7)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=15, stride=1, padding=7)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=15, stride=1, padding=7)
        self.bn3 = nn.BatchNorm1d(128)

        # LSTM layer
        self.gru = nn.GRU(input_size=128, hidden_size=128, num_layers=2, batch_first=True, dropout=0.5)

        # Fully connected layers
        # self.fc1 = nn.Linear(128 + 2, 64)
        self.fc1 = nn.Linear(128 + 1, 64)
        self.fc2 = nn.Linear(64, 1)


    def forward(self, x, gender, age_group):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool1d(x, 2)

        # (batch_size, sequence_length, num_features)
        x = x.permute(0, 2, 1)

        # RNN layers expect the input in the form (batch_size, sequence_length, num_features)
        _, h_n = self.gru(x)

        # Only take the output from the final timetep
        x = h_n[-1]

        x = torch.cat([x, gender.unsqueeze(1)], dim=1)
        # x = torch.cat([x, gender.unsqueeze(1), age_group.unsqueeze(1), rr_means, rr_stds], dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class CNNGRUAgePredictor2(nn.Module):
    def __init__(self):
        super().__init__()

        # 1D CNN layers
        self.conv1 = nn.Conv1d(12, 64, kernel_size=15, stride=1, padding=7)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=15, stride=1, padding=7)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=15, stride=1, padding=7)
        self.bn3 = nn.BatchNorm1d(256)

        # LSTM layer
        self.gru = nn.GRU(input_size=256, hidden_size=128, num_layers=2, batch_first=True, dropout=0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(128 + 1, 64)
        self.fc2 = nn.Linear(64, 1)

        # self.linear = nn.Linear(12*35, 64)

    def forward(self, x, gender, age_group):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool1d(x, 2)

        # (batch_size, sequence_length, num_features)
        x = x.permute(0, 2, 1)

        # RNN layers expect the input in the form (batch_size, sequence_length, num_features)
        _, h_n = self.gru(x)

        # Only take the output from the final timetep
        x = h_n[-1]

        # features = features.reshape(-1, 12 * 35)
        # feature = F.relu(self.linear(features))

        x = torch.cat([x, gender.unsqueeze(1)], dim=1)
        # x = torch.cat([x, gender.unsqueeze(1), age_group.unsqueeze(1)], dim=1)
        # x = torch.cat([x, gender.unsqueeze(1), age_group.unsqueeze(1), rr_means, rr_stds], dim=1)
        # x = torch.cat([x, gender.unsqueeze(1), age_group.unsqueeze(1), feature], dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class EnhancedCNNGRUAgePredictor(nn.Module):
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

        # Fully connected layers
        self.fc1 = nn.Linear(256 + 1, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x, gender, age_group):
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

        x = torch.cat([x, gender.unsqueeze(1)], dim=1)
        # x = torch.cat([x, gender.unsqueeze(1)], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# class CNNGRUAgePredictor(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         # # 1D CNN layers
#         # self.conv1 = nn.Conv1d(12, 64, kernel_size=15, stride=1, padding=7)
#         # self.bn1 = nn.BatchNorm1d(64)
#         # self.conv2 = nn.Conv1d(64, 128, kernel_size=15, stride=1, padding=7)
#         # self.bn2 = nn.BatchNorm1d(128)
#         # self.conv3 = nn.Conv1d(128, 128, kernel_size=15, stride=1, padding=7)
#         # self.bn3 = nn.BatchNorm1d(128)
#
#         self.conv_initial = nn.Conv1d(12, 32, kernel_size=7, stride=1, padding=3)
#
#         self.block1 = ResidualBlock(32, 64)
#         self.block2 = ResidualBlock(64, 128)
#         self.block3 = ResidualBlock(128, 128)
#
#         # LSTM layer
#         self.gru = nn.GRU(input_size=128, hidden_size=64, num_layers=2, batch_first=True, dropout=0.5)
#
#         # Fully connected layers
#         self.fc1 = nn.Linear(64 + 2, 32)
#         self.fc2 = nn.Linear(32, 1)
#
#     def forward(self, x, gender, age_group):
#         x = self.conv_initial(x)
#         x = self.block1(x)
#         x = F.max_pool1d(x, 2)
#         x = self.block2(x)
#         x = F.max_pool1d(x, 2)
#         x = self.block3(x)
#         x = F.max_pool1d(x, 2)
#
#         # (batch_size, sequence_length, num_features)
#         x = x.permute(0, 2, 1)
#
#         # RNN layers expect the input in the form (batch_size, sequence_length, num_features)
#         _, h_n = self.gru(x)
#
#         # Only take the output from the final timetep
#         x = h_n[-1]
#
#         x = torch.cat([x, gender.unsqueeze(1), age_group.unsqueeze(1)], dim=1)
#
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#
#         return x


class EnhancedECGNet(nn.Module):
    def __init__(self):
        super(EnhancedECGNet, self).__init__()

        # 1D CNN layers
        self.conv1 = nn.Conv1d(12, 32, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=7, stride=1, padding=3)
        self.bn4 = nn.BatchNorm1d(256)

        # Gender will be concatenated, so +1
        self.fc1 = nn.Linear(256*312 + 2, 128)  # 5000 / 2 / 2 / 2 / 2 = 625
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x, gender, age_group):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool1d(x, 2)

        # Flatten the feature maps.
        x = x.view(x.size(0), -1)

        # Concatenate with gender
        x = torch.cat([x, gender.unsqueeze(1), age_group.unsqueeze(1)], dim=1)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, padding=3):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.residual_transform = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.conv1.in_channels != self.conv2.out_channels:
            residual = self.residual_transform(residual)
        out += residual
        out = F.relu(out)
        return out

class ECGResNet(nn.Module):
    def __init__(self):
        super(ECGResNet, self).__init__()

        self.conv_initial = nn.Conv1d(12, 64, kernel_size=7, stride=1, padding=3)

        self.block1 = ResidualBlock(64, 128)
        self.block2 = ResidualBlock(128, 256)
        self.block3 = ResidualBlock(256, 512)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(512 + 1, 256) # 512 + 1 (for gender)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x, gender, age_group):
        x = self.conv_initial(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)

        # Concatenate with gender
        # x = torch.cat([x, gender.unsqueeze(1), age_group.unsqueeze(1)], dim=1)
        x = torch.cat([x, gender.unsqueeze(1)], dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
