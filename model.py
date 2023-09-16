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


class Model2(nn.Module):
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


class CNN1D(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(CNN1D, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=out_channels, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(512)  # Avg pooling to make the sequence length 512
        )
    
    def forward(self, x):
        return self.cnn(x)

class CNNBERT(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.cnn = CNN1D(in_channels)

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
