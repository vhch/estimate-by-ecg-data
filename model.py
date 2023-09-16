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
            num_attention_heads=8,
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
