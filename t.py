import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader

from customdataset import CustomDataset, CustomDataset2
from model import *


# 모델 로드
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model = AttiaNetworkAge()
model = EnhancedCNNGRUAgePredictor2().to(device)
# input_shape = (12, 500 * 10)  # ECG 데이터 크기에 맞게 설정
# nb_classes = 1  # 나이 예측을 위한 출력 뉴런 수
# model = InceptionTime(input_shape, nb_classes).to(device)

# model_path = 'checkpoint/Attianosam.pth'
model_path = 'checkpoint/Cnntogru_sample2.pth'
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
# model.load_state_dict(torch.load('best_model_checkpoint.pth'))
model = model.to(device)
model.eval()


# 데이터셋 및 데이터로더 설정
infer_dataset = CustomDataset(data_path='data_test')
infer_loader = DataLoader(infer_dataset, batch_size=32)
criterion = nn.L1Loss()  # Mean Squared Error for regression

mae = []

# 나이 추론
predicted_ages = []
with torch.no_grad():
    for batch_idx, (data, gender, targets, age_group) in enumerate(tqdm(infer_loader)):
        data, gender, targets, age_group = data.to(device), gender.to(device), targets.to(device), age_group.to(device)
        output = model(data, gender, age_group)
        loss = criterion(output.reshape(-1), targets.reshape(-1))
        mae.append(loss.item())

print(f'test val : {sum(mae)/len(infer_loader)}')
