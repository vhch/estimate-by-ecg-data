import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from customdataset import CustomDataset, InferenceDataset
from model import *


# 모델 로드
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNNGRUAgePredictor()
checkpoint = torch.load('checkpoint/Cnntogru_concat_85cut_batch128_1e-3.pth')
model.load_state_dict(checkpoint['model_state_dict'])
# model.load_state_dict(torch.load('best_model_checkpoint.pth'))
model = model.to(device).half()
model.eval()


csv_path = 'dataset/submission.csv'
numpy_folder = 'dataset/valid/'
# 데이터셋 및 데이터로더 설정
infer_dataset = InferenceDataset(csv_path=csv_path, numpy_folder=numpy_folder)
infer_loader = DataLoader(infer_dataset, batch_size=32)

# 나이 추론
predicted_ages = []
with torch.no_grad():
    for data, gender, age_group in infer_loader:
        data, gender, age_group = data.to(device).half(), gender.to(device).half(), age_group.to(device).half()
        outputs = model(data, gender, age_group)
        predicted_ages.extend(outputs.cpu().numpy().flatten())

# CSV 파일의 AGE 열에 추론된 나이 기록
df = pd.read_csv(csv_path)
df['AGE'] = predicted_ages

# 수정된 CSV 파일 저장
df.to_csv('submission.csv', index=False)
