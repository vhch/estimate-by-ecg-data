import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from customdataset import CustomDataset, InferenceDataset
from model import *

# 모델 로드 및 기타 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Stacked K-Fold 설정 (예: 5 folds)
num_folds = 10
checkpoints = [torch.load(f'checkpoint/Cnntogru_concat_85cut_batch128_1e-3_filter_zscorenorm_move5-2_{i}.pth') for i in range(num_folds)]
models = [CNNGRUAgePredictor().to(device).half() for _ in range(num_folds)]

for model, checkpoint in zip(models, checkpoints):
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

csv_path = 'dataset/submission.csv'
numpy_folder = 'dataset/valid/'

# 데이터셋 및 데이터로더 설정
infer_dataset = InferenceDataset(csv_path=csv_path, numpy_folder=numpy_folder)
infer_loader = DataLoader(infer_dataset, batch_size=32)

# 나이 추론
predicted_ages_list = []
with torch.no_grad():
    for batch_idx, (data, gender, age_group) in enumerate(infer_loader):
        data, gender, age_group = data.to(device).half(), gender.to(device).half(), age_group.to(device).half()

        # 각 모델의 예측값을 저장할 임시 리스트
        fold_predictions = []

        for model in models:
            outputs = model(data, gender, age_group)
            fold_predictions.append(outputs.cpu().numpy().flatten())

        # 모든 fold의 예측 평균 계산
        mean_predictions = np.mean(fold_predictions, axis=0)
        predicted_ages_list.extend(mean_predictions)


# CSV 파일의 AGE 열에 추론된 나이 기록
df = pd.read_csv(csv_path)
df['AGE'] = predicted_ages_list

# 수정된 CSV 파일 저장
df.to_csv('submission.csv', index=False)

