import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from customdataset import CustomDataset, InferenceDataset
from model import Model


# 모델 로드
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Model()
model.load_state_dict(torch.load('best_model_checkpoint.pth'))
model = model.to(device)
model.eval()


csv_path = 'dataset/submission.csv'
numpy_folder = 'dataset/valid/'
# 데이터셋 및 데이터로더 설정
infer_dataset = InferenceDataset(csv_path=csv_path, numpy_folder=numpy_folder)
infer_loader = DataLoader(infer_dataset, batch_size=32)

# 나이 추론
predicted_ages = []
with torch.no_grad():
    for data, gender in infer_loader:
        data, gender = data.to(device), gender.to(device)
        outputs = model(data)
        predicted_ages.extend(outputs.cpu().numpy().flatten())

# CSV 파일의 AGE 열에 추론된 나이 기록
df = pd.read_csv(csv_path)
df['AGE'] = predicted_ages

# 수정된 CSV 파일 저장
df.to_csv('result.csv', index=False)
