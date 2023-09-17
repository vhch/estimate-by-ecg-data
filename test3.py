import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from customdataset import CustomDataset, InferenceDataset
from model import *


# 모델 로드 함수
def load_model(model_path):
    model = Cnntobert()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).half()
    model.eval()
    return model

def load_model_adult(model_path):
    model = Cnntobert_adult()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).half()
    model.eval()
    return model


# 모델 로드
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_child = load_model('checkpoint/Cnntobert_child8.pth')  # ecg_child_*를 위한 모델
model_adult = load_model('checkpoint/Cnntobert_adult2.pth')  # ecg_adult_*를 위한 모델


csv_path = 'dataset/submission.csv'
numpy_folder = 'dataset/valid/'


# 나이 추론 함수
def infer_age(model, loader):
    predicted_ages = []
    with torch.no_grad():
        for data, gender in loader:
            data, gender = data.to(device).half(), gender.to(device).half()
            outputs = model(data)
            predicted_ages.extend(outputs.cpu().numpy().flatten())
    return predicted_ages


def infer_age_adult(model, loader):
    predicted_ages = []
    with torch.no_grad():
        for data, gender in loader:
            data, gender = data.to(device).half(), gender.to(device).half()
            outputs = model(data)
            outputs = F.softmax(outputs, dim=1)
            _, outputs = torch.max(outputs, dim=1)
            predicted_ages.extend(outputs.cpu().numpy().flatten())
    return predicted_ages


# 파일 패턴에 따라 데이터 로드 및 추론
child_files = [f for f in os.listdir(numpy_folder) if f.startswith("ecg_child_")]
adult_files = [f for f in os.listdir(numpy_folder) if f.startswith("ecg_adult_")]

child_dataset = InferenceDataset(csv_path=csv_path, numpy_folder=numpy_folder, file_list=child_files)
adult_dataset = InferenceDataset(csv_path=csv_path, numpy_folder=numpy_folder, file_list=adult_files)

child_loader = DataLoader(child_dataset, batch_size=32)
adult_loader = DataLoader(adult_dataset, batch_size=32)

child_predicted_ages = infer_age(model_child, child_loader)
adult_predicted_ages = infer_age_adult(model_adult, adult_loader)

predicted_ages = child_predicted_ages + adult_predicted_ages

# CSV 파일의 AGE 열에 추론된 나이 기록
df = pd.read_csv(csv_path)
df['AGE'] = predicted_ages

# 수정된 CSV 파일 저장
df.to_csv('submission.csv', index=False)
