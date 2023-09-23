import argparse
import os
import pandas as pd
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from customdataset import CustomDataset, InferenceDataset
from model import *


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# 나이 추론 함수
def infer_age(checkpoints, loader, model, dataset):
    predicted_ages = []
    nan_files = []  # NaN 값을 가진 파일들을 저장할 리스트

    with torch.no_grad():
        for batch_idx, (data, gender, age_group) in enumerate(loader):
            data, gender, age_group = data.to(device), gender.to(device), age_group.to(device)
            fold_predictions = []

            for checkpoint_path in checkpoints:
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()

                outputs = model(data, gender, age_group)

                # NaN 값을 확인하고 그에 따라 특정 값을 설정
                if torch.isnan(outputs).any():
                    # print(f"Found NaN predictions in batch {batch_idx + 1}")

                    for idx, output in enumerate(outputs):
                        if torch.isnan(output):
                            file_idx = batch_idx * loader.batch_size + idx  # 전체 데이터셋에서의 인덱스
                            filename = dataset.file_list[file_idx]
                            nan_files.append(filename)  # NaN 파일 추가
                            if filename.startswith("ecg_child_"):
                                outputs[idx] = 0.083333333
                            else:
                                outputs[idx] = 66

                fold_predictions.append(outputs.cpu().numpy().flatten())

            mean_predictions = np.mean(fold_predictions, axis=0)
            predicted_ages.extend(mean_predictions)

    print(f"Files with NaN predictions: {nan_files[0]}")
    return predicted_ages




parser = argparse.ArgumentParser()
parser.add_argument('--csv_file', type=str, default='dataset/submission.csv', help="csv file path")
parser.add_argument('--numpy_folder', type=str, default='dataset/valid_feature/', help="numpy file directory")
parser.add_argument('--checkpoint_path', type=str, default="check", help="trained model path")
args = parser.parse_args()

chk_path = args.checkpoint_path
numpy_folder = args.numpy_folder
csv_path = args.csv_file

seed_everything(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_folds = 10
child_checkpoints = [f'{chk_path}/Cnntogru_child_100cut_batch128_4e-4_filter_zscorenorm_feature128_{i}.pth' for i in range(num_folds)]
adult_checkpoints = [f'{chk_path}/EnhancedCnntogru_adult_100cut_batch128_4e-4_filter_zscorenorm_feature128_{i}.pth' for i in range(num_folds)]


# 파일 패턴에 따라 데이터 로드 및 추론
child_files = [f for f in os.listdir(numpy_folder) if f.startswith("ecg_child_")]
# child_files.sort()
adult_files = [f for f in os.listdir(numpy_folder) if f.startswith("ecg_adult_")]
# adult_files.sort()

child_dataset = InferenceDataset(csv_path=csv_path, numpy_folder=numpy_folder, file_list=child_files)
adult_dataset = InferenceDataset(csv_path=csv_path, numpy_folder=numpy_folder, file_list=adult_files)

child_loader = DataLoader(child_dataset, batch_size=32)
adult_loader = DataLoader(adult_dataset, batch_size=32)

child_model = EnhancedCNNGRUAgePredictor_child().to(device)
adult_model = EnhancedCNNGRUAgePredictor_adult().to(device)

child_predicted_ages = infer_age(child_checkpoints, child_loader, child_model, child_dataset)
adult_predicted_ages = infer_age(adult_checkpoints, adult_loader, adult_model, adult_dataset)

predicted_ages = child_predicted_ages + adult_predicted_ages

df = pd.read_csv(csv_path)
df['AGE'] = predicted_ages
df.to_csv('./submission.csv', index=False)
