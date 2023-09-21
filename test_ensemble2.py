import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
from customdataset import CustomDataset, InferenceDataset
from model import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_folds = 10
child_checkpoints = [f'checkpoint/Cnntogru_child_85cut_batch128_1e-3_filter_zscorenorm_{i}.pth' for i in range(num_folds)]
adult_checkpoints = [f'checkpoint/Cnntogru_adult_85cut_batch128_1e-3_filter_zscorenorm_{i}.pth' for i in range(num_folds)]

csv_path = 'dataset/submission.csv'
numpy_folder = 'dataset/valid/'


# 나이 추론 함수
def infer_age(checkpoints, loader):
    predicted_ages = []

    # 모델을 저장할 임시 변수 초기화
    temp_model = CNNGRUAgePredictor2().to(device).half()

    with torch.no_grad():
        for data, gender, age_group in loader:
            data, gender, age_group = data.to(device).half(), gender.to(device).half(), age_group.to(device).half()

            fold_predictions = []

            for checkpoint_path in checkpoints:
                # 모델 상태 로드
                checkpoint = torch.load(checkpoint_path)
                temp_model.load_state_dict(checkpoint['model_state_dict'])
                temp_model.eval()

                outputs = temp_model(data, gender, age_group)
                fold_predictions.append(outputs.cpu().numpy().flatten())

            mean_predictions = np.mean(fold_predictions, axis=0)
            predicted_ages.extend(mean_predictions)

    return predicted_ages


# 파일 패턴에 따라 데이터 로드 및 추론
child_files = [f for f in os.listdir(numpy_folder) if f.startswith("ecg_child_")]
adult_files = [f for f in os.listdir(numpy_folder) if f.startswith("ecg_adult_")]

child_dataset = InferenceDataset(csv_path=csv_path, numpy_folder=numpy_folder, file_list=child_files)
adult_dataset = InferenceDataset(csv_path=csv_path, numpy_folder=numpy_folder, file_list=adult_files)

child_loader = DataLoader(child_dataset, batch_size=32)
adult_loader = DataLoader(adult_dataset, batch_size=32)

child_predicted_ages = infer_age(child_checkpoints, child_loader)
adult_predicted_ages = infer_age(adult_checkpoints, adult_loader)

predicted_ages = child_predicted_ages + adult_predicted_ages

df = pd.read_csv(csv_path)
df['AGE'] = predicted_ages
df.to_csv('submission.csv', index=False)



# import pandas as pd
# import torch
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
#
# from customdataset import CustomDataset, InferenceDataset
# from model import *
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# num_folds = 10
# checkpoints = [f'checkpoint/Cnntogru_concat_85cut_batch128_1e-3_filter_zscorenorm_move5-2_{i}.pth' for i in range(num_folds)]
#
# csv_path = 'dataset/submission.csv'
# numpy_folder = 'dataset/valid/'
#
# infer_dataset = InferenceDataset(csv_path=csv_path, numpy_folder=numpy_folder)
# infer_loader = DataLoader(infer_dataset, batch_size=32)
#
# predicted_ages_list = []
#
# # 모델을 저장할 임시 변수 초기화
# temp_model = CNNGRUAgePredictor().to(device).half()
#
# with torch.no_grad():
#     for batch_idx, (data, gender, age_group) in enumerate(infer_loader):
#         data, gender, age_group = data.to(device).half(), gender.to(device).half(), age_group.to(device).half()
#
#         fold_predictions = []
#
#         for checkpoint_path in checkpoints:
#             # 모델 상태 로드
#             checkpoint = torch.load(checkpoint_path)
#             temp_model.load_state_dict(checkpoint['model_state_dict'])
#             temp_model.eval()
#
#             outputs = temp_model(data, gender, age_group)
#             fold_predictions.append(outputs.cpu().numpy().flatten())
#         
#         mean_predictions = np.mean(fold_predictions, axis=0)
#         predicted_ages_list.extend(mean_predictions)
#
# df = pd.read_csv(csv_path)
# df['AGE'] = predicted_ages_list
# df.to_csv('submission.csv', index=False)
