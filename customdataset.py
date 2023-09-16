import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from scipy.signal import butter, filtfilt

# # Custom Dataset
# class CustomDataset(Dataset):
#     def __init__(self, csv_path, numpy_folder):
#         self.df = pd.read_csv(csv_path)
#         self.numpy_folder = numpy_folder
#         self.b, self.a = butter(1, [0.05/250, 100/250], btype='band')
#
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, idx):
#         filename = self.df.iloc[idx]['FILENAME']
#         data = np.load(self.numpy_folder + '/' + filename + '.npy')
#         data = filtfilt(self.b, self.a, data)
#         data = data.reshape(12, 5000)
#         age = self.df.iloc[idx]['AGE']
#         if self.df.iloc[idx]['GENDER'] == 'MALE':
#             gender = 1
#         elif self.df.iloc[idx]['GENDER'] == 'FEMALE':
#             gender = 2
#         else:
#             gender = 0
#
#         return torch.tensor(data.copy(), dtype=torch.float32), torch.tensor(gender, dtype=torch.float32), torch.tensor(age, dtype=torch.float32)
#
#
# class InferenceDataset(Dataset):
#     def __init__(self, csv_path, numpy_folder):
#         self.df = pd.read_csv(csv_path)
#         self.numpy_folder = numpy_folder
#         self.b, self.a = butter(1, [0.05/250, 100/250], btype='band')
#
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, idx):
#         filename = self.df.iloc[idx]['FILENAME']
#         data = np.load(self.numpy_folder + '/' + filename + '.npy')
#         data = filtfilt(self.b, self.a, data)
#         data = data.reshape(12, 5000)
#         if self.df.iloc[idx]['GENDER'] == 'MALE':
#             gender = 1
#         elif self.df.iloc[idx]['GENDER'] == 'FEMALE':
#             gender = 2
#         else:
#             gender = 0
#
#         return torch.tensor(data.copy(), dtype=torch.float32), torch.tensor(gender, dtype=torch.float32)

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, csv_path, numpy_folder):
        self.df = pd.read_csv(csv_path)
        self.numpy_folder = numpy_folder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df.iloc[idx]['FILENAME']
        data = np.load(self.numpy_folder + '/' + filename + '.npy')
        data = data.reshape(12, 5000)
        age = self.df.iloc[idx]['AGE']
        if self.df.iloc[idx]['GENDER'] == 'MALE':
            gender = 1
        elif self.df.iloc[idx]['GENDER'] == 'FEMALE':
            gender = 2
        else:
            gender = 0

        return torch.tensor(data, dtype=torch.float32), torch.tensor(gender, dtype=torch.float32), torch.tensor(age, dtype=torch.float32)


# class InferenceDataset(Dataset):
#     def __init__(self, csv_path, numpy_folder):
#         self.df = pd.read_csv(csv_path)
#         self.numpy_folder = numpy_folder
#
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, idx):
#         filename = self.df.iloc[idx]['FILENAME']
#         data = np.load(self.numpy_folder + '/' + filename + '.npy')
#         data = data.reshape(12, 5000)
#         if self.df.iloc[idx]['GENDER'] == 'MALE':
#             gender = 1
#         elif self.df.iloc[idx]['GENDER'] == 'FEMALE':
#             gender = 2
#         else:
#             gender = 0
#
#         return torch.tensor(data, dtype=torch.float32), torch.tensor(gender, dtype=torch.float32)


class InferenceDataset(Dataset):
    def __init__(self, csv_path, numpy_folder, file_list=None):
        self.df = pd.read_csv(csv_path)

        # 파일 목록이 제공된 경우 해당 파일만 선택
        if file_list is not None:
            self.df = self.df[self.df['FILENAME'].isin([f.replace('.npy', '') for f in file_list])]

        self.numpy_folder = numpy_folder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df.iloc[idx]['FILENAME']
        data = np.load(self.numpy_folder + '/' + filename + '.npy')
        data = data.reshape(12, 5000)

        if self.df.iloc[idx]['GENDER'] == 'MALE':
            gender = 1
        elif self.df.iloc[idx]['GENDER'] == 'FEMALE':
            gender = 2
        else:
            gender = 0

        return torch.tensor(data, dtype=torch.float32), torch.tensor(gender, dtype=torch.float32)

