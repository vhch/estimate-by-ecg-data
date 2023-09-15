import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset


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


class InferenceDataset(Dataset):
    def __init__(self, csv_path, numpy_folder):
        self.df = pd.read_csv(csv_path)
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
