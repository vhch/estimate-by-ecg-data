import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from scipy.signal import butter, lfilter, freqz, iirnotch, find_peaks
import pywt


def min_max_scaling(data):
    normalized_data = np.zeros_like(data)
    for i in range(data.shape[0]):  # 각 lead에 대해서
        min_val = np.min(data[i])
        max_val = np.max(data[i])
        range_val = max_val - min_val

        if range_val == 0:  # 모든 값이 동일한 경우
            normalized_data[i] = data[i]  # 원본 데이터를 그대로 사용하거나 0으로 설정
        else:
            normalized_data[i] = (data[i] - min_val) / range_val
    return normalized_data


# Z-Score Normalization for 12 leads
def z_score_normalization(data):
    # Assuming data shape is (12, num_samples)
    normalized_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        normalized_data[i] = (data[i] - np.mean(data[i])) / np.std(data[i])
    return normalized_data


def extract_wavelet_features(ecg_lead):
    coeffs = pywt.wavedec(ecg_lead, 'db1', level=4)
    # Level 4 coefficients as feature
    return coeffs[4]


def extract_fourier_features(ecg_lead):
    f_transform = np.fft.fft(ecg_lead)
    # Magnitude of first N/2 elements (positive frequencies) as features
    magnitude = np.abs(f_transform)[:len(ecg_lead)//2]
    return magnitude


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def notch_filter(data, notch_freq, q, fs):
    nyq = 0.5 * fs
    freq = notch_freq / nyq
    b, a = iirnotch(freq, q)
    y = lfilter(b, a, data)
    return y


def filter_all_leads(data, fs):
    num_leads, num_samples = data.shape
    filtered_data = np.empty((num_leads, num_samples))

    for i in range(num_leads):
        # Apply bandpass filter
        ecg_bandpass = butter_bandpass_filter(data[i], lowcut, highcut, fs)
        # If you're in a region with 60Hz powerline interference, you can also apply a 60Hz notch filter
        ecg_notched = notch_filter(ecg_bandpass, 60.0, 30, fs)
        filtered_data[i] = ecg_notched

    return filtered_data


def find_rr_features(ecg_data, fs):
    # 가정: `ecg_data`는 shape (12, 5000)의 ECG 데이터
    leads = ecg_data.shape[0]

    # 결과를 저장하기 위한 리스트 초기화
    rr_means = []
    rr_stds = []

    for lead in range(leads):
        # 현재 채널의 ECG 데이터 선택
        ecg_channel = ecg_data[lead]
        
        # R-peak 검출
        peaks, _ = find_peaks(ecg_channel, distance=fs/2.5)  # fs는 샘플링 주파수입니다.
        
        # RR 간격 계산
        rr_intervals = np.diff(peaks) / fs  # 샘플의 인덱스를 초로 변환
        
        # RR 간격의 평균 및 표준편차 계산 후 리스트에 추가
        rr_means.append(np.mean(rr_intervals))
        rr_stds.append(np.std(rr_intervals))

    return np.array(rr_means), np.array(rr_stds)


fs = 500.0  # sampling frequency
lowcut = 0.5
highcut = 50.0



class CustomDataset(Dataset):
    def __init__(self, csv_path, numpy_folder):
        self.df = pd.read_csv(csv_path)
        self.numpy_folder = numpy_folder

        # self.df = self.df[self.df['AGE'] <= 90]
        self.df = self.df[self.df['AGE'] <= 85]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df.iloc[idx]['FILENAME']

        # Check if 'adult' or 'child' is in the filename
        if 'adult' in filename:
            age_group = 0  # Let's say 0 for adult
        elif 'child' in filename:
            age_group = 1  # And 1 for child
        else:
            age_group = 2  # 2 for others (if any)

        data = np.load(self.numpy_folder + '/' + filename + '.npy')
        data = data.reshape(12, 5000)

        data = filter_all_leads(data, fs)
        data = min_max_scaling(data)

        age = self.df.iloc[idx]['AGE']
        if self.df.iloc[idx]['GENDER'] == 'MALE':
            gender = 0
        elif self.df.iloc[idx]['GENDER'] == 'FEMALE':
            gender = 1
        else:
            gender = 2

        return torch.tensor(data.copy(), dtype=torch.float32), torch.tensor(gender, dtype=torch.float32), torch.tensor(age, dtype=torch.float32), torch.tensor(age_group, dtype=torch.float32)


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

        # Check if 'adult' or 'child' is in the filename
        if 'adult' in filename:
            age_group = 0  # Let's say 0 for adult
        elif 'child' in filename:
            age_group = 1  # And 1 for child
        else:
            age_group = 2  # 2 for others (if any)

        data = np.load(self.numpy_folder + '/' + filename + '.npy')
        data = data.reshape(12, 5000)

        data = filter_all_leads(data, fs)


        if self.df.iloc[idx]['GENDER'] == 'MALE':
            gender = 0
        elif self.df.iloc[idx]['GENDER'] == 'FEMALE':
            gender = 1
        else:
            gender = 2

        return torch.tensor(data.copy(), dtype=torch.float32), torch.tensor(gender, dtype=torch.float32), torch.tensor(age_group, dtype=torch.float32)
