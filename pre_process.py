import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from scipy.signal import butter, lfilter, freqz, iirnotch, find_peaks, medfilt
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


def z_score_normalization(data):
    normalized_data = np.zeros_like(data)
    for i in range(data.shape[0]):  # 각 lead에 대해서
        mean_val = np.mean(data[i])
        std_val = np.std(data[i])
        
        if std_val == 0:  # 모든 값이 동일한 경우
            # print(data)
            normalized_data[i] = np.zeros_like(data[i])  # 0으로 설정
        else:
            normalized_data[i] = (data[i] - mean_val) / std_val
    return normalized_data


def extract_wavelet_features(ecg_lead):
    leads = ecg_lead.shape[0]
    coeffs_list = []
    for lead in range(leads):
        ecg_channel = ecg_lead[lead]
        coeffs = pywt.wavedec(ecg_channel, 'db1', level=4)
        coeffs_list.append(coeffs[4])
    # Level 4 coefficients as feature
    return coeffs_list


def extract_fourier_features(ecg_lead):
    leads = ecg_lead.shape[0]
    magnitude_list = []
    for lead in range(leads):
        ecg_channel = ecg_lead[lead]
        f_transform = np.fft.fft(ecg_channel)
        # Magnitude of first N/2 elements (positive frequencies) as features
        magnitude = np.abs(f_transform)[:len(ecg_channel)//2]
        magnitude_list.append(magnitude)
    # Level 4 coefficients as feature
    return magnitude_list


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


def median_filter(data, window_size=5):
    """메디안 필터 적용 함수"""
    return medfilt(data, window_size)


def moving_average_filter(data, window_size=5):
    """이동 평균 필터 적용 함수"""
    # convolve 함수를 사용하여 이동 평균 계산
    # 'valid'는 원본 데이터와 같은 길이의 결과를 반환합니다.
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')


def wavelet_denoising(data):
    coeffs = pywt.wavedec(data, 'db1', level=6)
    coeffs[-1] = np.zeros_like(coeffs[-1])
    coeffs[-2] = np.zeros_like(coeffs[-2])
    reconstructed_data = pywt.waverec(coeffs, 'db1')
    return reconstructed_data

# def denoise_ecg_wavelet(ecg_data, wavelet='db6', level=4, threshold_type='hard'):
#     """
#     ECG 데이터에 대한 Wavelet 노이즈 제거 함수
#
#     :param ecg_data: (12, 5000) 형태의 ECG 데이터
#     :param wavelet: 사용할 wavelet 종류. 'db6' 추천
#     :param level: 분해 레벨
#     :param threshold_type: 임계값 처리 방법 ('soft' 또는 'hard')
#     :return: 노이즈가 제거된 ECG 데이터
#     """
#
#     # 각 lead에 대해 wavelet 분해
#     coeffs = pywt.wavedec(ecg_data, wavelet, level=level)
#
#     # Universal thresholding (Donoho's method)
#     sigma = (np.median(np.abs(coeffs[-1]))) / 0.6745
#     threshold = sigma * np.sqrt(2 * np.log(len(ecg_data)))
#
#     # Coefficients thresholding
#     for j in range(1, len(coeffs)):
#         coeffs[j] = pywt.threshold(coeffs[j], threshold, mode=threshold_type)
#
#     # Reconstructed signal
#     denoised_data = pywt.waverec(coeffs, wavelet)
#
#     return denoised_data


def filter_all_leads(data, fs):
    num_leads, num_samples = data.shape
    filtered_data = np.empty((num_leads, num_samples))

    for i in range(num_leads):
        # Apply bandpass filter
        ecg_bandpass = butter_bandpass_filter(data[i], lowcut, highcut, fs)
        # ecg_denoised = denoise_ecg_wavelet(ecg_bandpass)
        # ecg_denoised = wavelet_denoising(ecg_bandpass)

        # If you're in a region with 60Hz powerline interference, you can also apply a 60Hz notch filter
        # ecg_notched = notch_filter(ecg_denoised, 60.0, 30, fs)
        ecg_notched = notch_filter(ecg_bandpass, 60.0, 30, fs)

        # ecg_avg = moving_average_filter(ecg_notched)
        # ecg_median = median_filter(ecg_avg, window_size=3)

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

def process_and_save_npy_files(csv_path, numpy_folder, output_folder):
    # 입력 csv를 불러옵니다.
    df = pd.read_csv(csv_path)
    
    # 결과를 저장할 폴더가 존재하지 않는 경우 생성합니다.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 모든 파일에 대하여
    for idx in range(len(df)):
        filename = df.iloc[idx]['FILENAME']
        input_path = os.path.join(numpy_folder, filename + '.npy')
        output_path = os.path.join(output_folder, filename + '.npy')
        
        # .npy 파일을 불러옵니다.
        data = np.load(input_path)
        data = data.reshape(12, 5000)
        
        # 함수를 적용합니다.
        data = filter_all_leads(data, fs)
        data = z_score_normalization(data)

        rr_means, rr_stds = find_rr_features(data, fs)

        wave = extract_wavelet_features(data)
        fourier = extract_fourier_features(data)

        rr_means = rr_means.reshape(-1, 1)
        rr_stds = rr_means.reshape(-1, 1)
        # print(np.array(data).shape)
        # print(np.array(wave).shape)
        # print(np.array(fourier).shape)
        # print(np.array(rr_means).shape)
        # print(np.array(rr_stds).shape)
        data = np.hstack((data, rr_means, rr_stds, wave, fourier))

        # 결과를 .npy로 저장합니다.
        np.save(output_path, data)


data_dir = "dataset/data_filt_zscore_feature"

# 함수를 호출하여 작업을 실행합니다.
process_and_save_npy_files('dataset/ECG_adult_age_train.csv', 'dataset/adult/train', data_dir)
process_and_save_npy_files('dataset/ECG_child_age_train.csv', 'dataset/child/train', data_dir)

print(f"task end : {data_dir}")

