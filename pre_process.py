import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from scipy.signal import butter, lfilter, freqz, iirnotch, find_peaks, medfilt
from biosppy.signals import ecg  # biosppy 라이브러리를 임포트
import pywt
import scipy
from sklearn.decomposition import PCA
import argparse


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
            normalized_data[i] = np.zeros_like(data[i]) # 0으로 설정
        else:
            normalized_data[i] = (data[i] - mean_val) / std_val
    return normalized_data


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



# Pan-Tompkins 알고리즘
def pan_tompkins_qrs(ecg, sampling_rate=500):
    if ecg.size == 0:
        print("ECG array is empty.")
    if ecg.size == 0:
        print("ECG array is empty.")
    # 차분
    differentiated = np.ediff1d(ecg)
    # 제곱
    squared = differentiated ** 2

    # 이동 평균
    integration_window = int(sampling_rate * 0.150)  # 150 ms window
    integrated = np.convolve(squared, np.ones(integration_window)/integration_window)

    # 피크 찾기
    r_peaks, _ = find_peaks(integrated, height=[0.6, 1.2])

    return r_peaks


def extract_statistical_features(coeffs):
    """Wavelet 계수에 대한 통계적 특성 추출"""
    mean = np.mean(coeffs)
    std = np.std(coeffs)
    var = np.var(coeffs)
    skewness = scipy.stats.skew(coeffs)
    kurtosis = scipy.stats.kurtosis(coeffs)

    return [mean, std, var, skewness, kurtosis]


def extract_wavelet_features(signal, wavelet='db1', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    features = []

    for coeff in coeffs:
        stat_features = extract_statistical_features(coeff)
        features.extend(stat_features)

    return features

def extract_basic_stats(segment):
    """신호 세그먼트에 대한 기초 통계치를 추출합니다."""
    return [np.mean(segment), np.std(segment), np.min(segment), np.max(segment)]


def extract_ecg_features(ecg_data, fs, filename):
    leads = ecg_data.shape[0]
    all_features = []

    for lead in range(leads):
        signal = ecg_data[lead]
        lead_features = []
        check = False

        # R-peak 검출
        try:
            out = ecg.ecg(signal=signal, sampling_rate=fs, show=False)
            r_peaks = out['rpeaks']
        except ValueError:
            # print(f"An error occurred in file: {filename}")
            r_peaks = [0]
            check = True
            # return None  # 문제가 생긴 경우 None 반환
            # continue
            # Handle this case (e.g., skip this sample or use a different method)

        # r_peaks = pan_tompkins_qrs(signal)  # fs는 샘플링 주파수입니다.
        if len(r_peaks) < 2:
            # print("r_peaks < 2")
            rr_intervals = np.array([0])


        # R-R 간격
        rr_intervals = np.diff(r_peaks) / fs
        # lead_features.append(('rr_mean', np.mean(rr_intervals)))
        # lead_features.append(('rr_std', np.std(rr_intervals)))
        lead_features.append(np.mean(rr_intervals))
        lead_features.append(np.std(rr_intervals))

        # QRS 복잡도 (간단한 예: QRS 폭)
        qrs_widths = np.diff(r_peaks)
        # lead_features.append(('qrs_mean_width', np.mean(qrs_widths)))
        # lead_features.append(('qrs_std_width', np.std(qrs_widths)))
        lead_features.append((np.mean(qrs_widths)))
        lead_features.append((np.std(qrs_widths)))

        # FFT 특성
        fft_values = scipy.fft.fft(signal)
        fft_freq = scipy.fft.fftfreq(len(fft_values), 1/fs)
        lead_features.append((fft_freq[np.argmax(np.abs(fft_values))]))

        # 통계적 특성
        lead_features.append((np.mean(signal)))
        lead_features.append((np.std(signal)))
        lead_features.append((scipy.stats.skew(signal)))

        # Wavelet 특성
        wavelet_features = extract_wavelet_features(signal)
        lead_features.extend((wavelet_features))

        # for r in r_peaks:
        #     p_wave_segment = signal[max(0, r-50):r]  # 예: R-peak 전 50개 샘플을 P-wave로 가정
        #     t_wave_segment = signal[r:min(len(signal), r+50)]  # 예: R-peak 후 50개 샘플을 T-wave로 가정
        #
        #     lead_features.append((extract_basic_stats(p_wave_segment)))
        #     lead_features.append((extract_basic_stats(t_wave_segment)))
        
        # if check:
        #     print(lead_features)


        # 해당 리드의 모든 특성을 리스트에 추가
        all_features.append(lead_features)

    return all_features


def perform_pca(data, n_components=2):
    # PCA 객체 생성
    pca = PCA(n_components=n_components)

    # 데이터에 PCA 적용
    pca_result = pca.fit_transform(data)

    return pca_result



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

        # # 함수를 적용합니다.
        # data = filter_all_leads(data, fs)
        # data = z_score_normalization(data)

        all_features = extract_ecg_features(data, fs, filename)
        if all_features is None:  # 문제가 생긴 경우
            # print(f"Skipping file: {filename}")
            continue  # 현재 파일을 건너뛰고 다음 파일로

        pca = perform_pca(data)
        data = np.hstack((data, pca, all_features))

        # 결과를 .npy로 저장합니다.
        np.save(output_path, data)


fs = 500.0  # sampling frequency
lowcut = 0.5
highcut = 50.0

# -- argument
parser = argparse.ArgumentParser()
parser.add_argument('--csv_file', type=str, default='dataset/submission.csv', help="csv file path")
parser.add_argument('--input_folder', type=str, default='dataset/valid', help="numpy file directory")
parser.add_argument('--output_folder', type=str, default='dataset/valid_pre', help="preprocess output numpy file directory")
args = parser.parse_args()


csv_file = args.csv_file
input_dir = args.input_folder
output_dir = args.output_folder

# 함수를 호출하여 작업을 실행합니다.
process_and_save_npy_files(csv_file, input_dir, output_dir)

print(f"preprocess end : {output_dir}")
