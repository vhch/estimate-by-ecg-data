import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup
import random
import numpy as np
from scipy.interpolate import interp1d
from scipy.fft import fft, ifft, fftshift

from model import *
from customdataset import CustomDataset

class AugmentedSubset(Subset):
    def __init__(self, subset, transform=None):
        super(AugmentedSubset, self).__init__(subset.dataset, subset.indices)
        self.transform = transform

    def __getitem__(self, idx):
        data, gender, age, age_group = self.dataset[self.indices[idx]]
        if self.transform:
            data = self.transform(data)
        data = torch.tensor(data, dtype=torch.float16).clone().detach()

        return data, gender, age, age_group
        # return torch.tensor(data, dtype=torch.float16), gender, age, age_group

def time_shift(data, shift):
    """
    data: 2D numpy array of shape (12, 5000)
    shift: integer, time shift value
    """
    if shift > 0:
        return np.pad(data, ((0, 0), (shift, 0)), mode='constant')[:, :-shift]
    elif shift < 0:
        return np.pad(data, ((0, 0), (0, -shift)), mode='constant')[:, -shift:]
    else:
        return data

def add_noise(data):
    """
    data: 2D numpy array of shape (12, 5000)
    """
    noise = np.random.normal(0, 0.05, data.shape)
    return data + noise

def lead_permutation(data):
    """
    data: 2D numpy array of shape (12, n), where n is the number of data points for each lead.
    """
    return np.random.permutation(data)

def random_scale(signal, min_scale=0.95, max_scale=1.05):
    scale_factor = np.random.uniform(min_scale, max_scale)
    return signal * scale_factor

def time_stretch_multi_lead(signal, stretch_factor):
    stretched_signal = np.zeros_like(signal)
    for i in range(signal.shape[0]):  # 여기서 0번째 차원이 lead에 해당되어야 합니다.
        stretched_signal[i] = time_stretch(signal[i], stretch_factor)
    return stretched_signal

def frequency_shift_multi_lead(signal, shift_amount):
    shifted_signal = np.zeros_like(signal, dtype=np.float64)  # 복소수 값을 다룰 수 있도록 dtype을 설정
    for i in range(signal.shape[0]):
        shifted_signal[i] = frequency_shift(signal[i], shift_amount)
    return np.real(shifted_signal)  # 실수 부분만 반환

def train_augment(x):
    x = time_shift(x, np.random.randint(-10, 10))
    x = add_noise(x)
    x = random_scale(x)
    x = lead_permutation(x)
    return x

# Seed 값을 고정
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 경로 설정
checkpoint_dir = "checkpoint"

# 경로에 디렉토리가 없으면 생성
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


# GPU 사용 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
scaler = GradScaler()

data_dir = 'dataset/data_filt_zscore_feature2'
# checkpoint_path = 'checkpoint/Cnntogru_concat_85cut_batch128_1e-3_filter_zscorenorm_featureaa4.pth'
checkpoint_path = 'checkpoint/Cnntogru_concat_85cut_batch128_1e-3_filter_zscorenorm_featureaa.pth'
# checkpoint_path = 'checkpoint/resnet_concat_85cut_batch128_4e-4_filter_zscorenorm_feature2.pth'

# Paths
csv_path_adult = 'dataset/ECG_adult_age_train.csv'
numpy_folder_adult = 'dataset/adult/train/'
numpy_folder_adult = data_dir

csv_path_child = 'dataset/ECG_child_age_train.csv'
numpy_folder_child = 'dataset/child/train/'
numpy_folder_child = data_dir

dataset_adult = CustomDataset(csv_path_adult, numpy_folder_adult)
dataset_child = CustomDataset(csv_path_child, numpy_folder_child)

dataset = ConcatDataset([dataset_adult, dataset_child])


# dataset = dataset_adult
# dataset = dataset_child
batch_size = 128
num_epochs = 100
accumulation_steps = 1


train_len = int(0.9 * len(dataset))
val_len = len(dataset) - train_len

train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

# train_dataset = AugmentedSubset(train_dataset, transform=train_augment)
# val_dataset = AugmentedSubset(val_dataset, transform=None)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, num_workers=4)


# model = Model().to(device)
# model = EnhancedCNNGRUAgePredictor3().to(device)
model = EnhancedCnntobert2().to(device)
model = Model().to(device)

# Loss and Optimizer
# criterion = nn.HuberLoss()
criterion = nn.MSELoss()  # Mean Squared Error for regression
criterion_val = nn.L1Loss()
# optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5, betas=(0.9, 0.999))
# optimizer = optim.AdamW(model.parameters(), lr=1e-3)
optimizer = optim.AdamW(model.parameters(), lr=4e-4)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=len(train_loader) * num_epochs / accumulation_steps)

best_val_loss = float('inf')

# Checkpoint 불러오기 (만약 있다면)
start_epoch = 0

# 모델이 이미 있는 경우에만 실행
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']


# Train the model
for epoch in range(start_epoch, num_epochs):
    model.train()
    train_loss = 0.0

    for batch_idx, (data, gender, targets, age_group) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        data, gender, targets, age_group = data.to(device), gender.to(device), targets.to(device), age_group.to(device)
        if torch.isnan(data).any():
            print("There are NaN values in your training tensor.")
        with autocast():
            output = model(data, gender, age_group)
            loss = criterion(output.reshape(-1), targets.reshape(-1))

        train_loss += loss.item()
        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            scheduler.step()

    train_loss /= len(train_loader)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (data, gender, targets, age_group) in enumerate(val_loader):
            data, gender, targets, age_group = data.to(device).float(), gender.to(device), targets.to(device), age_group.to(device)
            outputs = model(data, gender, age_group)
            val_loss += criterion_val(outputs.reshape(-1), targets.reshape(-1)).item()
    # print(outputs[:5], targets[:5])
    val_loss /= len(val_loader)

    # Checkpoint 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'best_val_loss': best_val_loss
        }
        torch.save(checkpoint, f'{checkpoint_path}')

        print(f"epoch:{epoch+1}, New best validation loss ({best_val_loss:.4f}), saving model: {checkpoint_path}")

    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, best_val_loss: {best_val_loss}")
