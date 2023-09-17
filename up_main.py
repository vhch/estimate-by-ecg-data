import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup
import numpy as np
import random

from model import *
from customdataset import CustomDataset


# Seed 값을 고정
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def upsampling(train_indices, dataset, min_threshold=100):
    df = dataset.df
    indices_by_age_group = {}

    # 범위별로 데이터를 그룹화합니다.
    for idx in train_indices:
        age = df.iloc[idx]['AGE']
        age_group = int(age)
        if age_group not in indices_by_age_group:
            indices_by_age_group[age_group] = []
        indices_by_age_group[age_group].append(idx)

    print(indices_by_age_group.keys())
    print([len(indices) for indices in indices_by_age_group.values()])

    # 가장 큰 그룹의 크기를 찾습니다.
    max_size = max([len(indices) for indices in indices_by_age_group.values()])
    avg_size = int(np.mean([len(indices) for indices in indices_by_age_group.values()]))

    # 모든 그룹의 크기를 가장 큰 그룹의 크기와 동일하게 맞춰주기 위해 업샘플링합니다.
    upsampled_indices = []
    for indices in indices_by_age_group.values():
        # upsampled_indices.extend(np.random.choice(indices, size=avg_size, replace=True))
        # 해당 그룹의 크기가 min_threshold 이하일 경우, 업샘플링을 건너뛴다.
        if len(indices) <= min_threshold:
            upsampled_indices.extend(indices)
        else:
            upsampled_indices.extend(np.random.choice(indices, size=max_size, replace=True))

    return upsampled_indices


# GPU 사용 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
scaler = GradScaler()

# Paths
csv_path_adult = 'dataset/ECG_adult_age_train.csv'
numpy_folder_adult = 'dataset/adult/train/'

csv_path_child = 'dataset/ECG_child_age_train.csv'
numpy_folder_child = 'dataset/child/train/'

dataset_adult = CustomDataset(csv_path_adult, numpy_folder_adult)
dataset_child = CustomDataset(csv_path_child, numpy_folder_child)

dataset = ConcatDataset([dataset_adult, dataset_child])

# dataset = dataset_adult
dataset = dataset_child
checkpoint_path = 'cnntolstm_child2_up_thresold.pth'


train_len = int(0.9 * len(dataset))
val_len = len(dataset) - train_len

train_dataset_original, val_dataset = random_split(dataset, [train_len, val_len])

# 학습 데이터만 업샘플링
upsampled_train_indices = upsampling(train_dataset_original.indices, dataset)
train_dataset = Subset(dataset, upsampled_train_indices)

print("train dataset len", len(train_dataset))

batch_size = 100
num_epochs = 200
accumulation_steps = 1

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, num_workers=4)


# model = Model().to(device)
# model = CNNTOLSTM(input_size=32, hidden_size=100, num_layers=1).to(device)
model = CNNTOLSTM(input_size=32, hidden_size=256, num_layers=2, dropout=0.1).to(device)

# Loss and Optimizer
# criterion = nn.HuberLoss()
criterion = nn.MSELoss()  # Mean Squared Error for regression
criterion_val = nn.L1Loss()
# optimizer = optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-5, betas=(0.9, 0.999))
optimizer = optim.AdamW(model.parameters(), lr=0.001)
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

    for batch_idx, (data, gender, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        # forward
        data, gender, targets = data.to(device), gender.to(device), targets.to(device)
        with autocast():
            output = model(data)
            # print(data.shape)
            # print(gender.shape)
            # print(output.shape)
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
        for batch_idx, (data, gender, targets) in enumerate(val_loader):
            data, gender, targets = data.to(device), gender.to(device), targets.to(device)
            outputs = model(data)
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

        print(f"epoch:{epoch}, New best validation loss ({best_val_loss:.4f}), saving model...")

    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, best_val_loss: {best_val_loss}")
