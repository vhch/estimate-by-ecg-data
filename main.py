import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup
import random
import numpy as np

from model import *
from customdataset import CustomDataset

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

# Paths
csv_path_adult = 'dataset/ECG_adult_age_train.csv'
numpy_folder_adult = 'dataset/adult/train/'

csv_path_child = 'dataset/ECG_child_age_train.csv'
numpy_folder_child = 'dataset/child/train/'

dataset_adult = CustomDataset(csv_path_adult, numpy_folder_adult)
dataset_child = CustomDataset(csv_path_child, numpy_folder_child)

dataset = ConcatDataset([dataset_adult, dataset_child])


# dataset = dataset_adult
# dataset = dataset_child
batch_size = 128
num_epochs = 400
accumulation_steps = 1
checkpoint_path = 'checkpoint/Cnntogru_concat_85cut_batch128_1e-3_filter_minmax.pth'


train_len = int(0.9 * len(dataset))
val_len = len(dataset) - train_len

train_dataset, val_dataset = random_split(dataset, [train_len, val_len])


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, num_workers=4)


# model = Model().to(device)
model = CNNGRUAgePredictor().to(device)
# model = Cnn1d().to(device)

# Loss and Optimizer
# criterion = nn.HuberLoss()
criterion = nn.MSELoss()  # Mean Squared Error for regression
criterion_val = nn.L1Loss()
# optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5, betas=(0.9, 0.999))
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
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
        with autocast():
            output = model(data, gender, age_group)
            # print(data.shape)
            # print(gender.shape)
            # print(targets.shape)
            # print(age_group.shape)
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
        for batch_idx, (data, gender, targets, age_group) in enumerate(val_loader):
            data, gender, targets, age_group = data.to(device), gender.to(device), targets.to(device), age_group.to(device)
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
