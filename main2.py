import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold

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

# StratifiedKFold 설정
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
labels = [age for _, _, _, age_group in dataset]

batch_size = 128
num_epochs = 400
accumulation_steps = 1

# Loss and Optimizer
criterion = nn.MSELoss()
criterion_val = nn.L1Loss()
optimizer = optim.AdamW(model.parameters(), lr=4e-4)

best_val_loss = float('inf')
best_fold = -1

for fold, (train_idx, val_idx) in enumerate(skf.split(dataset, labels)):
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, num_workers=4)
    
    model = Cnntobert2().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=4e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=len(train_loader) * num_epochs / accumulation_steps)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (data, gender, targets, age_group) in enumerate(tqdm(train_loader)):
            data, gender, targets, age_group = data.to(device), gender.to(device), targets.to(device), age_group.to(device)
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
        print(f"Fold: {fold+1}/{n_splits}, Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_idx, (data, gender, targets, age_group) in enumerate(val_loader):
                data, gender, targets, age_group = data.to(device), gender.to(device), targets.to(device), age_group.to(device)
                outputs = model(data, gender, age_group)
                val_loss += criterion_val(outputs.reshape(-1), targets.reshape(-1)).item()

        val_loss /= len(val_loader)
        print(f"Fold: {fold+1}/{n_splits}, Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

        # 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_fold = fold
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, f'{checkpoint_dir}/model_fold_{fold}.pth')
            print(f"Fold: {fold+1}/{n_splits}, epoch:{epoch+1}, New best validation loss ({best_val_loss:.4f}), saving model: {checkpoint_dir}/model_fold_{fold}.pth")

print(f"Overall best validation loss: {best_val_loss:.4f} at fold {best_fold+1}")
