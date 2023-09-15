import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from model import Model
from customdataset import CustomDataset


# GPU 사용 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Paths
csv_path_adult = 'dataset/ECG_adult_age_train.csv'
numpy_folder_adult = 'dataset/adult/train/'

csv_path_child = 'dataset/ECG_child_age_train.csv'
numpy_folder_child = 'dataset/child/train/'

dataset_adult = CustomDataset(csv_path_adult, numpy_folder_adult)
dataset_child = CustomDataset(csv_path_child, numpy_folder_child)

dataset = ConcatDataset([dataset_adult, dataset_child])

dataset = dataset_adult

# for i in range(5):
#     data_sample, gender_sample, age_sample = dataset[i]
#     print(f"Sample {i+1}:")
#     print(data_sample.shape)
#     print(gender_sample)
#     print(age_sample)
#     print("------")
# exit()

train_len = int(0.9 * len(dataset))
val_len = len(dataset) - train_len

train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)


model = Model().to(device)

# Loss and Optimizer
num_epochs = 100

criterion = nn.MSELoss()  # Mean Squared Error for regression
criterion_val = nn.L1Loss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * num_epochs)

best_val_loss = float('inf')

# Train the model
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for batch_idx, (data, gender, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        # forward
        data, gender, targets = data.to(device), gender.to(device), targets.to(device)
        outputs = model(data)
        loss = criterion(outputs, targets.unsqueeze(1))
        train_loss += loss.item()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step()

    train_loss /= len(train_loader)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (data, gender, targets) in enumerate(val_loader):
            data, gender, targets = data.to(device), gender.to(device), targets.to(device)
            outputs = model(data)
            val_loss += criterion_val(outputs, targets.unsqueeze(1)).item()
    # print(outputs[:5], targets[:5])
    val_loss /= len(val_loader)

    # Checkpoint 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model_checkpoint.pth')
        print(f"New best validation loss ({best_val_loss:.4f}), saving model...")

    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, best_val_loss: {best_val_loss}")
