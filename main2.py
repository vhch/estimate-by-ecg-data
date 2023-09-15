import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

BATCH_SIZE = 1000

csv_path_adult = 'dataset/ECG_adult_age_train.csv'
csv_path_child = 'dataset/ECG_child_age_train.csv'
numpy_folder_adult = 'dataset/adult/train/'
numpy_folder_child = 'dataset/child/train/'

# CSV 파일 불러오기
df1 = pd.read_csv(f'{csv_path_adult}')
df2 = pd.read_csv(f'{csv_path_child}')

# 데이터프레임에 소속 폴더 컬럼 추가
df1['FOLDER'] = numpy_folder_adult
df2['FOLDER'] = numpy_folder_child

gender_mapping = {'MALE': 1, 'FEMALE': 0}
df1['GENDER'] = df1['GENDER'].map(gender_mapping)
df2['GENDER'] = df2['GENDER'].map(gender_mapping)

# 두 데이터프레임 합치기
df = pd.concat([df1, df2], ignore_index=True)

# 전체 데이터에서 학습 및 검증 데이터 분할
train_df, valid_df = train_test_split(df, test_size=0.1, random_state=42)


def data_generator(train_df):
    n = len(train_df)

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        subset = train_df.iloc[start:end]

        data = []
        labels = []
        for _, row in subset.iterrows():
            filename = row['FILENAME']
            folder = row['FOLDER']
            path = os.path.join(folder, f"{filename}.npy")
            data.append(np.load(path).flatten())
            labels.append(row['AGE'])

        yield pd.DataFrame(data), np.array(labels)


gen = data_generator(train_df)

# 검증 데이터 미리 로드
valid_data = []
valid_labels = []
for _, row in valid_df.iterrows():
    filename = row['FILENAME']
    folder = row['FOLDER']
    path = os.path.join(folder, f"{filename}.npy")
    valid_data.append(np.load(path).flatten())
    valid_labels.append(row['AGE'])
X_valid = pd.DataFrame(valid_data)
y_valid = np.array(valid_labels)

model = XGBRegressor(objective='reg:squarederror')

for X_batch, y_batch in gen:
    model.fit(X_batch, y_batch, eval_set=[(X_valid, y_valid)], eval_metric='mae', early_stopping_rounds=10, verbose=True)

y_pred = model.predict(X_valid)
mae = mean_absolute_error(y_valid, y_pred)
print(f'Mean Absolute Error: {mae}')
