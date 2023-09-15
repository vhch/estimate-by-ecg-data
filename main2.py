import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from concurrent.futures import ProcessPoolExecutor

csv_path_adult = 'dataset/ECG_adult_age_train.csv'
csv_path_child = 'dataset/ECG_child_age_train.csv'
numpy_folder_adult = 'dataset/adult/train/'
numpy_folder_child = 'dataset/child/train/'  # 두 번째 numpy 폴더 경로

# CSV 파일 불러오기
df1 = pd.read_csv(f'{csv_path_adult}')
df2 = pd.read_csv(f'{csv_path_child}')

# 데이터프레임에 소속 폴더 컬럼 추가
df1['FOLDER'] = numpy_folder_adult
df2['FOLDER'] = numpy_folder_child

# 두 데이터프레임 합치기
df = pd.concat([df1, df2], ignore_index=True)

# 메모리에 모든 npy 파일 로드하기
file_data = {}
for filename in df['FILENAME']:
    if filename in file_data:  # 이미 로딩한 경우 스킵
        continue

    path = os.path.join(numpy_folder_adult, f"{filename}.npy")
    if not os.path.exists(path):  # 다른 폴더에서 찾기
        path = os.path.join(numpy_folder_child, f"{filename}.npy")
    file_data[filename] = np.load(path).flatten()

# 메모리에 로딩한 데이터를 사용하여 DataFrame 생성하기
data = [file_data[filename] for filename in df['FILENAME']]
numpy_df = pd.DataFrame(data)

# 원본 df와 numpy_df를 연결
df = pd.concat([df, numpy_df], axis=1)

# 필요없는 'FILENAME' 및 'FOLDER' 열을 제거
df.drop(columns=['FILENAME', 'FOLDER'], inplace=True)

# 데이터 분할
X = df.drop(columns=['AGE'])
y = df['AGE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 모델 학습
model = XGBRegressor(objective='reg:squarederror')
model.fit(X_train, y_train)

# 예측 및 성능 확인
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

