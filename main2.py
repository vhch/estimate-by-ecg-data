import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

csv_path_adult = 'dataset/ECG_adult_age_train.csv'
numpy_folder_adult = 'dataset/adult/train/'

# CSV 파일 불러오기
df = pd.read_csv(f'{csv_path_adult}')

# 빈 리스트를 생성하여 numpy 데이터를 추가
data = []

for index, row in df.iterrows():
    filename = row['FILENAME']
    arr = np.load(f'{numpy_folder_adult}/{filename}.npy')
    data.append(arr.flatten())  # 2D 배열을 1D로 변경

# numpy 데이터를 pandas DataFrame 형식으로 변경
numpy_df = pd.DataFrame(data)

# 원본 df와 numpy_df를 연결
df = pd.concat([df, numpy_df], axis=1)


# 필요없는 'FILENAME' 열을 제거
df.drop(columns=['FILENAME'], inplace=True)


# 데이터 분할
X = df.drop(columns=['AGE'])
y = df['AGE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 모델 학습
model = XGBRegressor(objective='reg:squarederror')
model.fit(X_train, y_train)

# 예측 및 성능 확인
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
