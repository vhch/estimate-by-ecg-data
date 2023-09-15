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

# numpy 데이터 로드 함수
def load_and_flatten(row):
    filename = row['FILENAME']
    folder = row['FOLDER']
    arr = np.load(f'{folder}/{filename}.npy')
    return arr.flatten()

# 병렬 처리를 사용하여 데이터 로딩
with ProcessPoolExecutor() as executor:
    data = list(executor.map(load_and_flatten, df.to_dict('records')))

# numpy 데이터를 pandas DataFrame 형식으로 변경
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

