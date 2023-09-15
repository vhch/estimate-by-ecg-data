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

gender_mapping = {'male': 1, 'female': 0}  # 예를 들어 이렇게 매핑. 실제 데이터에 맞게 조정해야 합니다.
df1['GENDER'] = df1['GENDER'].map(gender_mapping)
df2['GENDER'] = df2['GENDER'].map(gender_mapping)

# 두 데이터프레임 합치기
df = pd.concat([df1, df2], ignore_index=True)

def data_generator(df):
    n = len(df)
    
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        subset = df.iloc[start:end]
        
        data = []
        for _, row in subset.iterrows():
            filename = row['FILENAME']
            folder = row['FOLDER']
            path = os.path.join(folder, f"{filename}.npy")
            data.append(np.load(path).flatten())
        
        batch_df = pd.DataFrame(data)
        X = batch_df
        y = subset['AGE']
        
        yield X, y

gen = data_generator(df)

model = XGBRegressor(objective='reg:squarederror')

# 초기 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['FILENAME', 'FOLDER', 'AGE']), df['AGE'], test_size=0.1, random_state=42)

for batch_X, batch_y in gen:
    model.fit(batch_X, batch_y, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

