import argparse
import pandas as pd
import random, os
import numpy as np
import torch

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 자신이 사용한 프레임워크(ex. tensorflow, torch, sklearn)가 있다면 random seed를 정해주셔야 합니다.
    # 만약 고정하지 않아서 생기는 성능 저하 등 문제로 인한 책임은 제출자가 집니다.

def inference(args):
    df = pd.read_csv(args.csv_file)
    ### 직접 작성해주세요.
    ### 코드 예시) 저희는 모든 사람이 50살이라고 예측했을때 mae가 가장 작다는 결론을 냈습니다.
    df["AGE"] = 50
    return df

if __name__ == "__main__":
    # -- argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, default = None, help="csv file path")
    parser.add_argument('--numpy_folder', type=str, default= None, help="numpy file directory")
    parser.add_argument('--model_path', type=str, dafault= None, help="trained model path")
    args = parser.parse_args()

    # -- set seed
    seed_everything(42)

    # -- inference
    result_df = inference(args)

    # -- save result
    result_df.to_csv("./submission.csv", index=False)
