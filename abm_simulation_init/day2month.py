# %%
import os
import ast
import numpy as np
import pandas as pd

# 원본 파일
infile = "../result/emulation_beta_A10140_lambda.csv"

# 저장 파일
outfile = "../result/emulation_beta_A10140_lambda_monthly.csv"

days_per_month = 30
n_months = 19   # A 기간

# 읽기
raw_df = pd.read_csv(infile)

def parse_series(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, list):
        return x
    return ast.literal_eval(x)

def daily_to_monthly(series, days_per_month=30, n_months=19):
    arr = np.array(series, dtype=float)

    # 필요한 길이만 사용
    arr = arr[:days_per_month * n_months]

    # 30일씩 묶어서 월합
    monthly = arr.reshape(n_months, days_per_month).sum(axis=1)

    # 다시 리스트로
    return monthly.tolist()

# 각 셀 변환
monthly_df = raw_df.copy()

for col in monthly_df.columns:
    monthly_df[col] = monthly_df[col].apply(
        lambda x: daily_to_monthly(parse_series(x), days_per_month, n_months)
        if pd.notna(x) else np.nan
    )

# 저장
monthly_df.to_csv(outfile, index=False)

print("saved ->", outfile)
print(monthly_df.head())
# %%
