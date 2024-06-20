import pandas as pd
from datasets import Dataset,load_dataset

# 1. 허깅 페이스 데이터셋 로드
en_ko = load_dataset("bongsoo/news_talk_en_ko")

# 2. 로드한 데이터 포맷 설정
en_ko.set_format(type="pandas")

# 3. 모든 데이터셋을 train키로 지정
df = en_ko["train"][:]

# 4. 데이터셋에 en ko 헤더 지정
example_0 = list(df.columns)
example_0_df = pd.DataFrame({col: [value] for col, value in zip(('en', 'ko'), example_0)})
df.columns = ('en', 'ko')
en_ko_df = pd.concat([example_0_df, df],).reset_index(drop=True)
en_ko_df.head()
dataset = Dataset.from_pandas(en_ko_df)

# 5. 훈련, 검증, 테스트로 데이터 분할 개수 설정
num_train = 1200000
num_valid = 90000
num_test = 10000

# 6. 데이터셋 분할
en_ko_df_train = en_ko_df.iloc[:num_train]
en_ko_df_valid = en_ko_df.iloc[num_train:num_train+num_valid]
en_ko_df_test = en_ko_df.iloc[-num_test:]

# 7. 분할 데이터셋 저장
en_ko_df_train.to_csv("train.tsv", sep='\t', index=False)
en_ko_df_valid.to_csv("valid.tsv", sep='\t', index=False)
en_ko_df_test.to_csv("test.tsv", sep='\t', index=False)