import pandas as pd
df = pd.read_csv('logistic_working.csv')
target = df.columns[-1]
print(df[target].value_counts())