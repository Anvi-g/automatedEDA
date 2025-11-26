import pandas as pd
df = pd.read_csv('housing_data.csv')
df.dropna(inplace=True)
numeric_df = df.select_dtypes(include=['number'])
Q1 = numeric_df.quantile(0.25)
Q3 = numeric_df.quantile(0.75)
IQR = Q3 - Q1
df_out = numeric_df[~((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)]
cleaned_df = df.loc[df_out.index]
cleaned_df.to_csv('cleaned_data.csv', index=False)
print('Data cleaning complete')
print(cleaned_df.head())
