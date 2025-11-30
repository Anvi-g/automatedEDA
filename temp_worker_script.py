import pandas as pd
train_data = pd.read_csv('linear_working.csv')
train_data.to_csv('data/processed/linear_ready_train.csv', index=False)
print("Saved final linear dataset.")

report_content = '''
# Cleaning Report

Initial cleaning was performed using standard_cleaning_tool, which removed outliers and standardized the data.

Further actions taken:
- Dropped 'rad' due to multicollinearity (VIF > 5) and lower correlation with the target compared to 'tax'.
- Applied log transform to 'crim' and 'zn' to reduce skewness.
- Applied square transform to 'black' to reduce skewness.
- Skipped re-transforming 'crim', 'zn', and 'black' due to anti-looping.
'''

with open('cleaning_report.md', 'w') as f:
    f.write(report_content)
print("SUCCESS: cleaning_report.md saved.")