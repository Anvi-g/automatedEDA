
# Cleaning Report

This report summarizes the cleaning steps performed to prepare the data for linear regression.

**Data Loading:**
- The data was loaded from 'data/processed/train.csv'.

**Cleaning Steps:**
- Removed outliers using the standard cleaning tool.
- Removed the 'ID' column.
- Removed rows with negative values in columns: 'crim', 'zn', 'indus', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat'.
- Created interaction term 'rm_x_lstat' = 'rm' * 'lstat'.
- Converted 'rad' to categorical using one-hot encoding.
- Dropped one 'rad' dummy column ('rad_2') to avoid multicollinearity.
- Applied log transform to 'crim' and 'dis' to reduce skewness.

**Remaining Issues:**
- Due to tool limitations, skewness, variance and multicollinearity were not accurately assessed.

