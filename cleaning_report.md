
# Cleaning Report

Initial cleaning was performed using standard_cleaning_tool, which removed outliers and standardized the data.

Further actions taken:
- Dropped 'rad' due to multicollinearity (VIF > 5) and lower correlation with the target compared to 'tax'.
- Applied log transform to 'crim' and 'zn' to reduce skewness.
- Applied square transform to 'black' to reduce skewness.
- Skipped re-transforming 'crim', 'zn', and 'black' due to anti-looping.
