report_content = '''
# Cleaning Report

**Initial Cleaning:**
*   Removed outliers using the standard cleaning tool.

**Multicollinearity Handling:**
*   Dropped 'rad' due to multicollinearity with 'tax'.

**Skewness Handling:**
*   Applied log transform to 'crim'.
*   Applied log transform to 'zn'.
*   Applied square transform to 'black'.

**Scaling:**
*   Applied StandardScaler to 'tax' and 'black'.

**Other:**
*   Dropped 'ID' column.

**Remaining Issues:**
*   Skewness in 'crim', 'zn', and 'black' (already transformed once).
'''

with open('cleaning_report.md', 'w') as f:
    f.write(report_content)
print("SUCCESS: cleaning_report.md saved.")