
# Cleaning Report

1.  **Initial Cleaning:** Outliers were removed using the standard cleaning tool.
2.  **Dropped 'ID':** Removed the ID column as it is an identifier.
3.  **Multicollinearity Handling:**
    *   Dropped 'rad' due to multicollinearity (lower correlation with target than 'tax').
    *   Dropped 'tax' due to multicollinearity with 'room_tax'.
    *   Dropped 'nox' due to multicollinearity with 'crim'.
4.  **Skewness Handling:**
    *   Applied log transform to 'crim', 'zn', 'dis', and 'black'.
    *   Repeated log transform to 'zn' and 'black' as suggested by the critic.
5.  **Scaling:** Applied StandardScaler to 'tax' and 'black'.
6.  **Interaction Term:** Created 'room_tax' as 'rm' * 'tax'.

