# DATA CLEANING REPORT

## 1. Overview
This report summarizes the automated cleaning and preprocessing performed by the ADEP logistic pipeline.

## 2. Critic Findings Over Iterations
- No critic messages were recorded.

## 3. Applied Transformations
- Log transform 'oldpeak', 'ca'; Standard scale 'id', 'chol' and other numerical columns, Ensure target is 0/1 (auto)
- Convert 'fbs' and 'exang' to integers (auto)

## 4. Final Dataset Snapshot
- Shape: (702, 16)
- Columns: ['id', 'age', 'sex', 'dataset', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']