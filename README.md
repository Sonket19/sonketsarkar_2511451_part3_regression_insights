# Part 3: Regression-Based Business Insights & Model Interpretation

**Student:** sonketsarkar | **ID:** 2511451 | **Part:** 3 of 4

## Business Problem
A retail chain wants to understand what drives monthly store sales across different store types and regions. Build and interpret regression models to identify key predictors and provide actionable recommendations.

## Dataset
- File: business_regression_data.xlsx
- Records: 320 store-month observations, 14 columns
- Target variable: monthly_sales
- Fields: store_id, month, region, store_type, marketing_spend, footfall, avg_discount_pct, staff_count, inventory_availability_pct, competitor_distance_km, holiday_flag, customer_rating, monthly_sales, monthly_profit

## Tools
Python 3.9, pandas, numpy, scikit-learn, scipy, openpyxl

## Tasks Completed
| Task | Description |
|------|-------------|
| Task 1 | Understand dataset - EDA, shape, missing values, target distribution |
| Task 2 | Prepare regression workbook - feature engineering, 80/20 train/test split, scaling |
| Task 3 | Create dummy variables - encode region and store_type |
| Task 4 | Run simple regression models - one predictor at a time ranked by R2 |
| Task 5 | Run multiple regression model - all 14 features simultaneously |
| Task 6 | Compare models - Baseline vs Simple vs Multiple regression |
| Task 7 | Residual analysis - normality check via Shapiro-Wilk test |
| Task 8 | Write model equation - coefficients and business interpretation |
| Task 9 | Write final recommendation - business insights from model drivers |

## Outputs
| File | Description |
|------|-------------|
| outputs/models/dataset_summary.csv | Descriptive statistics |
| outputs/models/simple_regression_results.csv | Simple model R2 and RMSE per feature |
| outputs/models/multiple_regression_coefficients.csv | All feature coefficients ranked |
| outputs/models/multiple_regression_metrics.json | R2, RMSE, MAE for multiple regression |
| outputs/models/model_comparison.csv | Baseline vs Simple vs Multiple |
| outputs/models/residuals.csv | Actual vs Predicted vs Residual |
| outputs/models/residual_analysis.json | Shapiro-Wilk normality test result |
| outputs/models/model_equation.txt | Full regression equation |
| outputs/models/final_recommendation.txt | Business recommendations |
| outputs/models/regression_analysis.xlsx | Full Excel workbook |

## Key Findings
- footfall is the strongest single predictor of monthly_sales (R2=0.69)
- Multiple regression achieves R2=0.825 explaining 82.5% of sales variance
- Top positive drivers: footfall, marketing_spend, staff_count
- Residential and High Street store types underperform vs Mall stores
- Residuals are normally distributed confirming model validity (Shapiro-Wilk p=0.37)
- RMSE of 41685 on average monthly sales of 680629

## Assumptions
- 80/20 train-test split with random_state=42 for reproducibility
- StandardScaler applied for fair coefficient comparison across features
- Missing competitor_distance_km and customer_rating imputed with column median
- monthly_profit excluded from features to avoid data leakage

## How to Run
pip install pandas numpy scikit-learn scipy openpyxl

python scripts/regression_analysis.py
