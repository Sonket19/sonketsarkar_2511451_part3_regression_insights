# Part 3: Regression-Based Business Insights & Model Interpretation

**Assignment:** Final Capstone Project – Business Analytics
**Student:** sonketsarkar
**Student ID:** 2511451
**Part:** 3 of 4

---

## 1. Assignment Title
Regression-Based Business Insights & Model Interpretation

## 2. Business Problem Summary
A retail chain wants to understand what drives monthly store sales across different store types and regions. By building and interpreting linear regression models, this analysis identifies the strongest predictors of monthly_sales, quantifies their impact, and provides actionable recommendations to maximise store performance.

## 3. Dataset Used
- **Source:** Google Drive (provided by instructor)
- **File:** `business_regression_data.xlsx` placed in `data/` folder
- **Records:** 320 store-month observations across 14 columns
- **Target variable:** monthly_sales
- **Key features:** store_id, month, region, store_type, marketing_spend, footfall, avg_discount_pct, staff_count, inventory_availability_pct, competitor_distance_km, holiday_flag, customer_rating

## 4. Tools Used
- **Python 3.9+**
- **pandas** – data handling
- **numpy** – numerical operations
- **scikit-learn** – LinearRegression, train_test_split, StandardScaler, metrics
- **scipy.stats** – Shapiro-Wilk residual normality test
- **openpyxl** – Excel output

## 5. Steps Performed

| Task | Description |
|------|-------------|
| Task 1 | Understand dataset – EDA, shape, missing values, target distribution |
| Task 2 | Prepare regression workbook – feature engineering, 80/20 train/test split, scaling |
| Task 3 | Create dummy variables – encode region and store_type |
| Task 4 | Run simple regression models – one predictor at a time, ranked by R² |
| Task 5 | Run multiple regression model – all 14 features simultaneously |
| Task 6 | Compare models – Baseline vs Simple vs Multiple regression |
| Task 7 | Residual analysis – normality check via Shapiro-Wilk test |
| Task 8 | Write model equation – coefficients and business interpretation |
| Task 9 | Write final recommendation – business insights from model drivers |

## 6. Key Outputs

| Output File | Description |
|-------------|-------------|
| `outputs/models/dataset_summary.csv` | Descriptive statistics |
| `outputs/models/simple_regression_results.csv` | Simple model R² and RMSE per feature |
| `outputs/models/multiple_regression_coefficients.csv` | All feature coefficients ranked by magnitude |
| `outputs/models/multiple_regression_metrics.json` | R², RMSE, MAE for multiple regression |
| `outputs/models/model_comparison.csv` | Baseline vs Simple vs Multiple comparison |
| `outputs/models/residuals.csv` | Actual vs Predicted vs Residual |
| `outputs/models/residual_analysis.json` | Shapiro-Wilk normality test result |
| `outputs/models/model_equation.txt` | Full regression equation with interpretation |
| `outputs/models/final_recommendation.txt` | Business recommendations |
| `outputs/models/regression_analysis.xlsx` | Comprehensive Excel workbook |

## 7. Business Insights
- footfall is the single strongest predictor of monthly_sales (simple R²=0.69)
- Multiple regression achieves R²=0.825 — explains 82.5% of sales variance
- Top positive drivers: footfall, marketing_spend, staff_count
- Residential and High Street store types underperform vs Mall stores
- Residuals are normally distributed (Shapiro-Wilk p=0.37) confirming model validity
- RMSE of ₹41,685 on average monthly sales of ₹680,629

## 8. Assumptions Made
- 80/20 train-test split with random_state=42 for reproducibility
- StandardScaler applied for fair coefficient comparison across features
- Missing competitor_distance_km and customer_rating imputed with column median
- drop_first=True for dummy variables to avoid multicollinearity
- monthly_profit excluded from features to avoid data leakage

## 9. Screenshots
Screenshots are in the `screenshots/` folder showing terminal output for each task.

## Folder Structure
## How to Run
```bash
pip install pandas numpy scikit-learn scipy openpyxl
python scripts/regression_analysis.py
```
