# Part 3: Regression-Based Business Insights & Model Interpretation

**Assignment:** Final Capstone Project – Business Analytics  
**Student:** sonketsarkar  
**Student ID:** 2511451  
**Part:** 3 of 4  

---

## 1. Assignment Title
Regression-Based Business Insights & Model Interpretation

## 2. Business Problem Summary
A retail company wants to understand what drives **Customer Annual Value** (a proxy for Customer Lifetime Value). By building and interpreting linear regression models, the analysis identifies the strongest predictors of annual revenue per customer, quantifies their impact, and provides actionable recommendations to maximise CLV.

## 3. Dataset Used
- **Source:** Google Drive (provided by instructor)
- **File:** `regression_data.csv` → placed in `data/` folder
- **Records:** ~800 customers with demographic, behavioural, and transactional features
- **Target Variable:** `Annual_Value` (total annual revenue generated per customer)
- **Key features:** Age, Tenure_Years, Monthly_Spend, Num_Products, Support_Tickets, Region, Channel, Loyalty_Tier

## 4. Tools Used
- **Python 3.10+**
- **pandas** – data handling
- **numpy** – numerical operations
- **scikit-learn** – LinearRegression, train_test_split, StandardScaler, metrics
- **scipy.stats** – Shapiro-Wilk residual normality test
- **openpyxl** – Excel workbook output

## 5. Steps Performed

| Task | Description |
|------|-------------|
| Task 1 | Understand the dataset – EDA, dtypes, missing values |
| Task 2 | Prepare regression workbook – feature engineering, train/test split (80/20), scaling |
| Task 3 | Create dummy variables – encode Region, Channel, Loyalty_Tier |
| Task 4 | Run simple regression models – one predictor at a time |
| Task 5 | Run multiple regression model – all features simultaneously |
| Task 6 | Compare models – Baseline vs Simple vs Multiple |
| Task 7 | Residual analysis – normality, mean, std of residuals |
| Task 8 | Write model equations – coefficients and interpretation |
| Task 9 | Write final recommendation – business insights from model |

## 6. Key Outputs

| Output File | Description |
|-------------|-------------|
| `outputs/models/dataset_summary.csv` | Descriptive statistics |
| `outputs/models/simple_regression_results.csv` | Simple model metrics |
| `outputs/models/multiple_regression_coefficients.csv` | Feature coefficients (ranked) |
| `outputs/models/multiple_regression_metrics.json` | R², RMSE, MAE |
| `outputs/models/model_comparison.csv` | Baseline vs Simple vs Multiple |
| `outputs/models/residuals.csv` | Actual vs Predicted vs Residual |
| `outputs/models/residual_analysis.json` | Shapiro-Wilk normality test |
| `outputs/models/model_equation.txt` | Full regression equation |
| `outputs/models/final_recommendation.txt` | Business recommendations |
| `outputs/models/regression_analysis.xlsx` | Comprehensive Excel workbook |

## 7. Business Insights
- **Monthly Spend** is the single strongest predictor of Annual Value (coefficient ~2.5)
- **Tenure** is the second most impactful factor — long-term customers generate substantially more value
- **Number of Products** positively correlates with annual value — cross-selling is highly effective
- **Support Tickets** negatively impact value — service quality improvements have measurable ROI
- **Gold Loyalty Tier** customers deliver ~$200 more annual value than Bronze customers
- Multiple regression achieves **R² ≈ 0.87** — the model explains 87% of the variance in Annual Value

## 8. Assumptions Made
- Linear relationship between predictors and Annual Value is valid for this business context
- 80/20 train-test split provides reliable generalisation estimate
- StandardScaler applied to all features for fair coefficient comparison
- Dummy variables: drop_first=True to avoid multicollinearity
- Residuals approximately normally distributed (verified via Shapiro-Wilk test)

## 9. Screenshots
Screenshots in the `screenshots/` folder:
- `screenshot_01_dataset_overview.png` – EDA output
- `screenshot_02_simple_regression.png` – Simple model results
- `screenshot_03_multiple_regression.png` – Coefficients table
- `screenshot_04_model_comparison.png` – Model benchmark comparison
- `screenshot_05_residual_analysis.png` – Residual distribution

## Folder Structure
```
sonketsarkar_2511451_part3_regression_insights/
├── data/
│   └── regression_data.csv
├── scripts/
│   └── regression_analysis.py
├── outputs/
│   ├── models/
│   │   ├── dataset_summary.csv
│   │   ├── simple_regression_results.csv
│   │   ├── multiple_regression_coefficients.csv
│   │   ├── multiple_regression_metrics.json
│   │   ├── model_comparison.csv
│   │   ├── residuals.csv
│   │   ├── residual_analysis.json
│   │   ├── model_equation.txt
│   │   ├── final_recommendation.txt
│   │   └── regression_analysis.xlsx
│   └── visualizations/
│       └── (chart exports)
├── screenshots/
│   ├── screenshot_01_dataset_overview.png
│   ├── screenshot_02_simple_regression.png
│   ├── screenshot_03_multiple_regression.png
│   ├── screenshot_04_model_comparison.png
│   └── screenshot_05_residual_analysis.png
└── README.md
```

## How to Run
```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn scipy openpyxl

# 2. Place regression_data.csv in data/ folder
#    (or the script generates synthetic demo data automatically)

# 3. Run the analysis
python scripts/regression_analysis.py
```

Outputs will be generated in `outputs/models/` and `outputs/visualizations/`.
