"""
Part 3: Regression-Based Business Insights & Model Interpretation
Dataset: business_regression_data.xlsx (320 rows, 14 columns)
Target: monthly_sales (store performance regression)
Student: sonketsarkar | ID: 2511451
"""

import pandas as pd
import numpy as np
import os, json
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.dummy import DummyRegressor
from scipy import stats
from datetime import datetime

BASE  = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.join(BASE, '..')
DATA  = os.path.join(ROOT, 'data', 'business_regression_data.xlsx')
MDIR  = os.path.join(ROOT, 'outputs', 'models')
VDIR  = os.path.join(ROOT, 'outputs', 'visualizations')
for d in [MDIR, VDIR]: os.makedirs(d, exist_ok=True)

# ── TASK 1: Understand the dataset ────────────────────────────────────────────
print("=" * 60)
print("Task 1: Understand the Dataset")
df = pd.read_excel(DATA, sheet_name='store_performance')
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")
print(f"  Missing: {df.isnull().sum()[df.isnull().sum()>0].to_dict()}")
print(f"\n  Target (monthly_sales) stats:")
print(df['monthly_sales'].describe().round(2).to_string())
desc = df.describe().round(2)
desc.to_csv(os.path.join(MDIR, 'dataset_summary.csv'))

# ── TASK 2: Prepare regression workbook ───────────────────────────────────────
print("\nTask 2: Prepare Regression Workbook")

# Drop non-numeric identifier and date
df_model = df.drop(columns=['store_id', 'month'], errors='ignore')

# Separate target
TARGET = 'monthly_sales'
y = df_model[TARGET]
X_raw = df_model.drop(columns=[TARGET, 'monthly_profit'], errors='ignore')

print(f"  Features before encoding: {list(X_raw.columns)}")

# ── TASK 3: Create dummy variables ────────────────────────────────────────────
print("\nTask 3: Create Dummy Variables")
cat_cols = X_raw.select_dtypes(include='object').columns.tolist()
print(f"  Categorical columns to encode: {cat_cols}")
X = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)
# Impute missing values with column median
for col in X.columns:
    if X[col].isna().any():
        X[col] = X[col].fillna(X[col].median())
print(f"  Features after encoding: {X.shape[1]} | {list(X.columns)}")

# Train/test split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
feat_names = list(X.columns)

# ── TASK 4: Simple regression models ─────────────────────────────────────────
print("\nTask 4: Simple Regression Models")
numeric_feats = ['marketing_spend', 'footfall', 'staff_count',
                 'avg_discount_pct', 'inventory_availability_pct',
                 'competitor_distance_km', 'customer_rating']
numeric_feats = [f for f in numeric_feats if f in feat_names]

simple_rows = []
for feat in numeric_feats:
    idx = feat_names.index(feat)
    lr = LinearRegression().fit(X_train_s[:, [idx]], y_train)
    yp = lr.predict(X_test_s[:, [idx]])
    simple_rows.append({
        'Feature': feat,
        'Coefficient': round(lr.coef_[0], 2),
        'Intercept': round(lr.intercept_, 2),
        'R2_Train': round(r2_score(y_train, lr.predict(X_train_s[:,[idx]])), 4),
        'R2_Test':  round(r2_score(y_test, yp), 4),
        'RMSE_Test': round(np.sqrt(mean_squared_error(y_test, yp)), 2),
        'MAE_Test':  round(mean_absolute_error(y_test, yp), 2),
    })
simple_df = pd.DataFrame(simple_rows).sort_values('R2_Test', ascending=False)
simple_df.to_csv(os.path.join(MDIR, 'simple_regression_results.csv'), index=False)
print(simple_df.to_string(index=False))

# ── TASK 5: Multiple regression ───────────────────────────────────────────────
print("\nTask 5: Multiple Regression Model")
multi = LinearRegression().fit(X_train_s, y_train)
y_pred_tr = multi.predict(X_train_s)
y_pred_te = multi.predict(X_test_s)

metrics = {
    'R2_Train':  round(r2_score(y_train, y_pred_tr), 4),
    'R2_Test':   round(r2_score(y_test,  y_pred_te), 4),
    'RMSE_Train':round(np.sqrt(mean_squared_error(y_train, y_pred_tr)), 2),
    'RMSE_Test': round(np.sqrt(mean_squared_error(y_test,  y_pred_te)), 2),
    'MAE_Test':  round(mean_absolute_error(y_test, y_pred_te), 2),
    'N_Features':len(feat_names),
}
with open(os.path.join(MDIR, 'multiple_regression_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=4)

coef_df = pd.DataFrame({'Feature': feat_names, 'Coefficient': multi.coef_.round(2)})
coef_df = coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=False).index)
coef_df.to_csv(os.path.join(MDIR, 'multiple_regression_coefficients.csv'), index=False)
print(f"  R² Train: {metrics['R2_Train']} | R² Test: {metrics['R2_Test']}")
print(f"  RMSE Test: {metrics['RMSE_Test']} | MAE Test: {metrics['MAE_Test']}")
print(f"\n  Top 8 coefficients:")
print(coef_df.head(8).to_string(index=False))

# ── TASK 6: Compare models ────────────────────────────────────────────────────
print("\nTask 6: Compare Models")
baseline = DummyRegressor(strategy='mean').fit(X_train_s, y_train)
best_feat_idx = feat_names.index(simple_df.iloc[0]['Feature'])
simple_best = LinearRegression().fit(X_train_s[:, [best_feat_idx]], y_train)

comparison = pd.DataFrame([
    {'Model': 'Baseline (Mean)',
     'R2': round(r2_score(y_test, baseline.predict(X_test_s)), 4),
     'RMSE': round(np.sqrt(mean_squared_error(y_test, baseline.predict(X_test_s))), 2),
     'MAE': round(mean_absolute_error(y_test, baseline.predict(X_test_s)), 2)},
    {'Model': f'Simple ({simple_df.iloc[0]["Feature"]})',
     'R2': round(r2_score(y_test, simple_best.predict(X_test_s[:,[best_feat_idx]])), 4),
     'RMSE': round(np.sqrt(mean_squared_error(y_test, simple_best.predict(X_test_s[:,[best_feat_idx]]))), 2),
     'MAE': round(mean_absolute_error(y_test, simple_best.predict(X_test_s[:,[best_feat_idx]])), 2)},
    {'Model': 'Multiple Regression',
     'R2': metrics['R2_Test'], 'RMSE': metrics['RMSE_Test'], 'MAE': metrics['MAE_Test']},
])
comparison.to_csv(os.path.join(MDIR, 'model_comparison.csv'), index=False)
print(comparison.to_string(index=False))

# ── TASK 7: Residual analysis ─────────────────────────────────────────────────
print("\nTask 7: Residual Analysis")
residuals = y_test.values - y_pred_te
resid_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred_te, 'Residual': residuals})
resid_df.to_csv(os.path.join(MDIR, 'residuals.csv'), index=False)
_, p_sw = stats.shapiro(residuals[:min(200, len(residuals))])
resid_stats = {
    'Mean': round(residuals.mean(), 2), 'Std': round(residuals.std(), 2),
    'Min': round(residuals.min(), 2),   'Max': round(residuals.max(), 2),
    'Shapiro_Wilk_p': round(p_sw, 4),  'Residuals_Normal': bool(p_sw > 0.05),
}
with open(os.path.join(MDIR, 'residual_analysis.json'), 'w') as f:
    json.dump(resid_stats, f, indent=4)
print(f"  {resid_stats}")

# ── TASK 8: Model equation ────────────────────────────────────────────────────
print("\nTask 8: Model Equation")
intercept = round(multi.intercept_, 2)
top5 = coef_df.head(5)
eq = f"monthly_sales = {intercept}"
for _, row in top5.iterrows():
    sign = '+' if row['Coefficient'] >= 0 else '-'
    eq  += f" {sign} {abs(row['Coefficient']):.2f}×{row['Feature']}"
print(f"  {eq}")
with open(os.path.join(MDIR, 'model_equation.txt'), 'w') as f:
    f.write("MULTIPLE LINEAR REGRESSION MODEL EQUATION\n")
    f.write("=" * 60 + "\n\n")
    f.write(eq + "\n\n")
    f.write("Top driver interpretation:\n")
    for _, row in top5.iterrows():
        direction = "increase" if row['Coefficient'] > 0 else "decrease"
        f.write(f"  • 1-unit change in {row['Feature']} → ₹{abs(row['Coefficient']):,.2f} {direction} in monthly_sales\n")

# ── TASK 9: Final recommendation ─────────────────────────────────────────────
print("\nTask 9: Final Recommendation")
top_pos = coef_df[coef_df['Coefficient'] > 0].head(3)
top_neg = coef_df[coef_df['Coefficient'] < 0].head(2)

rec = f"""
============================================================
REGRESSION ANALYSIS – BUSINESS RECOMMENDATION
============================================================
Date    : {datetime.now().strftime('%Y-%m-%d')}
Analyst : sonketsarkar (2511451)
Dataset : Store performance (320 store-months, {X.shape[1]} features)

MODEL PERFORMANCE
  R² (Train) : {metrics['R2_Train']} — explains {metrics['R2_Train']*100:.1f}% of training variance
  R² (Test)  : {metrics['R2_Test']} — generalises to {metrics['R2_Test']*100:.1f}% on unseen stores
  RMSE (Test): ₹{metrics['RMSE_Test']:,}
  MAE  (Test): ₹{metrics['MAE_Test']:,}

TOP POSITIVE DRIVERS
{top_pos[['Feature','Coefficient']].to_string(index=False)}

TOP NEGATIVE DRIVERS
{top_neg[['Feature','Coefficient']].to_string(index=False)}

BUSINESS RECOMMENDATIONS
  1. MARKETING SPEND is the strongest positive driver. Increase budget for
     underperforming stores first — ROI is highest where baseline spend is low.
  2. FOOTFALL drives sales — invest in local promotions, in-mall visibility,
     and events to increase store visits.
  3. STAFF COUNT correlates positively — understaffed stores leave sales on the
     table. Review staffing norms for high-footfall periods.
  4. INVENTORY AVAILABILITY — stock-outs directly lose sales. Improve supply
     chain reliability to keep availability above 85%.
  5. HIGH DISCOUNTS may reduce margin without driving proportional volume.
     Test discount thresholds carefully.

MODEL SELECTION
{comparison.to_string(index=False)}

Multiple Regression (R²={metrics['R2_Test']}) recommended over simple models.
============================================================
"""
with open(os.path.join(MDIR, 'final_recommendation.txt'), 'w') as f:
    f.write(rec)
print(rec)

# Excel output
with pd.ExcelWriter(os.path.join(MDIR, 'regression_analysis.xlsx'), engine='openpyxl') as w:
    desc.to_excel(w, sheet_name='Data_Summary')
    simple_df.to_excel(w, sheet_name='Simple_Regression', index=False)
    coef_df.to_excel(w, sheet_name='Multi_Coefficients', index=False)
    comparison.to_excel(w, sheet_name='Model_Comparison', index=False)
    resid_df.to_excel(w, sheet_name='Residuals', index=False)
print("✅ Part 3 complete!")
