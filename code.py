import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import PartialDependenceDisplay
from sklearn.pipeline import Pipeline

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

# ---------------------------------------------------------
# 1. Data Processing & Cleaning
# ---------------------------------------------------------
print("Loading data...")
df = pd.read_excel('Data, Nu (2)Final.xlsx', sheet_name='NU', header=1)
df['Reference'] = df['Unnamed: 0'].ffill()

features = [
    'Flow type', 'Base fluid', 'Base fluid thermal Conductivity (W/m.K)', 
    'Nanoparticle thermal Conductivity (W/m.K)', 'Size of nano particles (nm)', 
    'Reynolds number (No units)', 'Inlet temperature range (oC)', 
    'Concentration of nanoparticles (Vol %)'
]
target = 'Nusselt number'

df = df[['Reference'] + features + [target]].dropna()
df['Flow type'] = df['Flow type'].astype(str).str.strip().str.lower().replace({'laminor': 'laminar'})
df['Base fluid'] = df['Base fluid'].astype(str).str.strip().str.lower()

# Filter physical data entry typos
df = df[df['Base fluid thermal Conductivity (W/m.K)'] < 2.0]

# ---------------------------------------------------------
# 2. Train/Test Split (For Descriptive Analysis)
# ---------------------------------------------------------
X_physical = df[features]
y_physical = df[target]
y_log = np.log1p(df[target]) 

# 75% Train / 25% Test Split
X_train_phys, X_test_phys, y_train_phys, y_test_phys = train_test_split(
    X_physical, y_physical, test_size=0.25, random_state=42
)

numeric_cols = [
    'Base fluid thermal Conductivity (W/m.K)', 
    'Nanoparticle thermal Conductivity (W/m.K)', 'Size of nano particles (nm)', 
    'Reynolds number (No units)', 'Inlet temperature range (oC)', 
    'Concentration of nanoparticles (Vol %)'
]

def generate_stats(X_subset, y_subset):
    temp_df = X_subset[numeric_cols].copy()
    temp_df['Nusselt number'] = y_subset
    stats = pd.DataFrame()
    stats['Mean'] = temp_df.mean()
    stats['Standard Deviation'] = temp_df.std()
    stats['Minimum'] = temp_df.min()
    stats['Maximum'] = temp_df.max()
    stats['Skewness'] = temp_df.skew()
    return stats.round(2)

train_stats = generate_stats(X_train_phys, y_train_phys)
test_stats = generate_stats(X_test_phys, y_test_phys)

print(f"\nTotal experimental samples: {len(df)}")
print(f"Training samples (75%): {len(X_train_phys)} | Testing samples (25%): {len(X_test_phys)}\n")

# ---------------------------------------------------------
# 3. EDA Plots
# ---------------------------------------------------------
print("Generating Violin Plots...")
bright_palette = sns.color_palette("husl", len(features))

# Plot 1: Strictly Linear Scale
plt.figure(figsize=(15, 10))
for i, col in enumerate(features):
    ax = plt.subplot(3, 3, i+1)
    sns.violinplot(y=df[col], color=bright_palette[i], inner="box", ax=ax)
    ax.set_ylabel("Values")
    plt.title(col, fontsize=10)
plt.tight_layout()
plt.savefig('1a_Violin_Plots_Linear.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Log Scale
plt.figure(figsize=(15, 10))
for i, col in enumerate(features):
    ax = plt.subplot(3, 3, i+1)
    sns.violinplot(y=df[col], color=bright_palette[i], inner="box", ax=ax)
    if col in ['Nanoparticle thermal Conductivity (W/m.K)', 'Reynolds number (No units)', 'Concentration of nanoparticles (Vol %)']:
        ax.set_yscale('log')
        ax.set_ylabel("Values (Log Scale)")
    else:
        ax.set_ylabel("Values")
    plt.title(col, fontsize=10)
plt.tight_layout()
plt.savefig('1b_Violin_Plots_Log.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generating Correlation Heatmap...")
plt.figure(figsize=(10, 8))
corr_cols = [target] + numeric_cols
corr = df[corr_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
plt.title('Pearson Correlation Coefficient Analysis')
plt.savefig('2_Correlation_Heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# 4. Model Preparation
# ---------------------------------------------------------
df_model = pd.get_dummies(df, columns=['Flow type', 'Base fluid'], drop_first=True)

X = df_model.drop(columns=[target, 'Reference']).astype('float64')

clean_columns = []
for col in X.columns:
    c = col.replace('Flow type_', 'Flow: ').replace('Base fluid_', 'Base Fluid: ').title()
    c = c.replace('(W/M.K)', '(W/m.K)').replace('(Nm)', '(nm)').replace('(Oc)', '(oC)').replace('(Vol %)', '(Vol %)')
    c = c.replace('Size Of Nano Particles', 'Size of nanoparticles')
    clean_columns.append(c)
X.columns = clean_columns

X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_all_scaled = scaler.transform(X) 

# ---------------------------------------------------------
# 5. Model Training 
# ---------------------------------------------------------
print("Training Models and Calculating Metrics...")
models = {
    'CTB': CatBoostRegressor(depth=6, learning_rate=0.05, l2_leaf_reg=5, verbose=0, random_state=42),
    'XGB': XGBRegressor(max_depth=6, learning_rate=0.05, n_estimators=200, subsample=0.8, random_state=42),
    'LGB': LGBMRegressor(max_depth=6, learning_rate=0.05, n_estimators=200, min_child_samples=10, verbose=-1, random_state=42),
    'RF': RandomForestRegressor(max_depth=12, min_samples_leaf=1, n_estimators=200, random_state=42),
    'Bagging': BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=12), n_estimators=100, random_state=42),
    'ADB': AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=12), learning_rate=0.01, n_estimators=150, random_state=42),
    'DTB': DecisionTreeRegressor(max_depth=12, min_samples_leaf=2, random_state=42),
    'ANN': MLPRegressor(hidden_layer_sizes=(100, 50), alpha=0.01, max_iter=3000, random_state=42),
    'kNN': KNeighborsRegressor(n_neighbors=2, weights='distance')
}

y_train_orig = np.expm1(y_train_log)
y_test_orig = np.expm1(y_test_log)
y_all_orig = np.expm1(y_log)

performance_metrics = []
df_detailed = df[['Reference']].copy()
df_detailed['Actual Nusselt Number'] = y_all_orig.values
for col in X.columns:
    df_detailed[col] = X[col].values

plt.figure(figsize=(18, 18))
for i, (name, model) in enumerate(models.items()):
    model.fit(X_train_scaled, y_train_log)
    
    pred_train = np.expm1(np.clip(model.predict(X_train_scaled), 0, 20))
    pred_test = np.expm1(np.clip(model.predict(X_test_scaled), 0, 20))
    pred_all = np.expm1(np.clip(model.predict(X_all_scaled), 0, 20))
    
    df_detailed[f'{name} Predicted'] = pred_all
    df_detailed[f'{name} Absolute Error'] = np.abs(df_detailed['Actual Nusselt Number'] - pred_all)
    
    r2_train = r2_score(y_train_orig, pred_train) * 100
    rmse_train = np.sqrt(mean_squared_error(y_train_orig, pred_train))
    mae_train = mean_absolute_error(y_train_orig, pred_train)
    
    r2_test = r2_score(y_test_orig, pred_test) * 100
    rmse_test = np.sqrt(mean_squared_error(y_test_orig, pred_test))
    mae_test = mean_absolute_error(y_test_orig, pred_test)
    
    performance_metrics.append({
        'Model Used': name, 'Dataset': 'Training set', 
        'R2 (%) value': round(r2_train, 2), 'RMSE value': round(rmse_train, 2), 'MAE value': round(mae_train, 2)
    })
    performance_metrics.append({
        'Model Used': name, 'Dataset': 'Testing set', 
        'R2 (%) value': round(r2_test, 2), 'RMSE value': round(rmse_test, 2), 'MAE value': round(mae_test, 2)
    })
    
    plt.subplot(3, 3, i+1)
    plt.scatter(y_train_orig, pred_train, color='blue', label='Training set', s=12, alpha=0.7)
    plt.scatter(y_test_orig, pred_test, color='red', label='Testing set', s=12, alpha=0.7)
    min_val, max_val = df[target].min(), df[target].max()
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1)
    plt.xlabel('Actual Nusselt number')
    plt.ylabel('Predicted Nusselt number')
    plt.title(f'{name} Model Predictions')
    plt.text(0.55, 0.1, f'$R^2$ (Train) = {r2_train/100:.2%}\n$R^2$ (Test) = {r2_test/100:.2%}', 
             transform=plt.gca().transAxes, color='blue', fontsize=10)
    plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig('3_Actual_vs_Predicted.png', dpi=300, bbox_inches='tight')
plt.close()

performance_df = pd.DataFrame(performance_metrics)

# ---------------------------------------------------------
# 6. Export Results to Excel
# ---------------------------------------------------------
print("Exporting statistics and metrics to 'Publication_Results_Output.xlsx'...")
with pd.ExcelWriter('Publication_Results_Output.xlsx') as writer:
    train_stats.to_excel(writer, sheet_name='Train_Descriptive_Stats')
    test_stats.to_excel(writer, sheet_name='Test_Descriptive_Stats')
    performance_df.to_excel(writer, sheet_name='Model_Performance_Metrics', index=False)
    df_detailed.to_excel(writer, sheet_name='Detailed_Predictions', index=False)

# ---------------------------------------------------------
# 7. SHAP Analysis
# ---------------------------------------------------------
print("Generating SHAP Analysis...")
xgb_model = models['XGB']
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_train_scaled)
shap_values.data = X_train.values 
shap_values.feature_names = X.columns.tolist()

plt.figure(figsize=(10, 6))
shap.plots.bar(shap_values, max_display=10, show=False)
plt.title("(a) SHAP Bar plot analysis (Log-Transformed Impact)")
plt.savefig('4_SHAP_Bar.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
shap.plots.beeswarm(shap_values, max_display=10, show=False)
plt.title("(b) SHAP Beeswarm analysis (Log-Transformed Impact)")
plt.savefig('5_SHAP_Beeswarm.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# 8. Partial Dependence Plots (Updated to 2x2 Grid)
# ---------------------------------------------------------
print("Generating Partial Dependence Plots (2x2 Grid)...")
pipeline = Pipeline([('scaler', scaler), ('xgb', xgb_model)])
pdp_features = ['Base Fluid Thermal Conductivity (W/m.K)', 
                'Nanoparticle Thermal Conductivity (W/m.K)',
                'Reynolds Number (No Units)', 
                'Inlet Temperature Range (oC)']
pdp_indices = [X.columns.get_loc(col) for col in pdp_features]

# Create a 2x2 grid framework with proportional sizing
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes_flat = axes.flatten()
X_train_pdp = X_train.astype('float64')

# Generate the plots onto the flattened 2x2 axes
display = PartialDependenceDisplay.from_estimator(
    pipeline, X_train_pdp, features=pdp_indices, feature_names=X_train_pdp.columns.tolist(), ax=axes_flat
)

# Customize titles and labels for publication
for axis, feature_name in zip(axes_flat, pdp_features):
    axis.set_title(f"PDP: {feature_name}", fontsize=11, weight='bold')
    axis.set_ylabel("Partial Dependence (Log(Nu))", fontsize=10)
    axis.set_xlabel(feature_name, fontsize=10)
    axis.grid(alpha=0.3)

plt.suptitle('One-way Partial Dependence Plots (Original Physical Scale)', y=1.02, fontsize=16, weight='bold')
plt.tight_layout()

# Save with High DPI (400) for sharp journal publication quality
plt.savefig('6_PDP_Plots_2x2.png', dpi=400, bbox_inches='tight')
plt.close()

print("Script execution completed successfully!")