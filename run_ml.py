import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, os, json, pickle
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

OUT = '/home/claude/house_project/static'
os.makedirs(OUT, exist_ok=True)

print('Loading data...')
df = pd.read_csv('/mnt/user-data/uploads/pyhouse.csv')
print(f'Shape: {df.shape}')

df['Amenity_Count'] = df['Amenities'].str.split(',').str.len()
df['Has_Pool'] = df['Amenities'].str.contains('Pool').astype(int)
df['Has_Gym']  = df['Amenities'].str.contains('Gym').astype(int)
df['Has_Garden']= df['Amenities'].str.contains('Garden').astype(int)
df['Property_Age'] = 2024 - df['Year_Built']
df['Floor_Ratio']  = df['Floor_No'] / df['Total_Floors'].replace(0,1)

cat_cols = ['State','City','Property_Type','Furnished_Status',
            'Public_Transport_Accessibility','Parking_Space','Facing','Owner_Type']
encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    df[c+'_enc'] = le.fit_transform(df[c].astype(str))
    encoders[c] = le

features = ['BHK','Size_in_SqFt','Property_Age','Floor_No','Total_Floors',
            'Floor_Ratio','Amenity_Count','Has_Pool','Has_Gym','Has_Garden',
            'State_enc','City_enc','Property_Type_enc','Furnished_Status_enc',
            'Public_Transport_Accessibility_enc','Parking_Space_enc',
            'Facing_enc','Owner_Type_enc']

X = df[features]; y = df['Price_in_Lakhs']
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
print(f'Train={len(Xtr)} Test={len(Xte)}')

print('Training Linear Regression...')
lr = LinearRegression(); lr.fit(Xtr,ytr); lp = lr.predict(Xte)
print('Training Random Forest...')
rf = RandomForestRegressor(100,random_state=42,n_jobs=-1); rf.fit(Xtr,ytr); rp = rf.predict(Xte)
print('Training Gradient Boosting...')
gb = GradientBoostingRegressor(100,random_state=42); gb.fit(Xtr,ytr); gp = gb.predict(Xte)

res = {}
for nm,pred,mdl in [('Linear Regression',lp,lr),('Random Forest',rp,rf),('Gradient Boosting',gp,gb)]:
    mae=mean_absolute_error(yte,pred)
    rmse=np.sqrt(mean_squared_error(yte,pred))
    r2=r2_score(yte,pred)
    res[nm]={'mae':round(mae,2),'rmse':round(rmse,2),'r2':round(r2,4),'pred':pred,'mdl':mdl}
    print(f'{nm}: MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.4f}')

best_nm = max(res,key=lambda k:res[k]['r2'])
best_mdl = res[best_nm]['mdl']
best_pred = res[best_nm]['pred']
print(f'Best: {best_nm}')
print(f'Best: {best_nm}')

# ── Z-TEST ──────────────────────────────────────────────────
# H0: mean prediction error = 0  (model is unbiased)
# H1: mean prediction error != 0  (two-tailed)
print('\n--- Z-Test Results ---')
z_results = {}
for nm, info in res.items():
    errors   = yte.values - info['pred']
    n        = len(errors)
    mean_err = errors.mean()
    std_err  = errors.std(ddof=1)
    se       = std_err / np.sqrt(n)
    z_stat   = mean_err / se
    p_value  = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    ci_low   = mean_err - 1.96 * se
    ci_high  = mean_err + 1.96 * se
    reject_h0  = bool(p_value < 0.05)
    conclusion = ("Reject H0 - significant bias" if reject_h0
                  else "Fail to reject H0 - no significant bias")
    z_results[nm] = {
        'n'             : int(n),
        'mean_error'    : round(float(mean_err), 4),
        'std_error'     : round(float(std_err),  4),
        'standard_error': round(float(se),        6),
        'z_statistic'   : round(float(z_stat),    4),
        'p_value'       : round(float(p_value),   6),
        'ci_95'         : [round(float(ci_low), 4), round(float(ci_high), 4)],
        'reject_h0'     : reject_h0,
        'conclusion'    : conclusion,
    }
    print(f'{nm}: Z={z_stat:.4f}  p={p_value:.6f}  => {conclusion}')

# Z-test error distribution chart
best_errors = yte.values - best_pred
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(best_errors, bins=60, color='#4361ee', edgecolor='white', alpha=0.80, density=True)
xmin, xmax = ax.get_xlim()
x = np.linspace(xmin, xmax, 300)
mu, sigma = best_errors.mean(), best_errors.std()
ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, label=f'Normal fit  mean={mu:.2f}')
ax.axvline(0, color='k', ls='--', lw=1.5, label='Zero bias line')
ax.set_xlabel('Prediction Error (Actual - Predicted, Lakhs)')
ax.set_ylabel('Density')
ax.set_title(f'Z-Test: Error Distribution — {best_nm}')
ax.legend()
fig.tight_layout()
fig.savefig(f'{OUT}/ztest_error_dist.png')
plt.close()
print('Z-test chart saved')

with open('/home/claude/house_project/model.pkl','wb') as f:
    pickle.dump({'model':best_mdl,'encoders':encoders,'features':features},f)
with open('/home/claude/house_project/model.pkl','wb') as f:
    pickle.dump({'model':best_mdl,'encoders':encoders,'features':features},f)

sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 120

fig,ax=plt.subplots(figsize=(7,5))
ax.scatter(yte.values[:3000],best_pred[:3000],alpha=0.3,s=8,color='#4361ee')
mn,mx=yte.min(),yte.max(); ax.plot([mn,mx],[mn,mx],'r--',lw=1.5,label='Ideal')
ax.set_xlabel('Actual Price (Lakhs)'); ax.set_ylabel('Predicted Price (Lakhs)')
ax.set_title('Actual vs Predicted'); ax.legend()
fig.tight_layout(); fig.savefig(f'{OUT}/actual_vs_pred.png'); plt.close()
print('Chart 1 done')

residuals=yte.values-best_pred
fig,ax=plt.subplots(figsize=(7,5))
ax.scatter(best_pred[:3000],residuals[:3000],alpha=0.3,s=8,color='#f72585')
ax.axhline(0,color='k',lw=1.5,ls='--')
ax.set_xlabel('Predicted'); ax.set_ylabel('Residual'); ax.set_title('Residual Plot')
fig.tight_layout(); fig.savefig(f'{OUT}/residuals.png'); plt.close()
print('Chart 2 done')

fi=pd.Series(rf.feature_importances_,index=features).sort_values(ascending=True).tail(12)
fig,ax=plt.subplots(figsize=(7,5))
fi.plot(kind='barh',ax=ax,color='#4cc9f0')
ax.set_title('Feature Importance (Random Forest)'); ax.set_xlabel('Importance')
fig.tight_layout(); fig.savefig(f'{OUT}/feature_importance.png'); plt.close()
print('Chart 3 done')

fig,ax=plt.subplots(figsize=(7,5))
ax.hist(df['Price_in_Lakhs'],bins=60,color='#7209b7',edgecolor='white',alpha=0.85)
ax.set_xlabel('Price (Lakhs)'); ax.set_ylabel('Count'); ax.set_title('Price Distribution')
fig.tight_layout(); fig.savefig(f'{OUT}/price_dist.png'); plt.close()
print('Chart 4 done')

fig,ax=plt.subplots(figsize=(7,5))
df.groupby('BHK')['Price_in_Lakhs'].mean().plot(kind='bar',ax=ax,color='#06d6a0',edgecolor='white')
ax.set_title('Avg Price by BHK'); ax.set_xlabel('BHK'); ax.set_ylabel('Avg Price (Lakhs)')
ax.tick_params(axis='x',rotation=0)
fig.tight_layout(); fig.savefig(f'{OUT}/price_by_bhk.png'); plt.close()
print('Chart 5 done')

fig,ax=plt.subplots(figsize=(7,5))
df.groupby('Property_Type')['Price_in_Lakhs'].mean().sort_values().plot(kind='barh',ax=ax,color='#ff6b6b',edgecolor='white')
ax.set_title('Avg Price by Property Type'); ax.set_xlabel('Avg Price (Lakhs)')
fig.tight_layout(); fig.savefig(f'{OUT}/price_by_type.png'); plt.close()
print('Chart 6 done')

num_cols=['BHK','Size_in_SqFt','Price_in_Lakhs','Property_Age','Floor_No','Amenity_Count']
fig,ax=plt.subplots(figsize=(7,5))
sns.heatmap(df[num_cols].corr(),annot=True,fmt='.2f',cmap='coolwarm',ax=ax,square=True,linewidths=.5)
ax.set_title('Correlation Heatmap')
fig.tight_layout(); fig.savefig(f'{OUT}/heatmap.png'); plt.close()
print('Chart 7 done')

nms=list(res.keys()); r2s=[res[n]['r2'] for n in nms]
fig,ax=plt.subplots(figsize=(7,4))
bars=ax.bar(nms,r2s,color=['#4361ee','#f72585','#06d6a0'],edgecolor='white')
for bar,v in zip(bars,r2s):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.005,f'{v:.4f}',ha='center',fontsize=10)
ax.set_ylim(0,1.1); ax.set_ylabel('R² Score'); ax.set_title('Model Comparison (R²)')
fig.tight_layout(); fig.savefig(f'{OUT}/model_comparison.png'); plt.close()
print('Chart 8 done')

ts=df.groupby('State')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(10)
fig,ax=plt.subplots(figsize=(8,5))
ts.plot(kind='bar',ax=ax,color='#f4a261',edgecolor='white')
ax.set_ylabel('Avg Price (Lakhs)'); ax.set_title('Top 10 States by Avg Price')
ax.tick_params(axis='x',rotation=45)
fig.tight_layout(); fig.savefig(f'{OUT}/price_by_state.png'); plt.close()
print('Chart 9 done')

metrics = {nm:{k:v for k,v in r.items() if k not in ('pred','mdl')} for nm,r in res.items()}
metrics['best_model']=best_nm
metrics['best_model']=best_nm
metrics['z_test'] = z_results    # ← ADD THIS LINE
metrics['dataset']={'rows':len(df),'cols':df.shape[1]}
metrics['price_stats']={'min':round(df['Price_in_Lakhs'].min(),2),'max':round(df['Price_in_Lakhs'].max(),2),
    'mean':round(df['Price_in_Lakhs'].mean(),2),'median':round(df['Price_in_Lakhs'].median(),2)}
metrics['options']={
    'states':sorted(df['State'].unique().tolist()),
    'cities':sorted(df['City'].unique().tolist()),
    'property_types':df['Property_Type'].unique().tolist(),
    'furnished':df['Furnished_Status'].unique().tolist(),
    'transport':df['Public_Transport_Accessibility'].unique().tolist(),
    'facing':df['Facing'].unique().tolist(),
    'owner':df['Owner_Type'].unique().tolist(),
}
with open('/home/claude/house_project/metrics.json','w') as f:
    json.dump(metrics,f)
print('ALL DONE')
