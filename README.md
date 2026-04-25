# 🏠 Indian House Price ML Prediction System
**CA Project | Python & Machine Learning**

---

## 📌 Project Overview
A complete Machine Learning web application that:
- Loads and analyses **250,000 Indian property listings**
- Performs **Feature Engineering** (Property Age, Floor Ratio, Amenity Flags)
- Trains **4 ML models** and compares them
- Serves a **Flask web dashboard** with EDA charts and live price prediction

---

## 🗂️ Project Structure
```
house_project/
├── app.py              ← Flask web server (run this)
├── run_ml.py           ← ML training pipeline script
├── model.pkl           ← Saved trained model (auto-generated)
├── metrics.json        ← Model metrics & dataset stats
├── state_city.json     ← State → City mapping
├── requirements.txt    ← Python dependencies
├── templates/
│   └── index.html      ← Full dashboard HTML
└── static/
    ├── price_dist.png
    ├── model_comparison.png
    ├── actual_vs_pred.png
    ├── feature_importance.png
    └── ...             ← All 13 charts
```

---

## ⚙️ Setup & Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — (Optional) Re-run ML training
```bash
python run_ml.py
```

### Step 3 — Start the web server
```bash
python app.py
```

### Step 4 — Open in browser
```
http://127.0.0.1:5000
```

---

## 🤖 ML Pipeline

| Step | Details |
|------|---------|
| **Data Loading** | `pandas.read_csv()` — 250,000 rows × 17 columns |
| **Feature Engineering** | Property Age, Floor Ratio, Amenity Count, Has Pool/Gym/Garden |
| **Encoding** | `LabelEncoder` on 8 categorical columns |
| **Train/Test Split** | 80% train (200K) / 20% test (50K), `random_state=42` |
| **Models Trained** | Linear Regression, Ridge, Decision Tree, Gradient Boosting |
| **Evaluation** | MAE, RMSE, R² on held-out test set |
| **Saved** | Best model serialised with `pickle` |

---

## 📊 Dashboard Tabs

| Tab | Content |
|-----|---------|
| **Overview** | Hero stats, price distribution, property type chart |
| **EDA** | State counts, furnished status, transport, amenities, heatmap |
| **ML Models** | Comparison chart, metrics table, actual-vs-predicted, residuals, feature importance |
| **Predict** | Live form — enter property details, get instant price estimate |
| **Pipeline** | Visual flow, code snippet, library list |

---

## 📚 Libraries Used
- `pandas` — data loading & manipulation
- `numpy` — numerical operations
- `scikit-learn` — ML models (LinearRegression, Ridge, DecisionTree, GradientBoosting), LabelEncoder, train_test_split, metrics
- `matplotlib` — chart rendering
- `seaborn` — statistical visualisations (heatmap, boxplot)
- `Flask` — web server & REST API
- `pickle` — model serialisation

---

## ⚠️ Dataset Note
The `pyhouse.csv` dataset contains synthetically generated `Price_in_Lakhs` values with no statistical relationship to the property features (correlation ≈ 0.001). This is typical of demo/synthetic datasets. The ML pipeline is correctly implemented — the Prediction tab uses a **rule-based market estimator** grounded in real Indian property pricing logic (location multiplier, property type, furnishing, age, transport access) alongside the ML model output.

---

*Built with Python, scikit-learn, Flask | CA Project 2024*
