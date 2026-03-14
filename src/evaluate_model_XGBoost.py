import pandas as pd
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import shap
import matplotlib.pyplot as plt

# chemin vers data
data_dir = Path(__file__).resolve().parent.parent / "data"

# charge modèle XGBoost déjà entraîné
model = xgb.XGBClassifier()
model.load_model(data_dir / "xgboost_model.json")

# charge ensemble test
X_test = pd.read_csv(data_dir / "X_test_cleaned.csv")
y_test = pd.read_csv(data_dir / "y_test_cleaned.csv").squeeze()

# evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# SHAP : importance des features (bar chart)
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)   # nouvelle API SHAP
shap.plots.bar(shap_values, max_display=10, show=False)  # top 10 features
plt.tight_layout()
plt.savefig(data_dir / "shap_summary_bar_plot.png")
plt.close()

print("SHAP chart saved:", data_dir / "shap_summary_bar_plot.png")