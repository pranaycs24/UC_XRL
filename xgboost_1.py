# Importing necessary packages
import pdb
import xgboost as xgb
import shap
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Loading the abalone dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
columns = ["Sex", "Length", "Diameter", "Height", "WholeWeight",
		"ShuckedWeight", "VisceraWeight", "ShellWeight", "Rings"]
abalone_data = pd.read_csv(url, header=None, names=columns)

# Data preprocessing and feature engineering
# Assuming you want to predict the number of rings, which is a continuous target variable
X = abalone_data.drop("Rings", axis=1)
y = abalone_data["Rings"]

# Convert categorical feature 'Sex' to numerical using one-hot encoding
X = pd.get_dummies(X, columns=["Sex"], drop_first=True)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42)

# Creating an XGBRegressor model
model = xgb.XGBRegressor()
model.fit(X_train, y_train)
pdb.set_trace()
# Save the XGBoost model in binary format
model.save_model('model.json')
pdb.set_trace()

# Load the model from the saved binary file
loaded_model = xgb.XGBRegressor()
loaded_model.load_model('model.json')

pdb.set_trace()

# SHAP Explainer
explainer = shap.Explainer(loaded_model)
# pdb.set_trace()
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)
plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
# Initialize the SHAP JavaScript library
shap.initjs()