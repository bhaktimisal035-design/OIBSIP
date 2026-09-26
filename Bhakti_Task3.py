# %% [markdown]
# # Car Price Prediction with Machine Learning
# 
# **Objective:** Build a regression model that predicts the selling price of a used car based on features such as brand, age, mileage, fuel type, and transmission.
# 
# **Tech stack:** Python, pandas, scikit-learn, matplotlib, seaborn
# 
# **Dataset:** `data/car_data.csv` — used-car listings in the same shape as the popular CarDekho "Vehicle dataset" on Kaggle. See the README for a note on the dataset's origin.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)
pd.set_option("display.precision", 3)

# %% [markdown]
# ## 1. Load the data

# %%
df = pd.read_csv("../data/car_data.csv")
print(df.shape)
df.head()

# %% [markdown]
# ## 2. Data cleaning
# 
# We check for: null values, duplicate rows, and inconsistent categorical casing (e.g. `"Petrol"` vs. `"petrol"` vs. `"PETROL"`).

# %%
print("Missing values per column:")
print(df.isnull().sum())
print()
print("Duplicate rows:", df.duplicated().sum())

# %%
# Inconsistent categorical casing before cleaning
print("Raw 'fuel' values:")
print(df["fuel"].value_counts(dropna=False))

# %%
# Drop exact duplicate rows
df = df.drop_duplicates().reset_index(drop=True)

# Normalise categorical text casing
for col in ["fuel", "seller_type", "transmission", "owner"]:
    df[col] = df[col].astype(str).str.strip().str.title()
    df.loc[df[col] == "Nan", col] = np.nan

# Drop rows still missing critical fields (small % of data)
df = df.dropna(subset=["fuel", "seller_type", "km_driven"]).reset_index(drop=True)
df["km_driven"] = df["km_driven"].astype(int)

print("Shape after cleaning:", df.shape)
print(df["fuel"].value_counts())

# %% [markdown]
# ## 3. Feature engineering
# 
# - `car_age` = current year − `year`
# - `brand` = extracted from the `name` column (first word)

# %%
CURRENT_YEAR = 2023
df["car_age"] = CURRENT_YEAR - df["year"]
df["brand"] = df["name"].str.split().str[0]

df[["name", "brand", "year", "car_age"]].head()

# %% [markdown]
# ## 4. Exploratory Data Analysis

# %%
plt.figure(figsize=(8, 5))
sns.histplot(df["selling_price"], bins=30, kde=True)
plt.title("Distribution of selling price (lakhs INR)")
plt.xlabel("Selling price (lakhs)")
plt.show()

# %%
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="fuel", y="selling_price")
plt.title("Selling price by fuel type")
plt.show()

# %%
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="car_age", y="selling_price", hue="transmission", alpha=0.6)
plt.title("Selling price vs. car age")
plt.xlabel("Car age (years)")
plt.ylabel("Selling price (lakhs)")
plt.show()

# %% [markdown]
# ## 5. Encode categorical variables

# %%
categorical_cols = ["brand", "fuel", "seller_type", "transmission", "owner"]
numeric_cols = ["car_age", "km_driven"]

df_encoded = pd.get_dummies(df[categorical_cols + numeric_cols], columns=categorical_cols, drop_first=True)
df_encoded["selling_price"] = df["selling_price"].values

print(df_encoded.shape)
df_encoded.head()

# %% [markdown]
# ## 6. Feature correlation heatmap

# %%
plt.figure(figsize=(8, 6))
corr = df_encoded[numeric_cols + ["selling_price"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation heatmap (numeric features)")
plt.show()

# %% [markdown]
# ## 7. Train / test split

# %%
X = df_encoded.drop(columns=["selling_price"])
y = df_encoded["selling_price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

# %% [markdown]
# ## 8. Train regression models
# 
# We train three models: Linear Regression (baseline), Random Forest Regressor, and Gradient Boosting Regressor.

# %%
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42),
}

predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions[name] = model.predict(X_test)
print("Models trained:", list(models.keys()))

# %% [markdown]
# ## 9. Evaluate models: MAE, RMSE, R²

# %%
def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2}

results = [evaluate(y_test, predictions[name], name) for name in models]
results_df = pd.DataFrame(results).set_index("Model").sort_values("R2", ascending=False)
results_df

# %% [markdown]
# ## 10. Feature importance for the best-performing model

# %%
best_name = results_df["R2"].idxmax()
best_model = models[best_name]
print("Best model:", best_name)

if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values()
    plt.figure(figsize=(8, 10))
    importances.tail(15).plot(kind="barh", color="seagreen")
    plt.title(f"Top 15 feature importances — {best_name}")
    plt.xlabel("Relative importance")
    plt.tight_layout()
    plt.show()
else:
    coefs = pd.Series(best_model.coef_, index=X.columns).sort_values()
    plt.figure(figsize=(8, 10))
    coefs.tail(15).plot(kind="barh", color="steelblue")
    plt.title(f"Top 15 coefficients — {best_name}")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 11. Conclusion
# 
# - **Car age** and **km driven** are consistently among the strongest predictors of selling price — older, higher-mileage cars sell for less, as expected from the age scatter plot in section 4.
# - **Brand** and **transmission type** (automatic vs. manual) also carry meaningful weight — premium brands and automatic transmissions command higher resale prices.
# - Tree-based models (Random Forest / Gradient Boosting) typically outperform plain Linear Regression here because price depreciation isn't perfectly linear with age or mileage.
# - Exact metrics will shift if you swap in the real CarDekho dataset from Kaggle in place of the included synthetic one — see the README.
