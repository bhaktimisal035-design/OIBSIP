# Car Price Prediction with Machine Learning

A regression project that predicts the selling price of a used car from features like brand, age, mileage, fuel type, and transmission.

## Objective

Build and compare regression models that predict `selling_price` for a used car, using cleaned and engineered features from listing data.

## Tech stack

- Python 3
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- Jupyter Notebook

## Project structure

```
Bhakti_Task3/
├── data/
│   └── car_data.csv           # used-car listings (name, year, price, km, fuel, etc.)
├── notebooks/
│   └── Bhakti_Task3.ipynb     # full analysis: cleaning -> features -> EDA -> models -> evaluation
├── requirements.txt
└── README.md
```

## Dataset

`data/car_data.csv` columns:

| Column          | Description                                              |
|------------------|-----------------------------------------------------------|
| `name`           | Car name, e.g. `"Maruti Swift 2018"` (brand is the first word) |
| `year`           | Year of manufacture                                       |
| `selling_price`  | Resale price, in lakhs (INR 100,000s)                      |
| `km_driven`      | Total kilometers driven                                    |
| `fuel`           | Petrol / Diesel / CNG / LPG (casing is intentionally messy, for the cleaning step) |
| `seller_type`    | Individual / Dealer / Trustmark Dealer                     |
| `transmission`   | Manual / Automatic                                          |
| `owner`          | First Owner / Second Owner / Third Owner / Fourth & Above Owner |

The file also includes a handful of duplicate rows and missing values on purpose, so the notebook's data-cleaning step has real work to do.

**Note on the data:** this repo ships a synthetically generated dataset built to match the shape and general price signal (age, mileage, brand, and transmission driving price) of the well-known **CarDekho "Vehicle dataset"** on Kaggle, since this environment couldn't reach the internet to download it directly. If you want the original dataset instead, search **"car price prediction dataset"** on [Kaggle](https://www.kaggle.com) — "Vehicle dataset from cardekho" is the widely used one — and drop the CSV into `data/` in place of the included file. Column names may need a quick rename to match (`name`, `year`, `selling_price`, `km_driven`, `fuel`, `seller_type`, `transmission`, `owner`), after which the notebook runs unchanged.

## Setup

```bash
git clone <this-repo>
cd OIBSIP/Bhakti_Task3
python -m venv venv
source venv/bin/activate       # on Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/Bhakti_Task3.ipynb
```

## What the notebook covers

1. Load data
2. Data cleaning: nulls, duplicates, inconsistent categorical casing
3. Feature engineering: `car_age` from `year`, `brand` extracted from `name`
4. EDA: price distribution, price vs. fuel type boxplot, price vs. car age scatter plot
5. Encode categorical variables (one-hot encoding)
6. Feature correlation heatmap
7. Train/test split
8. Train 3 regression models: Linear Regression, Random Forest, Gradient Boosting
9. Evaluate with MAE, RMSE, R²
10. Feature importance chart for the best-performing model
11. Conclusion

## Results (on the included synthetic dataset)

| Model               | R²    |
|----------------------|-------|
| Linear Regression     | ~0.80 |
| Random Forest          | ~0.93 |
| Gradient Boosting      | ~0.96 |

**Takeaway:** car age and kilometers driven are the strongest price predictors, with brand and transmission type also mattering. Tree-based models beat plain linear regression since depreciation isn't perfectly linear. Numbers will vary if you swap in the real Kaggle dataset.

## License

MIT — feel free to use and adapt this for learning purposes.
