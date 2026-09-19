# Unemployment Analysis with Python 🇮🇳

Exploratory data analysis of unemployment trends across Indian states and union territories, with a focus on the impact of the **COVID-19 pandemic** on unemployment rates.

> Built as part of the **Oasis Infobyte — Data Science Track**.

## 📌 Objective

Perform exploratory data analysis on unemployment data to uncover regional and temporal trends, with a focus on the impact of the COVID-19 pandemic on unemployment rates in India.

## 🛠️ Tech Stack

- Python
- pandas
- matplotlib
- seaborn
- Jupyter Notebook

## 📁 Project Structure

```
unemployment-analysis/
├── data/
│   └── Unemployment_in_India.csv     # Dataset (Region, Date, Unemployment Rate, Employed, Labour Participation Rate, Area)
├── notebooks/
│   └── unemployment_analysis.ipynb   # Full analysis notebook
├── requirements.txt
└── README.md
```

## 📊 Dataset

The dataset covers monthly unemployment statistics for Indian states (Jan 2019 – Oct 2020), broken down by **Rural** and **Urban** area, with the following columns:

| Column | Description |
|---|---|
| `Region` | State / Union Territory |
| `Date` | Observation month |
| `Frequency` | Reporting frequency (Monthly) |
| `Estimated Unemployment Rate (%)` | % of labour force unemployed |
| `Estimated Employed` | Number of people employed |
| `Estimated Labour Participation Rate (%)` | % of population in the labour force |
| `Area` | Rural / Urban |

**Note:** `data/Unemployment_in_India.csv` in this repo is a realistic **synthetic sample** generated to match the schema of the well-known Kaggle dataset *"Unemployment in India"* (by Gokul Rejith), so the notebook runs end-to-end with no setup. To reproduce the analysis on real-world figures, download the original dataset from Kaggle and replace the CSV — no code changes are required since the column names match.

## 🔍 What's in the Notebook

1. Data loading & inspection (shape, dtypes, null check)
2. Data cleaning & type conversion (dates, whitespace, missing-value imputation)
3. Descriptive statistics
4. Region-wise average unemployment rate
5. Month-wise national trend (time-series line chart)
6. Time-series comparison across 4 major states (Delhi, Maharashtra, Tamil Nadu, Uttar Pradesh)
7. Top 10 states by average unemployment rate (bar chart)
8. Rural vs. Urban comparison
9. **COVID-19 impact analysis** — pre-lockdown vs. peak-lockdown vs. recovery period
10. Correlation heatmap of key indicators
11. Conclusions & key findings

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/unemployment-analysis.git
cd unemployment-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook notebooks/unemployment_analysis.ipynb
```

## 📈 Key Findings

- National unemployment spiked sharply during the April–June 2020 COVID-19 lockdown period, before gradually recovering.
- Urban areas show a consistently higher unemployment rate than rural areas.
- A handful of states sit well above the national average, while others trend consistently lower.
- Estimated Employed and Unemployment Rate are negatively correlated, as expected.

## 📄 License

This project is open-sourced for educational purposes as part of the Oasis Infobyte internship program.
