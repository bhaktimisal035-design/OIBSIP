# OIBSIP

My project submissions for the Oasis Infobyte Data Science Internship (OIBSIP).

**Intern:** Bhakti
**Track:** Data Science

---

## Task 1 — Iris Flower Classification

`Bhakti_Task1.ipynb`

For this task I built a model to classify iris flowers into one of three species — Setosa, Versicolor, or Virginica — using their sepal and petal measurements.

### Approach
- Loaded the Iris dataset from `sklearn.datasets` (it's built into scikit-learn, so no download was needed).
- Did some EDA first — checked the shape, data types, and for missing values, then looked at summary stats per species.
- Visualised the data with a pairplot, box plots for each feature, and a correlation heatmap to see how the species differ.
- Compared petal vs. sepal measurements to figure out which features actually separate the species best (turns out it's the petal measurements, by a wide margin).
- Split the data 80/20 (stratified so each species stays balanced in both sets).
- Trained four different models — Logistic Regression, KNN, Decision Tree, and Random Forest — to compare approaches.
- Evaluated each one with accuracy, a confusion matrix, and a classification report, then picked the best-performing model and explained why.

### Tech stack
Python · pandas · numpy · matplotlib · seaborn · scikit-learn · Jupyter Notebook

### How to run it
```bash
pip install -r requirements.txt
jupyter notebook Bhakti_Task1.ipynb
```
No dataset download required — it loads straight from scikit-learn.

### Result
All four models scored well on the held-out test set (90%+ accuracy). Petal length and petal width turned out to be the strongest predictors, which lines up with what the visualisations showed. Full breakdown and model comparison is in the notebook.

---

## Repo structure
```
OIBSIP/
├── Task1_Iris_Flower_Classification/
│   └── Bhakti_Task1.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

More tasks will be added here as I complete them, following the same `Bhakti_TaskNumber` naming.
