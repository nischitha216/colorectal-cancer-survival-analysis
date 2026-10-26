

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("colorectal_cancer_dataset.csv")

print("DATASET OVERVIEW")
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# 1. Target distribution
print("\nFIVE-YEAR SURVIVAL LABEL DISTRIBUTION")
print(df["Survival_5_years"].value_counts())
print(df["Survival_5_years"].value_counts(normalize=True).round(3))

# 2. Numeric summaries
numeric_cols = [
    "Age",
    "Tumor_Size_mm",
    "Healthcare_Costs",
    "Incidence_Rate_per_100K",
    "Mortality_Rate_per_100K"
]

print("\nNUMERIC FEATURE SUMMARY")
print(df[numeric_cols].describe().round(2))

# 3. Survival rate by selected variables
for col in [
    "Cancer_Stage",
    "Early_Detection",
    "Treatment_Type",
    "Screening_History",
    "Healthcare_Access"
]:
    rates = (
        df.groupby(col, observed=True)["Survival_5_years"]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .round(2)
    )
    print(f"\nSurvival label rate by {col} (%):")
    print(rates)

# 4. Plot survival label distribution
sns.countplot(data=df, x="Survival_5_years")
plt.title("Five-Year Survival Label Distribution")
plt.xlabel("Survival label")
plt.ylabel("Number of records")
plt.tight_layout()
plt.savefig("survival_distribution.png", dpi=150)
plt.close()

# 5. Plot survival label rate by cancer stage
stage_rates = (
    df.groupby("Cancer_Stage", observed=True)["Survival_5_years"]
    .apply(lambda s: (s == "Yes").mean() * 100)
)

stage_rates.plot(kind="bar")
plt.title("Survival Label Rate by Cancer Stage")
plt.ylabel("Records labelled Yes (%)")
plt.xlabel("Cancer stage")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("survival_by_stage.png", dpi=150)
plt.close()

print("\nAnalysis complete!")
print("Saved: survival_distribution.png")
print("Saved: survival_by_stage.png")



# 6. Correlation heatmap for numeric variables

numeric_df = df.select_dtypes(include="number").drop(
    columns=["Patient_ID"],
    errors="ignore"
)

correlation_matrix = numeric_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)
plt.title("Correlation Heatmap of Numeric Variables")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.close()

# 7. Survival label rate by screening history

screening_rates = (
    df.groupby("Screening_History", observed=True)["Survival_5_years"]
    .apply(lambda s: (s == "Yes").mean() * 100)
    .sort_values(ascending=False)
)

plt.figure(figsize=(7, 5))
sns.barplot(
    x=screening_rates.index,
    y=screening_rates.values
)
plt.title("Survival Label Rate by Screening History")
plt.xlabel("Screening history")
plt.ylabel("Records labelled Yes (%)")
plt.tight_layout()
plt.savefig("survival_by_screening.png", dpi=150)
plt.close()

print("\nAdditional plots saved:")
print("correlation_heatmap.png")
print("survival_by_screening.png")

# 6. Correlation heatmap for numeric variables

numeric_df = df.select_dtypes(include="number").drop(
    columns=["Patient_ID"],
    errors="ignore"
)

correlation_matrix = numeric_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)
plt.title("Correlation Heatmap of Numeric Variables")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.close()

# 7. Survival label rate by screening history

screening_rates = (
    df.groupby("Screening_History", observed=True)["Survival_5_years"]
    .apply(lambda s: (s == "Yes").mean() * 100)
    .sort_values(ascending=False)
)

plt.figure(figsize=(7, 5))
sns.barplot(
    x=screening_rates.index,
    y=screening_rates.values
)
plt.title("Survival Label Rate by Screening History")
plt.xlabel("Screening history")
plt.ylabel("Records labelled Yes (%)")
plt.tight_layout()
plt.savefig("survival_by_screening.png", dpi=150)
plt.close()

print("\nAdditional plots saved:")
print("correlation_heatmap.png")
print("survival_by_screening.png")
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# 8. Prepare data for machine learning

target = "Survival_5_years"

# Remove target, identifier, and possible leakage columns
drop_columns = [
    "Patient_ID",
    "Mortality",
    "Survival_Prediction",
    target
]

X = df.drop(columns=drop_columns, errors="ignore")
y = df[target].map({"No": 0, "Yes": 1})

# Encode categorical columns
X = pd.get_dummies(X, drop_first=True)

# Handle missing or infinite values
X = X.replace([float("inf"), float("-inf")], float("nan"))
X = X.fillna(X.median())

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nMODEL DATA PREPARATION")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Number of features:", X_train.shape[1])
print("Missing target labels:", y.isna().sum())
# 9. Train baseline and machine-learning models

models = {
    "Majority Baseline": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Balanced Accuracy": balanced_accuracy_score(
            y_test, predictions
        ),
        "F1 Score": f1_score(y_test, predictions, zero_division=0)
    }

    # ROC-AUC requires probability estimates
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
        row["ROC-AUC"] = roc_auc_score(y_test, probabilities)

    results.append(row)

    print(classification_report(
        y_test,
        predictions,
        target_names=["No", "Yes"],
        zero_division=0
    ))

results_df = pd.DataFrame(results)
print("\nMODEL COMPARISON")
print(results_df.round(4).to_string(index=False))

results_df.to_csv("model_comparison.csv", index=False)
print("\nSaved: model_comparison.csv")


# 10. Check whether features contain predictive signal

print("\nPREDICTIVE SIGNAL CHECK")

print("\nTarget distribution:")
print(y.value_counts(normalize=True).round(4))

print("\nSurvival rate by age group:")
age_groups = pd.cut(
    df["Age"],
    bins=[29, 39, 49, 59, 69, 79, 89],
    include_lowest=True
)

print(
    df.groupby(age_groups, observed=True)["Survival_5_years"]
      .apply(lambda s: (s == "Yes").mean())
      .round(4)
)

print("\nSurvival rate by cancer stage:")
print(
    df.groupby("Cancer_Stage", observed=True)["Survival_5_years"]
      .apply(lambda s: (s == "Yes").mean())
      .round(4)
)

print("\nCheck whether the two outcome-related columns match:")
print(pd.crosstab(
    df["Survival_5_years"],
    df["Survival_Prediction"],
    normalize="index"
))



from scipy.stats import chi2_contingency, kruskal

print("\nSTATISTICAL ASSOCIATION CHECKS")

categorical_cols = [
    "Gender", "Cancer_Stage", "Early_Detection", "Treatment_Type",
    "Screening_History", "Healthcare_Access", "Smoking_History",
    "Insurance_Status"
]

for col in categorical_cols:
    table = pd.crosstab(df[col], df["Survival_5_years"])
    chi2, p_value, _, _ = chi2_contingency(table)
    print(f"{col}: p-value = {p_value:.6g}")

numeric_cols = [
    "Age", "Tumor_Size_mm", "Healthcare_Costs",
    "Incidence_Rate_per_100K", "Mortality_Rate_per_100K"
]

for col in numeric_cols:
    groups = [
        group[col].dropna().values
        for _, group in df.groupby("Survival_5_years")
    ]
    statistic, p_value = kruskal(*groups)
    print(f"{col}: p-value = {p_value:.6g}")



# FINAL MODEL COMPARISON PLOT

import matplotlib.pyplot as plt
import pandas as pd

results = pd.read_csv("model_comparison.csv")

metrics = ["Accuracy", "Balanced Accuracy", "ROC-AUC"]

ax = results.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(10, 6),
    ylim=(0, 1),
    rot=15
)

ax.set_title("Colorectal Cancer Survival Model Comparison")
ax.set_ylabel("Score")
ax.set_xlabel("Model")
ax.legend(loc="lower right")
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("final_model_comparison.png", dpi=300)
plt.close()

print("\nSaved: final_model_comparison.png")
