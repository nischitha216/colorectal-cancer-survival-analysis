 Colorectal Cancer Survival Analysis

 Project Overview
This project explores a synthetic Kaggle dataset containing 167,497 records related to colorectal cancer. It uses exploratory data analysis (EDA), statistical tests, and machine learning to investigate whether the available variables can predict five-year survival.

 Objectives
- Explore patient and clinical characteristics.
- Visualize five-year survival distributions and patterns.
- Test associations between selected variables and survival.
- Compare baseline and machine learning classification models.

 Workflow
1. Data loading and inspection.
2. Exploratory data analysis and visualization.
3. Categorical encoding and missing-value handling.
4. Stratified 80/20 train-test split.
5. Evaluation of a majority-class baseline, logistic regression, and random forest.
6. Statistical association testing using chi-square and Kruskal–Wallis tests.

 Model Results

| Model | Accuracy | Balanced Accuracy | ROC-AUC |
|---|---:|---:|---:|
| Majority baseline | 0.5993 | 0.5000 | 0.5000 |
| Logistic regression | 0.5993 | 0.5000 | 0.4997 |
| Random forest | 0.5993 | 0.5000 | 0.4997 |

The models did not demonstrate predictive performance beyond the majority-class baseline. Their results suggest that the supplied features contain little usable predictive signal for the target in this dataset, or that the current modeling approach does not capture it.

 Statistical Analysis
The selected chi-square and Kruskal–Wallis tests returned p-values above 0.05. No statistically significant associations were detected in these tests.

 Key Findings
- Approximately 59.93% of records were labelled as surviving five years.
- The majority baseline achieved approximately 59.93% accuracy by predicting the most frequent class.
- Both machine learning models performed similarly to the baseline.
- The logistic regression model produced a convergence warning and should not be considered fully optimized.

 Limitations
- The dataset is synthetic and cannot establish clinical validity.
- Statistical significance alone does not measure the strength or practical importance of an association.
- The results do not support using these models for clinical decisions.
- Further preprocessing, feature validation, and model evaluation would be needed before drawing stronger conclusions.

 Output Files
- `survival_distribution.png`
- `survival_by_stage.png`
- `correlation_heatmap.png`
- `survival_by_screening.png`
- `final_model_comparison.png`
- `model_comparison.csv`

 Conclusion
This project demonstrates an end-to-end exploratory and baseline machine learning workflow for a cancer-related dataset. Its main finding is that the tested models did not outperform a simple baseline, highlighting the importance of data quality and meaningful predictive signal in biomedical machine learning.
