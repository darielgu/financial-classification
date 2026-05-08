# Financial Classification - Model Comparison Results

## Overview

This document compares three machine learning models trained to classify personal financial transactions into 18 different spending categories. All models use identical train/validation/test splits and shared feature engineering.

## Models Trained

1. **Support Vector Machine (SVM)** - Linear kernel with hyperparameter tuning
2. **Random Forest** - Ensemble method with hyperparameter optimization
3. **Logistic Regression** - Linear baseline for comparison

## Dataset Summary

- **Total transactions**: 12,283
- **Train/Val/Test split**: 60% / 20% / 20%
- **Test set size**: ~2,456 transactions
- **Number of categories**: 18

## Shared Feature Engineering

All models use the same feature pipeline:

| Feature Type | Details | Dimension |
|---|---|---|
| **Text** | TF-IDF on transaction descriptions (1-3 grams, max 5000 features) | ~5000 |
| **Categorical** | One-hot encoding (date_month, date_day_of_week, transaction_type, account_name) | ~20 |
| **Numeric** | Standardized transaction amount | 1 |
| **Total Features** | | ~5021 |

## Model Comparison Results

### Performance Metrics (Test Set)

> **Note**: Metrics are macro-averaged to account for class imbalance and treat all categories equally.

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
|---|---|---|---|---|
| SVM | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD |
| Logistic Regression | TBD | TBD | TBD | TBD |

*Run the notebook in Colab to populate these values*

### Winner Summary

| Metric | Winner | Value |
|---|---|---|
| **Accuracy** | *To be determined* | — |
| **Precision** | *To be determined* | — |
| **Recall** | *To be determined* | — |
| **F1-Score** | *To be determined* | — |
| **Overall Best** | *To be determined* | — |

## Model Details

### 1. Support Vector Machine (SVM)

**Implementation**: Linear SVM (`sklearn.svm.LinearSVC`)

**Hyperparameter Tuning**:
- Parameter combinations tested: 16
- Parameters varied:
  - **C** (regularization strength): [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
  - **class_weight**: [None, "balanced"]
- Tuning strategy: Validation set F1-score
- Best parameters: *See notebook output*

**Characteristics**:
- ✓ Fast training and inference
- ✓ Memory efficient with sparse features
- ✓ Good for high-dimensional data (5000+ features)
- — Linear decision boundaries only

### 2. Random Forest

**Implementation**: Ensemble of 100-500 decision trees

**Hyperparameter Tuning**:
- Search method: RandomizedSearchCV (20 iterations)
- Cross-validation: Stratified 5-fold
- Parameters varied:
  - **n_estimators**: [100, 200, 300, 500]
  - **max_depth**: [None, 10, 20, 30]
  - **max_features**: ["sqrt", "log2", 0.3]
  - **min_samples_split**: [2, 5, 10]
  - **min_samples_leaf**: [1, 2, 4]
  - **class_weight**: ["balanced", "balanced_subsample"]
- Scoring metric: Macro F1-score
- Best parameters: *See notebook output*

**Characteristics**:
- ✓ Captures non-linear relationships
- ✓ Feature importance analysis possible
- ✓ Handles feature interactions
- — Longer training and inference time
- — Higher memory usage

### 3. Logistic Regression

**Implementation**: Multinomial logistic regression

**Configuration**:
- Max iterations: 1000
- Class weight: "balanced"
- No hyperparameter tuning

**Characteristics**:
- ✓ Fast baseline
- ✓ Interpretable coefficients
- ✓ Probabilistic predictions
- — Assumes linear decision boundaries
- — No hyperparameter tuning

## Key Findings

### Performance Analysis

*To be populated after running the Colab notebook*

- Which model performs best overall?
- What are the performance gaps between models?
- Which metrics show the largest variance?
- Are there specific transaction categories where models differ?

### Class-Specific Performance

Some transaction categories may be easier to classify than others:
- Categories with distinctive keywords (e.g., "Rent", "Uber") → Higher accuracy
- Ambiguous categories (e.g., "Other", "Services") → Lower accuracy
- Look for systematic errors in confusion matrices

### Model Trade-offs

| Aspect | SVM | Random Forest | Logistic Regression |
|---|---|---|---|
| **Speed (Train)** | Fast | Slow | Fastest |
| **Speed (Predict)** | Fast | Fast | Fastest |
| **Accuracy** | — | — | — |
| **Interpretability** | Low | Medium | High |
| **Memory** | Low | Medium | Low |

## How to Use These Results

### Run the Comparison Notebook

Upload **`MODEL_COMPARISON.ipynb`** to Google Colab:

1. Open https://colab.research.google.com
2. Upload the notebook
3. Run all cells (takes ~10-15 minutes)
4. Results and visualizations generate automatically

### Interpret the Results

- **Accuracy**: Overall correctness (suitable if classes are balanced)
- **Precision**: Of predicted class X, how many are actually class X?
- **Recall**: Of actual class X, how many did we find?
- **F1-Score**: Harmonic mean of precision and recall (good overall metric)

Use **macro-averaging** (not micro) because:
- Treats rare and common categories equally
- Better for imbalanced datasets
- Reflects true generalization performance

### Next Steps

1. **Deploy the best model** for production
2. **Analyze errors** in the confusion matrix
3. **Consider ensembles** (voting between models)
4. **Retrain periodically** with new transaction data
5. **Monitor drift** as spending patterns change

## Files Generated

After running the notebook:

```
models/
├── svm.joblib                              # Trained SVM model
├── random_forest.joblib                    # Trained Random Forest
├── logistic_regression.joblib              # Trained Logistic Regression
├── svm_confusion_matrix.png                # Confusion matrix visualization
├── random_forest_confusion_matrix.png      # Confusion matrix visualization
├── logistic_regression_confusion_matrix.png # Confusion matrix visualization
├── model_comparison.png                    # Side-by-side metrics comparison
└── comparison_results.csv                  # Results in CSV format
```

## Conclusion

*To be updated after running the models*

This analysis provides a data-driven comparison of three classification approaches. The results inform the choice of which model to deploy and highlight areas for future improvement.

---

**Last Updated**: Run the notebook to generate fresh results  
**Dataset Version**: v1 (12,283 transactions, 18 categories)  
**Feature Engineering**: TF-IDF + One-Hot + StandardScaler
