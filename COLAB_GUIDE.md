# Quick Start: Model Comparison in Colab

## 🚀 How to Run

1. **Open Google Colab**: https://colab.research.google.com
2. **Upload Notebook**: Open `MODEL_COMPARISON.ipynb` from this repo
3. **Run All Cells**: Shift + Ctrl + Enter (or click "Run all")
4. **Wait**: Takes ~10-15 minutes to train all models
5. **Review Results**: Scroll through outputs and visualizations

## 📊 What You'll Get

### Metrics Table

```
Model                  Accuracy  Precision  Recall  F1-Score
─────────────────────────────────────────────────────────────
SVM                    [TBD]     [TBD]      [TBD]   [TBD]
Random Forest          [TBD]     [TBD]      [TBD]   [TBD]
Logistic Regression    [TBD]     [TBD]      [TBD]   [TBD]
```

### Visualizations

- 📈 **Metrics comparison chart** (side-by-side bars)
- 🔥 **Confusion matrices** (3 visualizations)
- 🎯 **Winner summary** (which model is best)

### CSV Export

- File: `models/comparison_results.csv`
- Ready for presentations or further analysis

## 🎯 Key Metrics Explained

| Metric        | Meaning                                 | When to Use                     |
| ------------- | --------------------------------------- | ------------------------------- |
| **Accuracy**  | % of all predictions correct            | When categories are balanced    |
| **Precision** | Of predicted class X, how many correct? | When false positives are costly |
| **Recall**    | Of actual class X, how many found?      | When false negatives are costly |
| **F1-Score**  | Balance between precision & recall      | General purpose metric          |

_All metrics are **macro-averaged** (equal weight per category)_

## 💡 What Each Model Does

### SVM (Linear)

- ✅ **Your implementation** — Dariels SVM
- ✅ Fast training & prediction
- ✅ Great with ~5000 features (TF-IDF)
- ❌ Only finds linear boundaries

### Random Forest

- ✅ Finds non-linear patterns
- ✅ Captures feature interactions
- ❌ Slower to train & predict
- ❌ Higher memory usage

### Logistic Regression

- ✅ Baseline comparison model
- ✅ Very fast
- ✅ Interpretable results
- ❌ Simple linear model

## 📋 Notebook Sections

| Section               | Purpose                              |
| --------------------- | ------------------------------------ |
| 1. Setup Environment  | Clone repo, install packages         |
| 2. Data Preprocessing | Create train/val/test splits         |
| 3. Train All Models   | Run SVM, RF, Logistic Regression     |
| 4. Extract Results    | Load predictions and compute metrics |
| 5. Comparison Results | Display side-by-side metrics         |
| 6. Visualizations     | Generate charts and plots            |
| 7. Detailed Reports   | Per-category performance             |
| 8. Summary            | Key findings and insights            |
| 9. Export Results     | Save to CSV                          |

## 🔍 Interpreting Results

### "Which model is best?"

Look at the **F1-Score (Macro)** column — this is the fairest metric for multi-class classification with potential class imbalance.

### "Why are some categories harder?"

Check the **classification report** in Section 7:

- High precision, low recall? → Model misses some transactions
- Low precision, high recall? → Model over-predicts that category
- Look at **confusion matrix** to see which categories it confuses

### Performance Expectations

- **Accuracy > 70%**: Good (18 categories, random baseline = 5.5%)
- **F1 > 0.60**: Excellent for this type of problem
- **Per-category F1 > 0.50**: Solid performance for that category

## 📁 Files After Running

```
models/
├── comparison_results.csv              ← Main results
├── model_comparison.png                ← Metrics chart
├── svm_confusion_matrix.png            ← Your SVM performance
├── random_forest_confusion_matrix.png  ← RF performance
└── logistic_regression_confusion_matrix.png  ← LR baseline
```

## ⚠️ Common Issues

**"No module named 'pandas'"**
→ The notebook installs dependencies in the first cell. Just run that cell first.

**"GPU not available"**
→ That's OK! CPU is fast enough for this dataset.

**"Models not loading"**
→ Make sure Section 3 (training) completed successfully before Section 4.

## 🎓 For Your CS 549 Project

Use this analysis to write your results section:

- Include the metrics table
- Include the comparison chart
- Discuss why your SVM performed a certain way
- Compare it to RF and LR baselines
- Note the trade-offs (speed vs accuracy)

## 📚 Next Steps

1. ✅ Run the notebook in Colab
2. ✅ Download the CSV with results
3. ✅ Save the comparison chart
4. ✅ Update your project report with findings
5. ⚙️ (Optional) Try ensemble methods or other models

---

**Questions?** Check `RESULTS.md` for detailed technical info.
