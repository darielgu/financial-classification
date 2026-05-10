# Financial Transaction Category Classification Using Machine Learning Models

Dariel Gutierrez, Bradley Mustoe, Joshua Sherrod, Zander Barajas  
CS 549 — Machine Learning  
Professor Joanne Chen  
May 2026  

# Abstract

Our project focuses on financial transaction category classification using supervised learning. Given a raw financial transaction including amount, description, merchant, and date, our goal is to automatically predict the semantic category of said transaction (e.g., groceries, rent, transportation, entertainment). We developed and evaluated several supervised machine learning models using two public Kaggle datasets containing transaction descriptions, merchant names, dates, account information, and transaction amounts. The datasets were cleaned, standardized, merged into one unified schema, and transformed through preprocessing techniques including duplicate removal, text normalization, one-hot encoding, and feature scaling.

Four machine learning models were implemented and evaluated: Logistic Regression, Random Forest, Support Vector Machine (SVM), and Neural Network. The models were trained and evaluated using accuracy, precision, recall, macro F1-score, and confusion matrices. Among the evaluated models, Random Forest achieved the strongest overall performance with the highest overall accuracy and macro F1-score. Although the overall performance metrics were modest, the experiments demonstrated the challenges associated with noisy financial transaction data, semantic overlap between categories, and class imbalance. The results suggest that ensemble-based methods such as Random Forest are more effective than simpler linear models for financial transaction classification tasks.

# Introduction

Financial transaction category classification is an important task in modern financial management systems because it enables automated budgeting, expense tracking, and fraud monitoring. Many banking applications rely on machine learning algorithms to automatically categorize user transactions into semantic spending groups. Automating this process improves usability for consumers by reducing the need for manual financial organization. The primary objective of this project was to design, implement, and compare supervised machine learning models capable of automatically classifying financial transactions into their correct categories.

The datasets used in this project contain both structured and unstructured financial data. Structured features include transaction amount, payment type, account name, user identifier, and transaction date. Unstructured features include merchant names and transaction descriptions. These text-based fields often contain semantic indicators of transaction categories. For example, restaurant names may indicate dining transactions, while utility providers may indicate recurring billing expenses. These datasets were chosen because they contain realistic transaction records with multiple spending categories and a combination of numerical and textual financial features.

Since the datasets originated from different sources, preprocessing and schema alignment were necessary before training. The preprocessing pipeline standardized category labels, removed duplicate records, handled missing values, normalized text formatting, and merged the datasets into a unified dataset containing 12,283 transactions across 18 categories.

Feature extraction was an important component of this project. To convert textual transaction descriptions into numerical representations, TF-IDF vectorization was used. TF-IDF was selected because it captures the importance of words within transaction descriptions while reducing the influence of common words that appear across many transactions. In addition, categorical variables were encoded using one-hot encoding and transaction amount features were standardized to improve compatibility with machine learning algorithms.

# Related Work

Related work leverages transformer-based architectures to better model the noisy textual descriptors used with financial transactions. The paper *Hierarchical Classification of Financial Transactions Through Context-Fusion of Transformer-based Embeddings and Taxonomy-aware Attention Layer* proposes Two-headed DragoNet, a model that uses stacked Transformer encoder layers to produce contextual embeddings from merchant name and business activity. These embeddings are fused and fed into two output heads that together predict transaction categories under a hierarchical structure. The authors introduce a taxonomy-aware attention layer to enforce consistency with the predefined category hierarchy, and they report F1-scores around 93–95%.

Our project is aligned with this body of work in several ways but focuses on a simpler approach. Like the Two-headed DragoNet paper, we aim to predict semantic categories of financial transactions based primarily on textual descriptors and structured fields, and we evaluate model performance with classification metrics such as accuracy, precision, recall, and F1-score. However, instead of designing a specialized transformer architecture, we concentrate on classical supervised machine learning classification models.

# Methodology

## Data Preprocessing

The preprocessing pipeline was designed to combine two separate financial transaction datasets into one consistent supervised learning dataset. Since the datasets used different column names and slightly different category formats, the first step was schema standardization. The project defined a shared set of canonical columns: `date`, `description`, `amount`, `transaction_type`, `account_name`, `category`, and `source_dataset`.

Each raw dataset was loaded separately, renamed into this shared schema, and tagged with a `source_dataset` field so the source of each row could still be tracked after merging.

Transaction descriptions were converted into normalized text by:
- stripping whitespace
- converting text to lowercase
- removing non-alphanumeric characters
- collapsing repeated whitespace

The cleaned text was stored as `description_clean`.

Transaction amounts were converted into numeric values, and rows with invalid or missing amounts were removed. Dates were parsed into datetime format, and rows with invalid dates were also removed.

Category labels were normalized to reduce mismatches between the two datasets. Similar labels were merged into shared category names. For example:
- `restaurants`, `fast food`, `food & dining`, `coffee shops`, and `food & drink` were merged into `Dining`
- `movies & dvds`, `television`, and `music` were merged into `Entertainment`
- `salary` and `paycheck` were merged into `Income`

After both datasets were standardized, they were merged into a single dataframe. Duplicate transactions were removed using the fields:
- `date`
- `description_clean`
- `amount`
- `category`

The preprocessing script then created train, validation, and test splits using stratified sampling so that each split preserved the overall class distribution. The test set used 20% of the data, and the remaining data was split again into training and validation sets.

Feature engineering was handled through a shared feature transformer used across all models. The cleaned transaction description, `description_clean`, was transformed using TF-IDF vectorization with:
- up to 10,000 features
- 1-to-4 word n-grams
- `min_df=2`
- `max_df=0.95`

Categorical features, including:
- `date_month`
- `date_day_of_week`
- `transaction_type`
- `account_name`

were encoded using one-hot encoding with unknown-value handling enabled.

The numerical `amount` feature was standardized using `StandardScaler` so that models sensitive to feature magnitude would not be dominated by raw transaction amounts.

## Model Selection

### Logistic Regression (Joshua Sherrod)

Logistic Regression was selected as a baseline model because it is simple, efficient, interpretable, and widely used for classification tasks. It can model the probability of a transaction belonging to a specific category, making it suitable for multi-class classification problems. Logistic Regression also performs effectively on sparse TF-IDF feature spaces and provided a strong baseline for comparing more complex models.

### Random Forest (Bradley Mustoe)

Random Forest was selected because it is an ensemble learning method that combines multiple decision trees to improve classification performance and reduce overfitting. Random Forest performs well on datasets containing both numerical and categorical features, making it appropriate for financial transaction data. Additionally, Random Forest can capture nonlinear relationships between transaction features that linear models may fail to detect.

### Support Vector Machine (Dariel Gutierrez)

Since transaction descriptions were represented using TF-IDF vectors, a linear SVM was appropriate for learning category boundaries from text-heavy features. Linear SVMs are effective in high-dimensional sparse feature spaces commonly produced by TF-IDF vectorization. Multiple implementations were tested, including Linear SVC and RBF kernel SVMs, but only small performance differences were observed, so Linear SVC was selected because of its computational efficiency. The SVM also used balanced class weights to reduce bias toward majority classes.

### Neural Network (Zander Barajas)

The neural network was selected because it can learn more complex patterns in the data that simpler models might miss. Since financial transactions contain a mixture of numerical, categorical, and textual features, a neural network may be able to capture relationships between these features more effectively than linear models.

## Training Process and Hyperparameter Tuning

All models used the same preprocessed train, validation, and test data. The shared feature transformer was fit on the training set and then applied to the validation and test sets. This prevented information leakage from the validation or test sets into the feature extraction process.

For Logistic Regression, the model was trained using balanced class weights and evaluated on the held-out test set using accuracy, precision, recall, and macro F1-score.

For SVM, hyperparameter tuning was performed using the validation set. The linear SVM tested several values of the regularization parameter `C`:
- 0.01
- 0.05
- 0.1
- 0.5
- 1.0
- 2.0
- 5.0
- 10.0

Each candidate used balanced class weighting. The best configuration was selected based on macro F1-score on the validation set.

For Random Forest, hyperparameter tuning was performed using `RandomizedSearchCV` with stratified 5-fold cross-validation. The search evaluated multiple combinations of:
- number of trees
- tree depth
- feature sampling ratio
- minimum samples per split
- minimum samples per leaf

The best-performing Random Forest configuration was selected using macro F1-score.

For Neural Network, training involved experimenting with dense fully connected layers and tuning learning rate, batch size, and hidden layer sizes to improve generalization performance.

# Results

## Overall Model Performance

After preprocessing and feature extraction, the four selected models were trained and evaluated on the held-out test dataset. Performance was measured using:
- accuracy
- macro precision
- macro recall
- macro F1-score

Macro-averaged metrics were emphasized because the dataset contained imbalanced transaction categories.

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
|---|---|---|---|---|
| Logistic Regression | 0.1095 | 0.1667 | 0.1818 | 0.1447 |
| SVM | 0.1302 | 0.1667 | 0.1771 | 0.1574 |
| Random Forest | 0.1656 | 0.1672 | 0.1839 | 0.1623 |

Among all evaluated models, Random Forest achieved the strongest overall performance across nearly every evaluation metric.

## Per Category Performance

Model performance varied significantly across transaction categories. Categories such as:
- Investment
- Other
- Credit Card Payment

generally achieved higher F1-scores because these categories contained stronger semantic patterns and more distinguishable transaction descriptors.

In contrast, categories such as:
- Dining
- Shopping
- Entertainment

were more difficult to classify because many merchants and transaction descriptions overlapped semantically across categories.

## Visualizations and Confusion Matrices

Confusion matrices were generated for all models to visualize classification behavior and category-level prediction errors. The confusion matrices revealed that the models frequently confused semantically related categories.

Common misclassification patterns included:
- Dining vs Shopping
- Entertainment vs Travel
- Bills & Utilities vs Mortgage & Rent

These errors suggest that many financial transaction descriptions lack sufficient contextual information for classical machine learning models to separate categories cleanly.

# Discussion

The results demonstrate that financial transaction classification is a difficult machine learning problem, especially when using classical machine learning models and sparse text representations. Although all four models achieved performance above baseline, the overall metrics remained relatively modest.

Several factors likely contributed to the limited performance. Transaction descriptions are highly noisy and inconsistent. Merchant names may appear abbreviated, misspelled, or formatted differently across transactions. In many cases, transaction descriptions contain limited semantic information, making it difficult for models to identify clear category boundaries.

Many financial categories overlap semantically. For example, transactions from coffee shops, restaurants, convenience stores, and retail stores may all contain similar merchant terminology. This overlap increases ambiguity between categories such as Dining, Shopping, and Entertainment.

Lastly, the dataset exhibited substantial class imbalance. Categories such as Dining and Credit Card Payment contained significantly more examples than categories such as Travel, Electronics & Software, and Health & Fitness. As a result, minority categories often suffered from lower recall and lower F1-scores.

## Comparative Analysis of Models

Random Forest achieved the strongest overall performance across nearly every evaluation metric. This suggests that nonlinear ensemble-based methods are more effective at capturing interactions between textual and numerical transaction features.

The SVM model achieved the second-best macro F1-score. Since the project relied heavily on TF-IDF text representations, the linear SVM benefited from its ability to operate effectively in high-dimensional sparse feature spaces. However, SVM performance remained limited by the noisy and overlapping nature of the transaction descriptions.

Logistic Regression served as an interpretable baseline model. While it achieved the weakest overall performance, it still demonstrated that some category structure exists within the feature space.

## Limitations and Challenges Encountered

Several limitations affected the project results.

One major limitation was the use of TF-IDF vectorization instead of contextual embeddings. TF-IDF treats words independently and cannot capture deeper semantic meaning or contextual relationships between merchants and spending categories.

Another limitation was dataset imbalance. Certain categories contained very few examples, reducing the models’ ability to generalize effectively to minority classes.

The datasets also contained substantial noise and inconsistency. Merchant names often appeared in multiple formats, and some transactions lacked descriptive information entirely.

# Conclusion

This project explored the use of supervised machine learning techniques for financial transaction category classification using two merged Kaggle transaction datasets. The preprocessing pipeline standardized category labels, cleaned noisy transaction descriptions, removed duplicate records, engineered numerical and categorical features, and transformed textual transaction descriptions using TF-IDF vectorization.

Four machine learning models were implemented and evaluated:
- Logistic Regression
- Support Vector Machine
- Random Forest
- Neural Network

Among the evaluated models, Random Forest achieved the strongest overall performance.

The results demonstrated that financial transaction classification is a challenging problem because of noisy merchant descriptions, overlapping semantic categories, and class imbalance. Nevertheless, the experiments showed that ensemble-based methods such as Random Forest outperform simpler linear approaches on this classification task.

Future work could significantly improve performance by incorporating:
- transformer-based embeddings
- contextual language models
- hierarchical category prediction systems
- larger datasets
- user-specific spending history
- deep learning architectures designed specifically for financial transaction semantics

# Contributions

## Dariel Gutierrez
- Implemented the Support Vector Machine (SVM) classifier
- Performed SVM hyperparameter tuning and validation experiments
- Assisted with preprocessing pipeline development and feature engineering
- Conducted comparative analysis between models
- Contributed to the methodology, results, and discussion sections of the report

## Joshua Sherrod
- Implemented the Logistic Regression model
- Assisted with baseline evaluation experiments
- Contributed to preprocessing validation and testing
- Assisted with report writing and editing

## Bradley Mustoe
- Implemented and tuned the Random Forest classifier
- Performed RandomizedSearchCV hyperparameter optimization
- Generated confusion matrices and evaluation outputs
- Assisted with results interpretation and analysis

## Zander Barajas
- Assisted with Neural Network implementation and experimentation
- Contributed to preprocessing validation
- Assisted with evaluation visualization generation
- Participated in report formatting and editing

# References

Busson, A. J. G., Rocha, R., Gaio, R., Miceli, R., Pereira, I., de S. Moraes, D., Colcher, S., Veiga, A., Rizzi, B., Evangelista, F., Santos, L., Marques, F., Rabaioli, M., Feldberg, D., Mattos, D., Pasqua, J., & Dias, D. (2023). *Hierarchical classification of financial transactions through context-fusion of transformer-based embeddings and taxonomy-aware attention layer*. arXiv preprint arXiv:2312.07730.
