# Thyroid Type Prediction

## Problem Statement
The goal is to create a classification methodology for predicting the type of thyroid a person has based on specified features. The dataset comprises 3772 instances and 30 features.

## Data Preprocessing
- Checked dataset shape: (3772, 30).
- Explored the initial rows of the dataset.

### Missing Values Handling
- Identified missing values replaced with '?' in the dataset.
- Replaced '?' with 'nan' and dropped irrelevant columns ('TBG').
- Removed columns indicating the presence of values in subsequent columns.

### Categorical Data Handling
- Mapped binary columns for efficient encoding.
- Utilized LabelEncoder for the output class and get_dummies for columns with more than two categories.

### Imputation of Missing Values
- Utilized KNNImputer for imputing missing values effectively.

### Data Distribution Analysis
- Analyzed and visualized the distribution of continuous data (age, TSH, T3, TT4, T4U, FTI).
- Applied log transformation to skewed data and dropped the 'TSH' column due to undesirable trends.

## Handling Imbalanced Data
- Recognized the highly imbalanced dataset.
- Used RandomOverSampler from imbalanced-learn library to address imbalances.

## Model Training and Metrics
- Trained classification models using the balanced dataset.
- Evaluated model performance using relevant metrics:
  - Accuracy: 94.17%

### Model Training (Continued)
- Split the dataset into training and testing sets (80-20 split).
- Utilized a K-Nearest Neighbors Classifier (KNN) for model training.

```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
```

### Model Prediction
- Predicted the output on the training set.

```python
knn_clf.predict(X_train)
```

### Model Evaluation
- Evaluated the model performance on the test set.

```python
knn_clf.score(X_test, y_test)
```

### Metrics
- **Accuracy:** 94.17%

## Conclusion
- The K-Nearest Neighbors Classifier (KNN) demonstrated the highest accuracy, reaching 94.17% on the test set. We further explored other classifiers, including Support Vector Classifier (SVC) and Decision Tree. SVC achieved an accuracy of 93.45%, while Decision Tree achieved 88.92%.

**Additional Analysis:**
- We used cross-validation for a more accurate model. SVC achieved an accuracy of 93.45%, and Decision Tree achieved 88.92%. Further model evaluation and optimization can be explored to enhance predictive performance.
