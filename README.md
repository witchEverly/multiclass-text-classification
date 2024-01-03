
## Data Description

Some imbalanced kaggle dataset.

Data source: https://www.kaggle.com/datasets/surajjha101/myntra-reviews-on-women-dresses-comprehensive/code?datasetId=2265536

## Modeling Details

Multiclass classification problem.

## Model 1: Logistic Regression

Models to consider: one-vs-rest, multinomial with L1 and L2 regularization.

Softmax to approximate indicator function.

LogisticRegressionCV

Notes:
- Regularization is applied by default with parameter 'C' (inverse of regularization strength) set to 1.0. The higher the value of 'C', the less the regularization.

### Model 1.1: Logistic Regression with TF-IDF

Parameters to consider:

- 'C' = [0.01, 0.1, 1, 10, 100] (default = 1.0)
- 'penalty' = ['l1', 'l2']
- class_weight = ['balanced', None]
- solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'] (default = 'lbfgs', but this solver does not support L1 regularization, 'newton-cg' and 'sag' only support l2). Sag and saga fast for large datasets, but make sure they are scaled. 
- 'multi_class' = ['ovr', 'multinomial', 'auto']



#### Model 1.1.1: Logistic Regression with TF-IDF and GridSearchCV



