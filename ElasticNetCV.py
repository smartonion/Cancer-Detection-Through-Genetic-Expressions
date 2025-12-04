#========================================================
# Big Picture: 
# This model takes a gene expression dataset 
# with a binary outcome (control vs cancer). Then 
# it splits it into training and testing sets. Then
# for many values of alpha and l1_ratio, it runs 
# Cross-Validation with a logistic regression model,
# a grid of lambda values, and a grid of alpha values,
# 5-fold cross-validation, and uses classification error
# to select the best model.
# Note: This was originally written in R. The python
#       version showed better results on this dataset.
#========================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

RANDOM_STATE = 123

gene = pd.read_csv('gene.txt', sep=' ')

Y = (gene["class"]=="cancer").astype(int)
X = gene.drop(columns=["class"]).to_numpy()

x_tr, x_te, y_tr, y_te = train_test_split(
    X, 
    Y, 
    train_size=0.7, 
    random_state=RANDOM_STATE,
    stratify=Y
    )
scaler = StandardScaler()
x_tr_scaled = scaler.fit_transform(x_tr)
x_te_scaled = scaler.transform(x_te)

alphas = [0.1737] # np.linspace(0, 0.3, 20)
logistic_regression = LogisticRegressionCV(
    penalty="elasticnet",
    solver="saga",
    l1_ratios=alphas,
    cv=5,
    Cs=200,
    scoring="f1",
    max_iter=10000,
    n_jobs=-1,
    refit=True
)
logistic_regression.fit(x_tr_scaled, y_tr)

pred_tr = logistic_regression.predict(x_tr_scaled)
pred_te = logistic_regression.predict(x_te_scaled)

cm = confusion_matrix(y_te, pred_te,labels=np.unique(Y))
accuracy = accuracy_score(y_te, pred_te)
err = np.mean(pred_te != y_te)
best_alpha = float(logistic_regression.l1_ratio_[0])
best_C = float(logistic_regression.C_[0])
best_lambda = 1.0 / best_C
print(f"Accuracy: {accuracy:.4f}")
print(f"Confusion Matrix: \n{cm}")
print(f"Error: {err:.4f}")
print(f"Chosen alpha: {best_alpha:.4f}  lambda: {best_lambda:.3g}")

# ------------------------------------------------------------------------------
feature_names = gene.drop(columns=["class"]).columns.to_numpy()
coefs = logistic_regression.coef_.ravel()
classes = logistic_regression.classes_
positive_class = classes[1]

print("Model is giving log odds for class:", positive_class)

pos_idx = np.argmax(coefs)
neg_idx = np.argmin(coefs)

top_pos_gene = feature_names[pos_idx]
top_neg_gene = feature_names[neg_idx]
top_pos_coef = coefs[pos_idx]
top_neg_coef = coefs[neg_idx]
coefs = logistic_regression.coef_.ravel()

nonzero_mask = np.abs(coefs) > 1e-6

n_nonzero = np.sum(nonzero_mask)
print("Number of non zero coefficient genes:", n_nonzero)

gene_names = gene.drop(columns=["class"]).columns.to_numpy()
nonzero_genes = gene_names[nonzero_mask]

print(f"Top positive gene: {top_pos_gene}  | log odds: {top_pos_coef:.4f}")
print(f"Top negative gene: {top_neg_gene}  | log odds: {top_neg_coef:.4f}")