import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import StandardScaler, RobustScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import f1_score

print("start read_csv")
df = pd.read_csv('../data/2022-10.csv')
print("read_csv complete")

df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Week'] = df['Date'].dt.isocalendar().week

columns_to_drop = ['Time', 'Date']
df.drop(columns=columns_to_drop, inplace=True)

outgoing_count = df['Sender_account'].value_counts()
incoming_count = df['Receiver_account'].value_counts()


def calculate_counts(row):
    # out_count = outgoing_count.get(row['Sender_account'], 0)
    # in_count = incoming_count.get(row['Receiver_account'], 0)
    total_sender_count = outgoing_count.get(row['Sender_account'], 0) + incoming_count.get(row['Sender_account'], 0)
    total_receiver_count = outgoing_count.get(row['Receiver_account'], 0) + incoming_count.get(row['Receiver_account'], 0)

    return pd.Series({
        'out_same_count': total_sender_count,
        'in_same_count': total_receiver_count
    })


df[['out_same_count', 'in_same_count']] = df.apply(calculate_counts, axis=1)

# print(df)

columns_labeled = ["Is_laundering", "Laundering_type"]
X = df.drop(columns=columns_labeled)
y = df["Is_laundering"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

numerical_features = X.select_dtypes(exclude="object").columns
categorical_features = X.select_dtypes(include="object").columns

transformer = ColumnTransformer(transformers=[
    ("OrdinalEncoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features),
    ("RobustScaler", RobustScaler(), numerical_features)
], remainder="passthrough")

X_train = transformer.fit_transform(X_train)
X_test = transformer.transform(X_test)

print(X_test[0])
# what is X_test

param_grid = {
    'max_depth': [16],
    'eta': [0.1],
}

xgb = XGBClassifier(eval_metric='logloss', random_state=42)

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=4,
    verbose=2
)

grid_search.fit(X_train, y_train)

print("Best Parameters: ", grid_search.best_params_)
best_model = grid_search.best_estimator_
test_probabilities = best_model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, test_probabilities)
print("Test AUC: ", test_auc)

fpr, tpr, thresholds = roc_curve(y_test, test_probabilities)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % test_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

desired_tpr = 0.90
closest_threshold = thresholds[np.argmin(np.abs(tpr - desired_tpr))]

y_pred = (test_probabilities >= closest_threshold).astype(int)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title(f'Confusion Matrix at {desired_tpr * 100}% TPR')
plt.show()

tn, fp, fn, tp = cm.ravel()
fpr_cm = fp / (fp + tn)
tpr_cm = tp / (tp + fn)

print(f"False Positive Rate (FPR): {fpr_cm:.3f}")
print(f"True Positive Rate (TPR): {tpr_cm:.3f}")

print(classification_report(y_test, y_pred))
