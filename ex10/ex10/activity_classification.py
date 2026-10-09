import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 2. Split features and target
X_train = train.drop(["Activity", "subject"], axis=1)
y_train = train["Activity"]
X_test = test.drop(["Activity", "subject"], axis=1)
y_test = test["Activity"]

# 3. Encode labels
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# 4. Scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 5. PCA (keep 95% variance)
pca = PCA(n_components=0.95, random_state=42)
X_train_p = pca.fit_transform(X_train_s)
X_test_p = pca.transform(X_test_s)
print("Original features :", X_train_s.shape[1])
print("After PCA features:", X_train_p.shape[1])
print("Variance retained : {:.2f}%".format(pca.explained_variance_ratio_.sum() * 100))

# 6. Train Logistic Regression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_p, y_train_enc)

# 7. Predict
y_pred = model.predict(X_test_p)

# 8. Evaluate
acc = accuracy_score(y_test_enc, y_pred)
print("\nAccuracy: {:.2f}%".format(acc * 100))
print("\nClassification Report:\n")
print(classification_report(y_test_enc, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_test_enc, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix - Activity Classification")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Predict a single sample
sample = X_test_p[0].reshape(1, -1)
print("Predicted activity:", le.inverse_transform(model.predict(sample))[0])
print("Actual activity   :", y_test.iloc[0])
