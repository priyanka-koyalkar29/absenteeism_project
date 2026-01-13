from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/absenteeism_cleaned.csv")
print("Cleaned data loaded for classification")


# Input features (drop targets)
X = df.drop(
    ["absenteeism_time_in_hours", "absenteeism_risk"],
    axis=1
)

# Classification target
y = df["absenteeism_risk"]

print("X shape:", X.shape)
print("y shape:", y.shape)

# TRAIN–TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# LOGISTIC REGRESSION
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

print("Logistic Regression Results")
print(confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

#DecisionTree Classifier 
DT_model = DecisionTreeClassifier()
DT_model.fit(X_train,y_train)
y_pred_DT = DT_model.predict(X_test)
print("Decision Tree classification:")
print(confusion_matrix(y_test,y_pred_DT))
print(classification_report(y_test,y_pred_DT))


#RandomForest Classifier 
random_model = RandomForestClassifier(n_estimators=10, random_state=42, class_weight="balanced")
random_model.fit(X_train,y_train)
y_pred_random = random_model.predict(X_test)
print("random forest classification:")
print(confusion_matrix(y_test,y_pred_random))
print(classification_report(y_test,y_pred_random))
