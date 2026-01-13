import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

# Load cleaned data (exported from ETL)
df = pd.read_csv("data/absenteeism_cleaned.csv")

print("Data loaded for regression")
print(df.shape)


# Target variable (what I want to predict)
y = df["absenteeism_time_in_hours"]

# Features (drop target and derived classification column)
X = df.drop(
    columns=["absenteeism_time_in_hours", "absenteeism_risk"]
)

print("Target and features created")
print("X shape:", X.shape)
print("y shape:", y.shape)


#Train-Test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Train-test split done")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)



# Create the baseline model
linear_model = LinearRegression()
# Train the model using training data
linear_model.fit(X_train, y_train)
print("Linear Regression model trained successfully")
# Predict absenteeism hours for test data
y_pred = linear_model.predict(X_test)
print("Predictions generated for test data")

#Evaluation 
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Linear Regression Evaluation Results")
print("MAE:", mae)
print("R2 Score:", r2)

#visualization 
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Absenteeism Hours")
plt.ylabel("Predicted Absenteeism Hours")
plt.title("Linear Regression: Actual vs Predicted")

#Ridge 
ridge_model = Ridge()
ridge_model.fit(X_train , y_train)
y_pred_ridge = ridge_model.predict(X_test)
mae_ridge = mean_absolute_error(y_test,y_pred_ridge)
r2_ridge = r2_score(y_test,y_pred_ridge)
print("Ridge Regression Results")
print("MAE:", mae)
print("R2 Score:", r2)

#visualization 
plt.figure()
plt.scatter(y_test, y_pred_ridge)
plt.xlabel("Actual Absenteeism Hours")
plt.ylabel("Predicted Absenteeism Hours")
plt.title("Ridge Regression: Actual vs Predicted")

#lasso .
lasso_model = Lasso()
lasso_model = lasso_model.fit(X_train,y_train)
y_pred_lasso = lasso_model.predict(X_test)
mae_lasso = mean_absolute_error(y_test,y_pred_lasso)
r2_lasso = r2_score(y_test,y_pred_lasso)
print("Lasso Regression Results")
print("MAE:", mae_lasso)
print("R2 Score:", r2_lasso)

#visualization 
plt.figure()
plt.scatter(y_test,y_pred_lasso)
plt.xlabel("Actual Absenteeism Hours")
plt.ylabel("Predicted Absenteeism Hours")
plt.title("Lasso Regression: Actual vs Prediction")

#DecisionTreeRegressor 
decisionTree_model = DecisionTreeRegressor()
decisionTree_model.fit(X_train,y_train)
y_pred_decisiontree = decisionTree_model.predict(X_test)
mae_decisiontree = mean_absolute_error(y_test,y_pred_decisiontree)
r2_decisiontree = r2_score(y_test,y_pred_decisiontree)
print("DecisionTreeRegressor results")
print("MAE:",mae_decisiontree)
print("r2 score:", r2_decisiontree)

#visualization 
plt.figure()
plt.scatter(y_test,y_pred_decisiontree)
plt.xlabel("Actual Absenteeism Hours")
plt.ylabel("Predicted Absenteeism Hours")
plt.title("Decision Tree : Actual vs Prediction")

#RandomForest 
randomforest_model = RandomForestRegressor(random_state=42)
randomforest_model.fit(X_train,y_train)
y_pred_randomforest = randomforest_model.predict(X_test)
mae_randomforest = mean_absolute_error(y_test,y_pred_randomforest)
r2_randomforest = r2_score(y_test,y_pred_randomforest)
print("Random Forest results")
print("MAE:", mae_randomforest)
print("r2 score:", r2_randomforest)

#visualization 
plt.figure()
plt.scatter(y_test,y_pred_randomforest)
plt.xlabel("Actual Absenteeism Hours")
plt.ylabel("Predicted Absenteeism Hours")
plt.title("Random Forest: Actual vs Prediction")



# Get feature importance values
importances = randomforest_model.feature_importances_

# Create DataFrame
feature_importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": importances
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(
    by="Importance", ascending=False
)

print(feature_importance_df)

#visualization 
plt.figure(figsize=(10, 6))
plt.barh(
    feature_importance_df["Feature"],
    feature_importance_df["Importance"]
)
plt.gca().invert_yaxis()
plt.xlabel("Importance Score")
plt.title("Feature Importance - Random Forest")
plt.show()






