import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sqlalchemy import create_engine 

df = pd.read_csv("data/absenteeism_cleaned.csv")

# 1. Prepare X and y
X_reg = df.drop(["absenteeism_time_in_hours", "absenteeism_risk"], axis=1)
y_reg = df["absenteeism_time_in_hours"]

# 2. Train-test split 
X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)
# 3. Train best model (Random Forest)
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
# 4. Apply model to FULL data
df["predicted_absenteeism_hours"] = rf_model.predict(X_reg)



#classification 
X_clf = df.drop(
    ["absenteeism_time_in_hours", "absenteeism_risk"], axis=1
)
y_clf = df["absenteeism_risk"]

rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_clf, y_clf)

df["predicted_absenteeism_risk"] = rf_clf.predict(X_clf)



#clustering 
kmeans = KMeans(n_clusters=3, random_state=42)
df["employee_cluster"] = kmeans.fit_predict(X_reg)



engine = create_engine(
    "mssql+pyodbc://localhost\\SQLEXPRESS/AbsenteeismDB"
    "?driver=ODBC+Driver+17+for+SQL+Server"
    "&trusted_connection=yes"
)

df.to_sql(
    name="absenteeism_ml_results",
    con=engine,
    if_exists="replace",
    index=False
)

print("Final ML results stored in SQL Server")
