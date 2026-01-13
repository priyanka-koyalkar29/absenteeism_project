#extraction 
from sqlalchemy import create_engine
import pyodbc
import pandas as pd
df = pd.read_csv("data/Absenteeism_at_work.csv",sep =";")
print("dataset loaded successfully")
print("shape(rows,columns):", df.shape)
print("\n columns names:")
print(df.columns)
print("\n first five rows")
print(df.head())

#transformation 
print("\n missing values per columns")
print(df.isnull().sum())
print("\n datatypes:")
print(df.dtypes)
print("\n basics statistics:")
print(df.describe())

#standardize the column names 
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("/", "_")
)

#verification after standardization 
print("\nColumn names AFTER standardization:")
print(df.columns)

# Create absenteeism risk category based on hours
def risk_level(hours):
    if hours <= 8:
        return "low"
    elif hours <= 24:
        return "medium"
    else: 
        return  "high"

df["absenteeism_risk"] = df["absenteeism_time_in_hours"].apply(risk_level)
#verify new column 
print(df[["absenteeism_time_in_hours", "absenteeism_risk"]].head(10))

# Prepare data for machine learning (drop identifier)
df_ml = df.drop(columns=["id"])

# Prepare data for SQL & dashboard (keep all columns)
df_sql = df.copy()

print("\nML dataset shape:", df_ml.shape)
print("SQL dataset shape:", df_sql.shape)



# Load the transformed absenteeism data into SQL Server for analytics and dashboards

connection_string = (
    "mssql+pyodbc://localhost\\SQLEXPRESS/AbsenteeismDB"
    "?driver=ODBC+Driver+17+for+SQL+Server"
    "&trusted_connection=yes"
)

engine = create_engine(connection_string)

df_sql.to_sql(
    name="absenteeism_cleaned",
    con=engine,
    if_exists="replace",
    index=False
)

print("Data successfully loaded into SQL Server.")

# Save cleaned data for ML usage
df_sql.to_csv("data/absenteeism_cleaned.csv", index=False)

print("Cleaned CSV saved for ML.")
