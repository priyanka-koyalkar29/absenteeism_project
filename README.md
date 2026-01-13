# Employee Absenteeism Analysis (ML + SQL + Tableau)

This project analyzes employee absenteeism data and builds Machine Learning models to:
- Predict absenteeism risk (Low / Medium / High)
- Predict absenteeism hours
- Cluster employees into similar groups
- Visualize insights using an interactive Tableau dashboard

---

##  Project Highlights
✅ Data Cleaning + Feature Engineering (ETL in Python)  
✅ ML Models:
- Classification (Absenteeism Risk)
- Regression (Absenteeism Hours)
- Clustering (Employee Segments)  
✅ SQL Queries for analysis  
✅ Tableau Dashboard (final visualization)

---

## 📂 Folder Structure

ABSENTEEISM_PROJECT/
│
├── data/
│ ├── Absenteeism_at_work.csv
│ ├── absenteeism_cleaned.csv
│ ├── absenteeism_ml_results.csv
│ └── Attribute Information.docx
│
├── etl/
│ └── etl_pipeline.py
│
├── ml/
│ ├── classification.py
│ ├── regression.py
│ ├── clustering.py
│ └── final_ml_output.py
│
├── sql/
│ └── queries.sql
│
└── README.md

---

## 🧠 Machine Learning Tasks

### 1️⃣ Classification (Absenteeism Risk)
Goal: Predict whether an employee has **Low / Medium / High** absenteeism risk.

### 2️⃣ Regression (Absenteeism Hours)
Goal: Predict the number of hours an employee may be absent.

### 3️⃣ Clustering (Employee Groups)
Goal: Segment employees into clusters for better HR decision-making.

---

## 📊 Tableau Dashboard
The dashboard includes:
- Risk Distribution (Low / Medium / High)
- Avg Absenteeism by Risk
- Actual vs Predicted (ML)
- Employee Clusters

🔗 **Tableau Public Link:** 

https://public.tableau.com/app/profile/k.p.priyanka/viz/AbsenteeismAnalysisDashboard_17682918529790/Dashboard1

## 🛠️ Tools & Technologies Used
- Python (Pandas, NumPy, Scikit-learn)
- SQL (queries + analysis)
- Tableau Public (dashboard visualization)
- VS Code

---

## 🚀 How to Run the Project

### 1) Install Dependencies
```bash 
pip install -r requirements.txt
```
### 2) Run ETL (Data Cleaning)
```bash 
python etl/etl_pipeline.py
```
 ### 3) Run ML Models
```bash
python ml/classification.py
python ml/regression.py
python ml/clustering.py
```
 ### 4) Generate Final ML Output File
 ```bash
python ml/final_ml_output.py
```
## 📌 Final Output Files
After running the project, you will get: 
- 'absenteeism_cleaned.csv → cleaned dataset

- 'absenteeism_ml_results.csv → predictions + clusters


👩‍💻 Author
K P Priyanka
MCA Graduate | AI/ML + Data Projects  
