# 🩺 PIMA Indian Diabetes Prediction

This project uses the **PIMA Indian Diabetes Dataset** to predict whether a patient has diabetes based on diagnostic health measurements. The dataset is sourced from the **UCI Machine Learning Repository** and is commonly used for binary classification problems in medical data science.

---

## 📚 Dataset Overview

The dataset consists of **768 observations** with **8 numerical input features** and **1 binary target variable** (`Outcome`).

### 🎯 Target Variable:
- `Outcome`:  
  - `0` = Non-diabetic  
  - `1` = Diabetic

### 📌 Features:
| Feature | Description |
|---------|-------------|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration (2 hours in OGTT) |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skin fold thickness (mm) |
| `Insulin` | 2-Hour serum insulin (mu U/ml) |
| `BMI` | Body mass index (weight in kg/(height in m)^2) |
| `DiabetesPedigreeFunction` | Diabetes pedigree function (family history) |
| `Age` | Age in years |

---

## 📈 Project Workflow

1. **Data Cleaning**
   - Dropped biologically implausible 0 values (for `Glucose`, `BloodPressure`, etc.)
   - Optionally dropped rows with invalid medical entries

2. **Exploratory Data Analysis (EDA)**
   - Visualized distribution of each feature against `Outcome`
   - Identified skewed features and relationships

3. **Preprocessing**
   - Scaled features using StandardScaler
   - Addressed class imbalance using SMOTE-ENN

4. **Model Training**
   - Trained multiple models: Logistic Regression, Random Forest, Gradient Boosting, etc.
   - Performed hyperparameter tuning

5. **Evaluation**
   - Compared models using accuracy, precision, recall, F1-score, and AUC-ROC
   - Visualized confusion matrix and ROC curve

---

## 🧪 Models Used

- Random Forest
- Gradient Boosting
- K-Neighbors Classifier
- Support Vector Classifier
- AdaBoost Classifier
- CatBoosting Classifier
- Decision Tree
- XGBClassifier 
- Logistic Regression

---

## 📊 Results

Check the FE_and_ModelTraining_on_Pima_Indians_Diabetes.ipynb notebook for results comparison on each models

---

## 🛠️ Dependencies

- ipykernel
- pandas
- numpy
- matplotlib
- plotly
- seaborn
- scipy
- scikit-learn
- imblearn
- xgboost
- catboost
- statsmodels
 

```bash
pip install -r requirements.txt
