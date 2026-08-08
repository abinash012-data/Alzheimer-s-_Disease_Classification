# Alzheimer-s-_Disease_Classification
Machine learning-based Alzheimer’s disease prediction using clinical and behavioral features
# 🧠 Alzheimer’s Disease Prediction Using Machine Learning

A machine learning project for predicting **Alzheimer’s disease diagnosis** using clinical, behavioral, cognitive, and health-related features.

The project performs exploratory data analysis, feature selection, and comparative evaluation of multiple classification algorithms. Among the models tested, **Random Forest achieved the highest test accuracy of 95.35%** and was selected as the final model.

> **Important:** This project is intended for educational and research purposes only and is not a medical diagnostic tool.

---

## 📌 Project Overview

Alzheimer’s disease is a progressive neurological condition associated with changes in memory, cognitive function, and daily activities.

This project investigates whether machine learning can identify patterns in patient-related data that are associated with an Alzheimer's disease diagnosis.

The workflow is:

```text
Patient Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Feature Selection
      │
      ▼
Train / Test Split
      │
      ▼
Multiple ML Models
      │
      ├── Logistic Regression
      ├── Decision Tree
      ├── Random Forest
      └── XGBoost
      │
      ▼
Model Comparison
      │
      ▼
Random Forest Selected
      │
      ▼
Prediction
```

---

## 🎯 Objectives

The main objectives are:

* Analyze the Alzheimer's disease dataset.
* Identify important features associated with the diagnosis.
* Explore relationships between clinical and behavioral variables.
* Train multiple classification models.
* Compare model performance.
* Select the best-performing classifier.
* Save the final trained model for future predictions.

---

## 📊 Dataset

The project uses:

```text
alzheimers_disease_data.csv
```

The dataset contains:

* **2,149 patient records**
* **35 columns**
* **No missing values**
* **No duplicate records**

The original dataset contains patient information including demographic, lifestyle, cardiovascular, cognitive, behavioral, and functional assessment variables.

The target variable is:

```text
Diagnosis
```

where the model predicts the patient's diagnosis class.

---

## 🧹 Data Preprocessing

Two columns are removed before model development:

```text
PatientID
DoctorInCharge
```

`PatientID` is an identifier rather than a predictive feature, while `DoctorInCharge` is not used in the modeling process.

The remaining variables are used for exploratory analysis and feature selection.

---

## 🔎 Exploratory Data Analysis

The notebook investigates:

* Dataset dimensions
* Data types
* Missing values
* Duplicate values
* Descriptive statistics
* Outliers
* Diagnosis distribution
* Age distribution
* MMSE distribution
* Age vs Diagnosis
* MMSE vs Diagnosis
* Memory Complaints vs Diagnosis
* Feature correlations

### Important observations

The analysis found:

* No missing values.
* No duplicate records.
* The diagnosis classes are sufficiently represented for classification.
* Lower MMSE scores are associated with the Alzheimer's diagnosis in the analyzed data.
* Memory complaints are associated with the diagnosis.
* Functional assessment and ADL show relatively strong relationships with the target compared with many other variables.

---

## 🧠 Feature Selection

Instead of using every available variable, the project calculates correlations between numerical features and the target variable.

The **15 features with the highest absolute correlation with `Diagnosis`** are selected.

### Selected Features

```text
1. FunctionalAssessment
2. ADL
3. MemoryComplaints
4. MMSE
5. BehavioralProblems
6. SleepQuality
7. EducationLevel
8. CholesterolHDL
9. Hypertension
10. FamilyHistoryAlzheimers
11. CholesterolLDL
12. Diabetes
13. CardiovascularDisease
14. BMI
15. Disorientation
```

These features are then used as the input variables for the classification models.

---

## ✂️ Train-Test Split

The dataset is divided into training and testing sets using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

This creates an **80/20 train-test split** while preserving the class distribution.

The resulting test set contains **430 observations**.

---

# 🤖 Machine Learning Models

Four classification algorithms are evaluated.

## 1. Logistic Regression

```python
LogisticRegression(
    random_state=42,
    solver="liblinear"
)
```

Test accuracy:

```text
81.40%
```

---

## 2. Decision Tree

```python
DecisionTreeClassifier(
    random_state=42
)
```

Test accuracy:

```text
88.60%
```

---

## 3. Random Forest

```python
RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
```

Test accuracy:

```text
95.35%
```

Random Forest produced the best performance among the four evaluated models.

---

## 4. XGBoost

```python
XGBClassifier(
    eval_metric="logloss"
)
```

Test accuracy:

```text
94.19%
```

---

# 📈 Model Comparison

| Model               |   Accuracy |  Precision |     Recall |   F1-Score |
| ------------------- | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression |     81.40% |     81.40% |     81.40% |     81.40% |
| Decision Tree       |     88.60% |     88.66% |     88.60% |     88.63% |
| **Random Forest**   | **95.35%** | **95.34%** | **95.35%** | **95.33%** |
| XGBoost             |     94.19% |     94.17% |     94.19% |     94.16% |

### Best Model

**Random Forest** achieved the highest test accuracy:

```text
95.35%
```

Therefore, it was selected as the final model.

---

# 📊 Random Forest Evaluation

The Random Forest confusion matrix is:

```text
[[270,   8],
 [ 12, 140]]
```

This represents the predictions made on the 430-sample test set.

The classification report gives:

```text
Accuracy  : 95.35%
Precision : 95.34%
Recall    : 95.35%
F1-Score  : 95.33%
```

The model achieved strong performance for both diagnosis classes on the held-out test data.

---

# 💾 Saved Model

The final Random Forest model is saved using Joblib:

```python
joblib.dump(rf, "random_forest_model.pkl")
```

The model artifact should be stored in:

```text
models/
└── random_forest_model.pkl
```

This allows the trained model to be reused without retraining it every time.

---

# ⚠️ Risk Categorization

The notebook also defines a probability-based risk categorization:

```text
Probability < 0.30
→ Low Risk

0.30 ≤ Probability < 0.70
→ Medium Risk

Probability ≥ 0.70
→ High Risk
```

The categories are derived from the Random Forest prediction probabilities.

**These categories should not be interpreted as clinical risk assessments.** They are simply probability thresholds implemented in the project for demonstration purposes.

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* XGBoost

### Model Persistence

* Joblib

### Development Environment

* Jupyter Notebook

---

# 📁 Project Structure

```text
Alzheimers-Disease-Prediction/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── alzheimers_disease_data.csv
│
├── notebooks/
│   └── alzheimers_disease_prediction.ipynb
│
├── models/
│   └── random_forest_model.pkl
│
└── screenshots/
    ├── diagnosis_distribution.png
    ├── correlation_matrix.png
    ├── model_comparison.png
    ├── app_interface

```

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Alzheimers-Disease-Prediction.git
cd Alzheimers-Disease-Prediction
```

Replace `<your-username>` with your GitHub username.

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
notebooks/alzheimers_disease_prediction.ipynb
```

If the notebook is run from the repository root, load the dataset using:

```python
df = pd.read_csv("data/alzheimers_disease_data.csv")
```

---

# 🔮 Future Improvements

Possible improvements include:

* Perform cross-validation rather than relying on a single train-test split.
* Tune Random Forest hyperparameters.
* Compare additional machine learning algorithms.
* Use ROC-AUC and Precision-Recall curves.
* Perform feature importance analysis.
* Investigate SHAP-based model explainability.
* Add a prediction interface using Streamlit.
* Save the complete preprocessing and prediction pipeline.
* Evaluate model calibration.
* Investigate class-specific performance in greater detail.
* Use clinically validated datasets and appropriate external validation.

---

# ⚠️ Disclaimer

This project is an **educational machine learning project**.

The predictions generated by this model should **not be used to diagnose Alzheimer's disease or make clinical decisions**.

A machine learning model requires appropriate clinical validation, external validation, calibration, and regulatory consideration before it could be considered for real-world medical use.

---

# 📌 Conclusion

This project demonstrates the application of machine learning to Alzheimer's disease classification using patient-related clinical, behavioral, cognitive, and health features.

Four classification algorithms were compared:

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost

Among the evaluated models, **Random Forest achieved the best performance with 95.35% accuracy on the test set**.

The final Random Forest model was saved as a Joblib artifact for potential reuse.
