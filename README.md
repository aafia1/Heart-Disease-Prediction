
# 🫀 Heart Disease Prediction – AI/ML Internship Task 3

## 📌 Task Objective
Build a binary classification model to predict whether a person is at risk of heart disease based on their health-related attributes. This task focuses on applying machine learning to a real-world medical dataset, performing exploratory analysis, training models, and evaluating their performance.

## 📊 Dataset Used
- **Name**: [Heart Disease UCI Dataset](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci)
- **Source**: UCI Machine Learning Repository via Kaggle
- **Features**: Age, Sex, Chest Pain Type, Resting BP, Cholesterol, Fasting Blood Sugar, ECG results, Max Heart Rate, Exercise Angina, ST Depression, etc.
- **Target**: Binary indicator of heart disease presence (0 = No, 1 = Yes)

## 🧠 Models Applied
- **Logistic Regression**
- **Decision Tree Classifier**

Both models were trained using scikit-learn after standard preprocessing and feature scaling.

## 📈 Key Results and Findings
- Logistic Regression achieved a test **accuracy of ~85%** with a well-balanced ROC-AUC curve.
- Important features affecting prediction include:
  - **Chest Pain Type**
  - **Max Heart Rate Achieved**
  - **ST Depression (Oldpeak)**
  - **Number of Major Vessels**
- Confusion matrix and ROC curve were used for detailed model evaluation.

## 📂 Project Structure
```
heart_disease_prediction/
│
├── heart.csv                  ← Dataset file
├── heart_disease_predictor.py ← Python module for core ML functions
├── eda_visuals.py             ← Optional: EDA plots and visualizations
├── main_notebook.ipynb        ← Final notebook with full analysis
└── README.md                  ← This file
```

## 👩‍💻 Author
**Aafia Azhar**  
GitHub: [@aafia1](https://github.com/aafia1)
