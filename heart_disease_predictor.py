
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    """Loads the heart disease dataset from a CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Splits and scales the data for training."""
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train, y_train, model_type='logistic'):
    """Trains a logistic regression or decision tree model."""
    if model_type == 'logistic':
        model = LogisticRegression()
    elif model_type == 'tree':
        model = DecisionTreeClassifier()
    else:
        raise ValueError("model_type must be 'logistic' or 'tree'")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluates the model using accuracy, confusion matrix, and ROC curve."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    print(f"Accuracy: {acc:.2f}")
    print("Confusion Matrix:")
    print(cm)
    print(f"ROC AUC Score: {roc_auc:.2f}")

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def display_feature_importance(model, feature_names):
    """Displays feature importance for the given model."""
    if hasattr(model, 'coef_'):
        importances = model.coef_[0]
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        print("Model does not support feature importance.")
        return

    feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_df = feature_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=feature_df, x='Importance', y='Feature', palette='Blues_r')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()
