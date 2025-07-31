
import matplotlib.pyplot as plt
import seaborn as sns

def plot_class_distribution(df):
    """Plots the class distribution of the target variable."""
    sns.countplot(data=df, x='target', palette='Set2')
    plt.title('Target Class Distribution')
    plt.xlabel('Heart Disease (1 = Yes, 0 = No)')
    plt.ylabel('Count')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df):
    """Plots the correlation heatmap of the dataset."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()
