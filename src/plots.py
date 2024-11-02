import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_data_distributions(data):
    plt.figure(figsize=(14, 10))
    numerical_features = [
        "App Usage Time (min/day)",
        "Screen On Time (hours/day)",
        "Battery Drain (mAh/day)",
        "Number of Apps Installed",
        "Data Usage (MB/day)",
        "Age",
    ]
    for i, feature in enumerate(numerical_features, 1):
        plt.subplot(3, 2, i)
        sns.histplot(data[feature], kde=True, color="skyblue")
        plt.title(f"Distribusi {feature}")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, X_test, y_test):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, model.predict(X_test))
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=np.unique(y_test),
        yticklabels=np.unique(y_test),
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.show()
    return cm


def plot_feature_importance(model, features):
    feature_importances = model.feature_importances_
    plt.figure(figsize=(10, 6))

    # Create the bar plot without hue or legend
    sns.barplot(x=feature_importances, y=features, palette="viridis")

    plt.title("Pentingnya Fitur dalam Model Random Forest")
    plt.show()
