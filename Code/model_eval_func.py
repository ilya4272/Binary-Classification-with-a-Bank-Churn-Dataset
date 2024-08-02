# Data Manipulation and Cleaning Libraries
import pandas as pd  # For data manipulation and data frames
import numpy as np  # For numerical operations and arrays
import matplotlib.pyplot as plt


from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    log_loss,
)

# Function 1 -----------------------------------------------------------------------------------


def evaluate_model_performance(clf, X_train, X_test, y_train, y_test):
    # Modeli Eğitme ve Test Verileri Üzerinde Değerlendirme
    y_train_pred = clf.predict(X_train)
    y_train_proba = clf.predict_proba(X_train)[:, 1]
    y_test_pred = clf.predict(X_test)
    y_test_proba = clf.predict_proba(X_test)[:, 1]

    # Doğruluk (Accuracy) Hesaplama
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # ROC AUC Skoru Hesaplama
    train_roc_auc = roc_auc_score(y_train, y_train_proba)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)

    # Log Loss (Kayıp) Hesaplama
    train_log_loss = log_loss(y_train, y_train_proba)
    test_log_loss = log_loss(y_test, y_test_proba)

    # Performans Sonuçlarını Görselleştirme
    metrics = ["Accuracy", "ROC AUC", "Log Loss"]
    train_scores = [train_accuracy, train_roc_auc, train_log_loss]
    test_scores = [test_accuracy, test_roc_auc, test_log_loss]

    x = np.arange(len(metrics))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(5, 3))
    rects1 = ax.bar(
        x - width / 2, train_scores, width, label="Train", color="blue", alpha=0.6
    )
    rects2 = ax.bar(
        x + width / 2, test_scores, width, label="Test", color="red", alpha=0.6
    )

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Scores")
    ax.set_title("Scores by group and metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    fig.tight_layout()

    plt.show()


# Function 2 -----------------------------------------------------------------------------------


# Modeli değerlendirme fonksiyonu
def evaluate_model(clf, X_train, X_test, y_train, y_test):
    global benchmark_results

    # Modelin eğitim ve test setleri üzerindeki performansını değerlendirme
    y_train_proba = clf.predict_proba(X_train)[:, 1]
    y_test_proba = clf.predict_proba(X_test)[:, 1]

    # Eğitim seti performansı
    train_roc_auc = roc_auc_score(y_train, y_train_proba)

    # Test seti performansı
    test_roc_auc = roc_auc_score(y_test, y_test_proba)

    # Sonuçları yazdırma
    print(f"Training ROC AUC: {train_roc_auc}")
    print("")
    print(f"***Test ROC AUC: {test_roc_auc}***")


def evaluate_model_loss(clf, X_train, X_test, y_train, y_test):

    # Modeli eğit
    clf.fit(X_train, y_train)

    # Modelin eğitim ve test setleri üzerindeki performansını değerlendirme
    y_train_proba = clf.predict_proba(X_train)
    y_test_proba = clf.predict_proba(X_test)

    # Eğitim seti performansı
    train_log_loss = log_loss(y_train, y_train_proba)

    # Test seti performansı
    test_log_loss = log_loss(y_test, y_test_proba)

    # Sonuçları yazdırma
    print(f"Training Log Loss: {train_log_loss}")
    print("")
    print(f"***Test Log Loss: {test_log_loss}***")
