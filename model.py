import pandas as pd
import joblib
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


def train_and_save():
    # Load data
    df = pd.read_csv('data/disease_prediction.csv')
    df.drop(columns=['patient_id'], inplace=True)

    # Separate features and target
    X, y = df.drop(columns=['disease']), df['disease']
    y = y.map({'Yes': 1, 'No': 0})

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define columns
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['str', 'category']).columns.tolist()

    # Build preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), numeric_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
        ]
    )

    # Pipeline 1: Logistic Regression
    pipeline_lr = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(max_iter=1000))
    ])

    # Pipeline 2: Random Forest
    pipeline_rf = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10))
    ])

    # Train
    pipeline_lr.fit(X_train, y_train)
    pipeline_rf.fit(X_train, y_train)

    # Predict
    y_pred_lr = pipeline_lr.predict(X_test)
    y_pred_rf = pipeline_rf.predict(X_test)

    # Evaluate Logistic Regression
    y_pred_train_lr = pipeline_lr.predict(X_train)
    lr_train_acc = accuracy_score(y_train, y_pred_train_lr)
    lr_test_acc  = accuracy_score(y_test,  y_pred_lr)
    lr_train_auc = roc_auc_score(y_train, pipeline_lr.predict_proba(X_train)[:, 1])
    lr_test_auc  = roc_auc_score(y_test,  pipeline_lr.predict_proba(X_test)[:, 1])


    print("Logistic Regression")
    print(f"  Accuracy — Train: {lr_train_acc:.4f} | Test: {lr_test_acc:.4f} | Gap: {lr_train_acc - lr_test_acc:+.4f}")
    print(f"  ROC-AUC  — Train: {lr_train_auc:.4f} | Test: {lr_test_auc:.4f} | Gap: {lr_train_auc - lr_test_auc:+.4f}")
    print(f"\n  Classification Report (Test):\n{classification_report(y_test, y_pred_lr)}")
    print(f"  Confusion Matrix (Test):\n{confusion_matrix(y_test, y_pred_lr)}")


    # Evaluate Random Forest
    y_pred_train_rf = pipeline_rf.predict(X_train)
    rf_train_acc = accuracy_score(y_train, y_pred_train_rf)
    rf_test_acc  = accuracy_score(y_test,  y_pred_rf)
    rf_train_auc = roc_auc_score(y_train, pipeline_rf.predict_proba(X_train)[:, 1])
    rf_test_auc  = roc_auc_score(y_test,  pipeline_rf.predict_proba(X_test)[:, 1])



    print("\nRandom Forest")
    print(f"  Accuracy — Train: {rf_train_acc:.4f} | Test: {rf_test_acc:.4f} | Gap: {rf_train_acc - rf_test_acc:+.4f}")
    print(f"  ROC-AUC  — Train: {rf_train_auc:.4f} | Test: {rf_test_auc:.4f} | Gap: {rf_train_auc - rf_test_auc:+.4f}")
    print(f"\n  Classification Report (Test):\n{classification_report(y_test, y_pred_rf)}")
    print(f"  Confusion Matrix (Test):\n{confusion_matrix(y_test, y_pred_rf)}")

    # Save best model (RF wins on test metrics)
    joblib.dump(pipeline_rf, 'model.pkl')

    # Save metrics for /model/info endpoint
    metrics = {
        'trained_at':      datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_accuracy':   round(rf_test_acc,                           4),
        'test_roc_auc':    round(rf_test_auc,                           4),
        'test_precision':  round(precision_score(y_test, y_pred_rf),    4),
        'test_recall':     round(recall_score(y_test, y_pred_rf),       4),
        'test_f1':         round(f1_score(y_test, y_pred_rf),           4),
        'train_accuracy':  round(rf_train_acc,                          4),
        'train_roc_auc':   round(rf_train_auc,                          4),
    }
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)

    print("\nModel saved to model.pkl")
    print("Metrics saved to metrics.json")

    return metrics


if __name__ == '__main__':
    train_and_save()
