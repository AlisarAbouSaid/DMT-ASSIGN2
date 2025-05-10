import pandas as pd
import pickle
import os
import gc
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib

matplotlib.use('TkAgg')  # Use TkAgg backend for GUI plotting
import matplotlib.pyplot as plt


# --------------- Data Preprocessing Functions -----------------

def load_data(filename):
    return pd.read_csv(filename)


def add_date_features(df):
    if 'date_time' in df.columns:
        df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')
        df['year'] = df['date_time'].dt.year
        df['month'] = df['date_time'].dt.month
        df['day'] = df['date_time'].dt.day
        df['hour'] = df['date_time'].dt.hour
    return df


def normalize_features(df):
    if 'srch_query_affinity_score' in df.columns:
        df['srch_query_affinity_score'] = df['srch_query_affinity_score'].fillna(df['srch_query_affinity_score'].mean())
        df['srch_query_affinity_score'] = (df['srch_query_affinity_score'] - df['srch_query_affinity_score'].mean()) / \
                                          df['srch_query_affinity_score'].std()
    if 'position' in df.columns:
        df['position'] = (df['position'] - df['position'].mean()) / df['position'].std()
    return df


def add_custom_features(df):
    # Add price difference to visitor history feature
    if 'price_usd' in df.columns and 'visitor_hist_adr_usd' in df.columns:
        df['price_diff_to_user_hist'] = df['price_usd'] - df['visitor_hist_adr_usd']
    return df


def add_price_per_star(df):
    if 'price_usd' in df.columns and 'prop_starrating' in df.columns:
        df['price_per_star'] = df['price_usd'] / df['prop_starrating'].replace(0, np.nan)
        df['price_per_star'] = df['price_per_star'].fillna(df['price_per_star'].mean())
    return df

def fill_missing(df):
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].mean())
    return df


def preprocess_training_data(df, kind="train"):
    drop_competitor_cols = [
        'comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff',
        'comp2_rate', 'comp2_inv', 'comp2_rate_percent_diff',
        'comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff',
        'comp4_rate', 'comp4_inv', 'comp4_rate_percent_diff',
        'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff',
        'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff',
        'comp7_rate', 'comp7_inv', 'comp7_rate_percent_diff',
        'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff'
    ]
    df = df.drop(columns=[col for col in drop_competitor_cols if col in df.columns], errors='ignore')
    df = add_date_features(df)
    df = add_custom_features(df)
    df = add_price_per_star(df)
    df = normalize_features(df)
    df = fill_missing(df)

    labels = None
    if kind == "train" and 'click_bool' in df.columns:
        labels = df['click_bool'].copy()
        df = df.drop(columns=['click_bool'])

    return df, labels


# ------------------- Main -------------------

if __name__ == "__main__":
    print("Starting training pipeline...")

    # Paths
    train_file = "training_set_VU_DM.csv"
    test_file = "test_set_VU_DM.csv"

    # Load data
    train_df = load_data(train_file)
    test_df = load_data(test_file)

    # Preprocess
    X_train, y_train = preprocess_training_data(train_df, kind="train")
    X_test, _ = preprocess_training_data(test_df, kind="test")

    # Keep srch_id and prop_id for submission
    srch_ids_test = X_test['srch_id'].copy() if 'srch_id' in X_test.columns else None
    prop_ids_test = X_test['prop_id'].copy() if 'prop_id' in X_test.columns else None

    # Drop non-feature columns before training
    drop_cols = [col for col in ['srch_id', 'prop_id', 'date_time'] if col in X_train.columns and col in X_test.columns]
    X_train_model = X_train.drop(columns=drop_cols, errors='ignore')
    X_test_model = X_test.drop(columns=drop_cols, errors='ignore')

    # Align columns
    missing_cols = set(X_train_model.columns) - set(X_test_model.columns)
    for col in missing_cols:
        X_test_model[col] = 0
    X_test_model = X_test_model[X_train_model.columns]  # Ensure same column order

    # Train/test split
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_model, y_train, test_size=0.2, random_state=42
    )

    # XGBoost model
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        n_estimators=100,
        max_depth=20,
        learning_rate=0.1
    )

    print("Training model...")
    model.fit(X_train_split, y_train_split)

    # Validation
    val_preds = model.predict(X_val_split)
    val_accuracy = accuracy_score(y_val_split, val_preds)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    cm = confusion_matrix(y_val_split, val_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot()
    plt.title("Validation Confusion Matrix")
    plt.show()

    # Test predictions
    print("Predicting on test data...")
    click_probs = model.predict_proba(X_test_model)[:, 1]

    # Result dataframe
    result_df = pd.DataFrame({
        'srch_id': srch_ids_test,
        'prop_id': prop_ids_test,
        'click_proba': click_probs
    })

    # Rank properties within each search
    result_df['rank'] = result_df.groupby('srch_id')['click_proba'].rank(method='first', ascending=False)

    # Create submission
    submission = result_df.sort_values(by=['srch_id', 'rank'])[['srch_id', 'prop_id']]
    submission.to_csv("submission.csv", index=False)
    print("✅ Submission file saved as 'submission.csv'")

    # Plot feature importance
    xgb.plot_importance(model)
    plt.show()  # This should now work with the TkAgg backend