import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # For GUI plotting, if needed
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

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
        df['srch_query_affinity_score'] = (df['srch_query_affinity_score'] - df['srch_query_affinity_score'].mean()) / df['srch_query_affinity_score'].std()
    if 'position' in df.columns:
        df['position'] = (df['position'] - df['position'].mean()) / df['position'].std()
    return df

def add_custom_features(df):
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
    print("🚀 Starting CatBoost pipeline...")

    # File paths
    train_file = "training_set_VU_DM.csv"
    test_file = "test_set_VU_DM.csv"

    # Load training data (sample 10%)
    train_df = pd.read_csv(train_file)
    train_df = train_df.sample(frac=0.1, random_state=42)  # Use only 10%

    # Load test data
    test_df = pd.read_csv(test_file)

    # Preprocess both datasets
    X_train, y_train = preprocess_training_data(train_df, kind="train")
    X_test, _ = preprocess_training_data(test_df, kind="test")

    # Save IDs for submission
    srch_ids_test = X_test['srch_id'].copy()
    prop_ids_test = X_test['prop_id'].copy()

    # Drop non-feature columns
    drop_cols = [col for col in ['srch_id', 'prop_id', 'date_time'] if col in X_train.columns]
    X_train_model = X_train.drop(columns=drop_cols, errors='ignore')
    X_test_model = X_test.drop(columns=drop_cols, errors='ignore')

    # Align columns between train and test
    missing_cols = set(X_train_model.columns) - set(X_test_model.columns)
    for col in missing_cols:
        X_test_model[col] = 0
    X_test_model = X_test_model[X_train_model.columns]  # Ensure same order

    # Train/test split for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_model, y_train, test_size=0.2, random_state=42
    )

    # Initialize and train CatBoostRegressor
    model = CatBoostRegressor(
        iterations=200,
        learning_rate=0.1,
        depth=6,
        verbose=100,
        random_seed=42
    )

    print("🧠 Training CatBoost Regressor...")
    model.fit(X_train_split, y_train_split)

    # Validate
    val_preds = model.predict(X_val_split)
    plt.hist(val_preds, bins=50)
    plt.title("Validation Score Distribution")
    plt.xlabel("Predicted Score")
    plt.ylabel("Count")
    plt.show()

    # Predict on test set
    print("🔮 Predicting on test data...")
    click_scores = model.predict(X_test_model)

    # Create ranking result
    result_df = pd.DataFrame({
        'srch_id': srch_ids_test,
        'prop_id': prop_ids_test,
        'click_score': click_scores
    })

    result_df['rank'] = result_df.groupby('srch_id')['click_score'].rank(method='first', ascending=False)

    # Generate submission
    submission = result_df.sort_values(by=['srch_id', 'rank'])[['srch_id', 'prop_id']]
    submission.to_csv("submission_catboost.csv", index=False)
    print("✅ Submission saved to 'submission_catboost.csv'")
