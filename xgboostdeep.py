
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from xgboost import XGBRanker
from sklearn.metrics import ndcg_score

# 1. Load data
train_file = "training_set_VU_DM.csv"
test_file = "test_set_VU_DM.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)



def handle_missing_values(df):
    low_missing_cols = ['prop_review_score', 'prop_location_score1']
    for col in low_missing_cols:
        if col in df.columns:
            df[col + '_missing'] = df[col].isna().astype(int)
            df[col] = df[col].fillna(df[col].median())

    moderate_missing_cols = ['prop_location_score2', 'orig_destination_distance']
    for col in moderate_missing_cols:
        if col in df.columns:
            df[col + '_missing'] = df[col].isna().astype(int)
            df[col] = df[col].fillna(df[col].median())

    high_missing_cols = ['srch_query_affinity_score', 'visitor_hist_adr_usd', 'visitor_hist_starrating']
    for col in high_missing_cols:
        if col in df.columns:
            df[col + '_exists'] = df[col].notna().astype(int)
            if df[col].dtype.kind in 'biufc':
                df[col] = df[col].fillna(df[col].median())

    competitor_cols = [col for col in df.columns if col.startswith('comp')]
    for col in competitor_cols:
        if col in df.columns:
            if '_rate' in col or '_inv' in col:
                df[col] = df[col].notna().astype(int)
            elif '_percent_diff' in col:
                df[col] = df[col].fillna(0)

    return df

def normalize_features(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if col in ["srch_id","prop_id"]:
            continue
        if df[col].nunique() > 2:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                df[col] = (df[col] - df[col].median()) / iqr
    return df

def add_custom_features(df):
    if 'price_usd' in df.columns:
        df['price_log'] = np.log1p(df['price_usd'])
        if 'visitor_hist_adr_usd' in df.columns:
            df['price_diff_to_hist'] = df['price_usd'] - df['visitor_hist_adr_usd'].fillna(df['price_usd'])
            df['price_ratio_to_hist'] = df['price_usd'] / (df['visitor_hist_adr_usd'].fillna(df['price_usd']) + 1)

    if 'prop_starrating' in df.columns:
        df['starrating_squared'] = df['prop_starrating'] ** 2
        if 'visitor_hist_starrating' in df.columns:
            df['star_diff_to_hist'] = df['prop_starrating'] - df['visitor_hist_starrating'].fillna(df['prop_starrating'])

    if 'prop_location_score1' in df.columns and 'prop_location_score2' in df.columns:
        df['location_score_avg'] = (df['prop_location_score1'] + df['prop_location_score2']) / 2
        df['location_score_diff'] = df['prop_location_score1'] - df['prop_location_score2']

    if 'date_time' in df.columns:
        df['date_time'] = pd.to_datetime(df['date_time'])
        df['day_of_week'] = df['date_time'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    return df

# 2. Feature Engineering (example features)
def feature_engineering(df):
    df['price_diff'] = df['price_usd'] - df['visitor_hist_adr_usd']
    df['star_diff'] = df['prop_starrating'] - df['visitor_hist_starrating']
    df['score1d2'] = df['prop_location_score1'] * df['prop_location_score2']
    df.fillna(-1, inplace=True)
    return df

train_df=handle_missing_values(train_df)

train_df = feature_engineering(train_df)
train_df = add_custom_features(train_df)
train_df = normalize_features(train_df)
test_df=handle_missing_values(test_df)

test_df = feature_engineering(test_df)
test_df = add_custom_features(test_df)
test_df = normalize_features(test_df)
# 3. Define features and label
#features = [
    #'price_usd', 'prop_starrating', 'prop_review_score', 'srch_length_of_stay',
    #'srch_booking_window', 'srch_adults_count', 'srch_children_count',
    #'srch_room_count', 'prop_location_score1', 'prop_location_score2',
    #'price_diff', 'star_diff', 'score1d2'
#]
drop_cols = ['srch_id', 'prop_id', 'date_time','position', 'click_bool', 'gross_bookings_usd', 'booking_bool']
features = [col for col in train_df.columns if col not in drop_cols]


X = train_df[features]
y = train_df['booking_bool'] * 5 + train_df['click_bool']  # Weighted target
group = train_df.groupby('srch_id').size().to_frame('size')['size'].to_numpy()

# 4. Cross-validation setup
gkf = GroupKFold(n_splits=5)
srch_ids = train_df['srch_id'].to_numpy()

ndcg_scores = []

for train_idx, valid_idx in gkf.split(X, y, groups=srch_ids):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    group_train = train_df.iloc[train_idx].groupby('srch_id').size().to_numpy()
    group_valid = train_df.iloc[valid_idx].groupby('srch_id').size().to_numpy()

    model = XGBRanker(
        objective='rank:pairwise',
        learning_rate=0.1,
        n_estimators=100,
        max_depth=6,
        subsample=0.75,
        colsample_bytree=0.75,
        random_state=42,
        tree_method='hist',
        verbosity=1
    )

    model.fit(X_train, y_train, group=group_train)

    # NDCG calculation
    valid_preds = model.predict(X_valid)
    valid_df = train_df.iloc[valid_idx].copy()
    valid_df['pred'] = valid_preds

    ndcg_per_query = []
    for qid, group_df in valid_df.groupby('srch_id'):
        if group_df['booking_bool'].sum() + group_df['click_bool'].sum() == 0:
            continue
        true_relevance = (group_df['booking_bool'] * 5 + group_df['click_bool']).to_numpy().reshape(1, -1)
        scores = group_df['pred'].to_numpy().reshape(1, -1)
        ndcg_per_query.append(ndcg_score(true_relevance, scores))

    fold_ndcg = np.mean(ndcg_per_query)
    print(f"Fold NDCG@k: {fold_ndcg:.4f}")
    ndcg_scores.append(fold_ndcg)

print(f"Mean NDCG@k over folds: {np.mean(ndcg_scores):.4f}")

# 5. Train final model on all training data
final_group = train_df.groupby('srch_id').size().to_numpy()
model.fit(X, y, group=final_group)

# 6. Predict on test set
X_test = test_df[features]
test_df['prediction'] = model.predict(X_test)

importances = model.feature_importances_
feature_importance_dict = dict(zip(features, importances))
sorted_importances = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

print("Feature Importances:")
for feature, importance in sorted_importances:
    print(f"{feature}: {importance:.4f}")

# 7. Create submission
submission = test_df[['srch_id', 'prop_id', 'prediction']]
submission.sort_values(['srch_id', 'prediction'], ascending=[True, False], inplace=True)
submission.drop('prediction', axis=1).to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")

