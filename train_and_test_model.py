import lightgbm as lgb
import pandas as pd
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt

from heppa_data import (
    GROUP_COLUMNS,
    FEATURES,
    add_relevance,
    build_lgb_dataset,
    build_value_map,
    load_raw_data,
    predict_heat,
    preprocess_heppa_data,
    prepare_test_heat,
    split_heat_groups,
)

def create_lgb_datasets(train_df, val_df, features, group_col):
    """Create LightGBM datasets for training and validation."""
    train_data = build_lgb_dataset(train_df, categorical_features=["track_condition"])
    val_data = build_lgb_dataset(val_df, categorical_features=["track_condition"])
    return train_data, val_data

def train_model(train_data, val_data, params, num_boost_round=10000):
    """Train the LightGBM model."""
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        num_boost_round=num_boost_round
    )
    return model

def evaluate_model(model, val_df, features, group_col):
    """Evaluate the model using NDCG on validation data."""
    val_df['pred'] = model.predict(val_df[features])
    ndcg_scores = []
    
    for heat_id, group in val_df.groupby(group_col):
        true_relevance = group['relevance'].values.reshape(1, -1)
        predicted_scores = group['pred'].values.reshape(1, -1)
        if len(group) > 1:
            score = ndcg_score(true_relevance, predicted_scores)
            ndcg_scores.append(score)
    
    mean_NDCG = sum(ndcg_scores) / len(ndcg_scores)
    print(f"Mean NDCG on validation heats: {mean_NDCG:.4f}")
    return mean_NDCG

def predict_on_single_heat(model, single_heat, features):
    """Predict scores and rankings for a single heat."""
    return predict_heat(model, single_heat, features=features)


def predict_on_test_heat(model, test_heat, features):
    """Predict scores and rankings for a test heat."""
    return predict_heat(model, test_heat, features=features)


def build_test_heat(entries, temperature, heat_num, location_name, blooded, heppas):
    """Build a DataFrame for a test heat prediction input using shared helpers."""
    location_to_id = build_value_map(heppas['location'])
    horse_to_id = build_value_map(heppas['horse'])
    horse_counts = heppas['horse'].value_counts()
    track_condition_categories = heppas['track_condition'].cat.categories

    test_heat = prepare_test_heat(
        horses=entries,
        location_name=location_name,
        heat_num=heat_num,
        blooded=blooded,
        temperature=temperature,
        track_condition='Hyvä kesärata',
        location_to_id=location_to_id,
        horse_to_id=horse_to_id,
        horse_counts=horse_counts,
        track_condition_categories=track_condition_categories,
    )
    test_heat = test_heat.rename(columns={'occurrences': 'occurences'})
    return test_heat


def test_forssa_heat_prediction(model, heppas, features):
    """Run the prediction test for the Forssa heat scenario."""
    test_heat = build_test_heat(
        [
            {'number': 6, 'horse': 'Herbie Malibu', 'odds': 24},
            {'number': 3, 'horse': 'Cyclone', 'odds': 53},
            {'number': 7, 'horse': 'Ramonez Nimbus', 'odds': 62},
            {'number': 12, 'horse': "Erna's Prince", 'odds': 403},
            {'number': 8, 'horse': 'Casino Vegas', 'odds': 320},
            {'number': 11, 'horse': 'Stamina', 'odds': 142},
            {'number': 2, 'horse': 'Bonu Bonaparte', 'odds': 362},
            {'number': 1, 'horse': 'Vivienne V.', 'odds': 614},
            {'number': 5, 'horse': 'Enjoy The Moment', 'odds': 784},
            {'number': 4, 'horse': 'MAS Imperial', 'odds': 88},
        ],
        temperature=19,
        heat_num=5,
        location_name='Forssa, Pilvenmäki',
        blooded=0,
        heppas=heppas,
    )

    test_heat = predict_on_test_heat(model, test_heat, features)
    print("Test heat prediction:")
    print(test_heat[['horse', 'pred_score', 'predicted_placing', 'odds', 'occurences']])


def test_kaustinen_heat_prediction(model, heppas, features):
    """Run the prediction test for the Kaustinen heat scenario."""
    test_heat = build_test_heat(
        [
            {'number': 1, 'horse': 'Kokardi', 'odds': 23},
            {'number': 2, 'horse': 'Pystin Virkku', 'odds': 160},
            {'number': 3, 'horse': 'Makkonen', 'odds': 78},
            {'number': 4, 'horse': 'Lavjuu', 'odds': 363},
            {'number': 6, 'horse': 'Villi-Riikka', 'odds': 329},
            {'number': 7, 'horse': 'Söören', 'odds': 79},
            {'number': 8, 'horse': 'Tuisku-Taika', 'odds': 30},
        ],
        temperature=21,
        heat_num=2,
        location_name='Kaustinen, Nikula',
        blooded=1,
        heppas=heppas,
    )

    test_heat = predict_on_test_heat(model, test_heat, features)
    print("Test heat prediction:")
    print(test_heat[['horse', 'pred_score', 'predicted_placing', 'odds', 'occurences']])

def main():
    # Load and preprocess data
    heppas = preprocess_heppa_data(load_raw_data())
    
    # Create relevance scores
    heppas = add_relevance(heppas)
    
    # Define features and target
    features = FEATURES
    group_col = GROUP_COLUMNS
    
    # Remove non-placers
    heppas = heppas[heppas['placing'].fillna(0) > 0]
    
    # Split data
    train_df, val_df = split_heat_groups(heppas)
    
    # Create datasets
    train_data, val_data = create_lgb_datasets(train_df, val_df, features, group_col)
    
    # Define model parameters
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'verbosity': 1,
        'learning_rate': 0.005,
        'boosting': 'dart',
        'num_leaves': 30,
        'min_data_in_leaf': 100,
    }
    
    # Train model
    model = train_model(train_data, val_data, params)
    
    # Evaluate model
    mean_NDCG = evaluate_model(model, val_df, features, group_col)
    
    # Save model
    model.save_model(f'model_NDCG_{mean_NDCG:.4f}.txt')
    
    # Plot feature importance
    lgb.plot_importance(model, importance_type="gain", figsize=(7, 6), title="LightGBM Feature Importance (Gain)")
    plt.show()
    
    # Predict on a single heat
    unique_heats = heppas[['location_id', 'heat_num', 'date_id']].drop_duplicates()
    single_heat = heppas.merge(unique_heats.iloc[11:12], on=group_col).copy()
    single_heat = predict_on_single_heat(model, single_heat, features)
    print("Single heat prediction:")
    print(single_heat[['horse', 'pred_score', 'predicted_placing', 'placing']])
    
    # Predict on dedicated test heat scenarios
    test_forssa_heat_prediction(model, heppas, features)
    test_kaustinen_heat_prediction(model, heppas, features)

if __name__ == "__main__":
    main()