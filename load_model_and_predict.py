import lightgbm as lgb
from pathlib import Path

from heppa_data import (
    FEATURES,
    build_value_map,
    load_raw_data,
    predict_heat,
    prepare_test_heat,
    preprocess_heppa_data,
)

def do_prediction(heppas):
    features = FEATURES
    location_name = 'Kaustinen, Nikula'
    heat_num = 4
    blooded = 0  # 0 = lämminveriset, 1 = kylmäveriset

    horses = [
        {'number': 1, 'horse': 'Zesty Leader', 'odds': 3.21},
        {'number': 3, 'horse': 'Markin Andre', 'odds': 20.12},
        {'number': 4, 'horse': 'Tesswin', 'odds': 17.75},
        {'number': 5, 'horse': 'Lookatshinydiamond ', 'odds': 50.30},
        {'number': 6, 'horse': 'Northwind Mega', 'odds': 5.58},
        {'number': 7, 'horse': 'Kapa o Pango', 'odds': 43.12},
        {'number': 8, 'horse': 'Olympe De Veluwe', 'odds': 18.86},
        {'number': 9, 'horse': 'Full Patch', 'odds': 9.43},
        {'number': 10, 'horse': 'Zaborsky', 'odds': 18.86},
        {'number': 11, 'horse': 'Duplantis Broline*', 'odds': 50.30},
        {'number': 12, 'horse': 'Wish Will Win', 'odds': 3.77},
    ]

    location_to_id = build_value_map(heppas['location'])
    horse_to_id = build_value_map(heppas['horse'])
    horse_counts = heppas['horse'].value_counts()
    track_condition_categories = heppas['track_condition'].cat.categories

    test_heat = prepare_test_heat(
        horses=horses,
        location_name=location_name,
        heat_num=heat_num,
        blooded=blooded,
        temperature=25,
        track_condition='Hyvä kesärata',
        location_to_id=location_to_id,
        horse_to_id=horse_to_id,
        horse_counts=horse_counts,
        track_condition_categories=track_condition_categories,
    )
    test_heat = test_heat.rename(columns={'occurrences': 'occurences'})
    test_heat = predict_heat(model, test_heat, features=features)
    print("-------------------------------------------------")
    print(test_heat[['horse', 'pred_score', 'predicted_placing', 'odds', 'occurences']])


model_files = Path("./").glob("model*.txt")
latest_model_file = max([f for f in model_files], key=lambda item: item.stat().st_ctime)

model = lgb.Booster(model_file=str(latest_model_file))
heppas = preprocess_heppa_data(load_raw_data())
do_prediction(heppas)