import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
import numpy as np

def find_location_id(location_name, heppas):
    # factorize once
    _, location_uniques = pd.factorize(heppas['location'])
    # mapping from location name to factorized id
    loc_to_id = {loc: i for i, loc in enumerate(location_uniques)}

    if isinstance(location_name, pd.Series):
        # map series of names to ids (will return NaN for missing)
        location_id = location_name.map(loc_to_id)

        # optional: warn for missing values
        missing = location_id.isna()
        if missing.any():
            missing_locs = location_name[missing].unique()
            print(f"Locations not found in data: {missing_locs.tolist()}")
    else:
        # single value lookup
        location_id = loc_to_id.get(location_name)
        if location_id is None:
            print(f"Location '{location_name}' not found in data")

    return location_id

def find_horse_id(horse_name, heppas):
    # factorize once
    _, horse_uniques = pd.factorize(heppas['horse'])
    # mapping from location name to factorized id
    horse_to_id = {horse: i for i, horse in enumerate(horse_uniques)}

    if isinstance(horse_name, pd.Series):
        # map series of names to ids (will return NaN for missing)
        horse_id = horse_name.map(horse_to_id)

        # optional: warn for missing values
        missing = horse_id.isna()
        if missing.any():
            missing_locs = horse_name[missing].unique()
            print(f"Horses not found in data: {missing_locs.tolist()}")
    else:
        # single value lookup
        horse_id = horse_to_id.get(horse_name)
        if horse_id is None:
            print(f"Horse '{horse_name}' not found in data")

    return horse_id

def get_data():
    heppas = pd.read_csv('heppa_all_results.csv')

    #Fix junk placing values to float
    heppas['placing']=heppas['placing'].str.extract(r'(\d+(?:\.\d+)?)').astype(float).fillna(0).astype(int)

    #Only heats with number < 10, to skip ponies and shit
    heppas = heppas[heppas['heat_num'] < 10]


    heppas['m_heat_info']=[i if (type(i) == str and "ryhmä" in i) else False for i in heppas.heat_info]
    heppas = heppas[heppas.m_heat_info != False]

    #Lämminveriset 0
    #Kylmäveriset 1
    heppas['blooded']=[0 if (type(i) == str and "Lämminveriset" in i) else 1 for i in heppas.heat_info]

    heppas['horse_id']=pd.factorize(heppas['horse'])[0]
    heppas['track_condition'] = heppas['track_condition'].astype('category')
    heppas['location_id'] = pd.factorize(heppas['location'])[0]
    heppas['date_id'] = pd.factorize(heppas['date'])[0]
    
    return heppas

def do_prediction(heppas):
    features = ['temperature', 'horse_id', 'track_condition', 'location_id', 'blooded']
    location_name = 'Kaustinen, Nikula'
    heat_num = 4
    blooded = 0 #0 = lämminveriset, 1=kylmäveriset

    a=[{'number': 1, 'horse': 'Zesty Leader', 'odds': 3.21},
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

    test_heat = pd.DataFrame(a)

    test_heat['temperature'] = 25
    test_heat['heat_num'] = heat_num
    test_heat['track_condition'] = 'Hyvä kesärata'
    test_heat['location'] = location_name
    test_heat['location_id'] = find_location_id(test_heat['location'], heppas)
    test_heat['blooded'] = blooded

    test_heat['horse_id'] = find_horse_id(test_heat['horse'], heppas)
    horse_counts = heppas['horse'].value_counts()
    test_heat['occurences'] = test_heat['horse'].map(horse_counts).fillna(0).astype(int)
    test_heat['track_condition'] = test_heat['track_condition'].astype('category')

    test_heat['pred_score'] = model.predict(test_heat[features])
    test_heat = test_heat.sort_values(by='pred_score', ascending=False).reset_index(drop=True)
    test_heat['predicted_placing'] = test_heat.index + 1
    print("-------------------------------------------------")
    print(test_heat[['horse', 'pred_score', 'predicted_placing', 'odds', 'occurences']])





model = lgb.Booster(model_file='model.txt')
heppas = get_data()
do_prediction(heppas)