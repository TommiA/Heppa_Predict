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

'''
Index(['location', 'date', 'start_time', 'heat_num', 'placing', 'number',
       'horse', 'driver', 'time', 'odds', 'distance', 'note', 'prize',
       'temperature', 'track_condition', 'heat_info', 'm_heat_info',
       'horse_id', 'track_condition_id', 'location_id'],
      dtype='object')
'''

heppas = heppas[['heat_num', 'placing', 'number', 'temperature', 'horse_id', 'horse', 'track_condition', 'location_id', 'location', 'date_id', 'odds', 'blooded']]


# Select relevant columns
features = ['temperature', 'horse_id', 'track_condition', 'location_id', 'blooded']
#features = ['number', 'temperature', 'horse_id', 'track_condition', 'location_id', 'heat_num']
target = 'placing'  # Lower = better
group_col = ['location_id', 'heat_num', 'date_id']

# Remove non-placers if needed
heppas = heppas[heppas['placing'].fillna(0) > 0]

# Flip placing to relevance score (lower placing = higher relevance)
heppas['relevance'] = heppas.groupby(group_col)['placing'].transform(lambda x: max(x) - x + 1)

# Sort data by group to prepare for LightGBM ranking
heppas = heppas.sort_values(by=group_col)

# Train/val split based on unique heats
#unique_heats = heppas[group_col].unique()
unique_heats = heppas[group_col].drop_duplicates()
train_heats, val_heats = train_test_split(unique_heats, test_size=0.33, random_state=42)

#train_df = heppas[heppas[group_col].isin(train_heats)]
#val_df = heppas[heppas[group_col].isin(val_heats)]
train_df = heppas.merge(train_heats, on=group_col)
val_df = heppas.merge(val_heats, on=group_col)

# Prepare group sizes (number of horses per heat)
train_group = train_df.groupby(group_col).size().tolist()
val_group = val_df.groupby(group_col).size().tolist()

# LightGBM Datasets
train_data = lgb.Dataset(train_df[features], label=train_df['relevance'], group=train_group)
val_data = lgb.Dataset(val_df[features], label=val_df['relevance'], group=val_group)

# LightGBM ranking parameters
params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'verbosity': 1,
    'learning_rate': 0.005,
    'boosting': 'dart', #'rf' 'dart'
    #'device': 'gpu',
    'num_leaves': 30,
    'min_data_in_leaf': 100,
}

# Train model
model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, val_data],
    num_boost_round=10000,
    #callbacks=[
    #    lgb.early_stopping(stopping_rounds=10000, min_delta=0.01),
    #]
)

# Predict on validation set
val_df['pred'] = model.predict(val_df[features])

# Evaluate ranking per heat using NDCG
ndcg_scores = []
for heat_id, group in val_df.groupby(group_col):
    true_relevance = group['relevance'].values.reshape(1, -1)
    predicted_scores = group['pred'].values.reshape(1, -1)
    if len(group) > 1:  # Need at least 2 to compute NDCG
        score = ndcg_score(true_relevance, predicted_scores)
        ndcg_scores.append(score)

mean_NDCG = sum(ndcg_scores)/len(ndcg_scores)

print(f"Mean NDCG on validation heats: {sum(ndcg_scores)/len(ndcg_scores):.4f}")

print("Saving model..")
model.save_model(f'model_NDCG_{mean_NDCG}.txt')
#bst = lgb.Booster(model_file='model.txt')

lgb.plot_importance(model, importance_type="gain", figsize=(7,6), title="LightGBM Feature Importance (Gain)")
plt.show()

single_heat = heppas.merge(unique_heats.iloc[11:12], on=group_col).copy()

# 🔸 Predict scores
single_heat['pred_score'] = model.predict(single_heat[features])

# 🔹 Sort by predicted score (descending → higher rank)
single_heat = single_heat.sort_values(by='pred_score', ascending=False).reset_index(drop=True)

# 🔸 Add predicted placing (rank 1 = most likely winner)
single_heat['predicted_placing'] = single_heat.index + 1

# ✅ Result
#print(single_heat[['horse_id', 'pred_score', 'predicted_placing', 'placing']])
print(single_heat[['horse', 'pred_score', 'predicted_placing', 'placing']])

#print(single_heat)
print(ndcg_score(single_heat['relevance'].values.reshape(1, -1), single_heat['pred_score'].values.reshape(1, -1)))

#

location_name = 'Forssa, Pilvenmäki'
#Lähtö 2
heat_num = 5
blooded = 0

'''
Forssa Pilvenmäki, 10.7.

Lämpötila	19
Radan kunto	
Hyvä kesärata


6 Herbie Malibu
3 Cyclone
7 Ramonez Nimbus
12 Erna's Prince
8 Casino Vegas
11 Stamina
2 Bonu Bonaparte
1 Vivienne V.
5 Enjoy The Moment
4 MAS Imperial
'''
a=[{'number': 6, 'horse': 'Herbie Malibu', 'odds': 24},
   {'number': 3, 'horse': 'Cyclone',  'odds': 53},
   {'number': 7, 'horse': 'Ramonez Nimbus',  'odds': 62},
   {'number': 12, 'horse': 'Erna\'s Prince',  'odds': 403},
   {'number': 8, 'horse': 'Casino Vegas',  'odds': 320},
   {'number': 11, 'horse': 'Stamina',  'odds': 142},
   {'number': 2, 'horse': 'Bonu Bonaparte',  'odds': 362},
   {'number': 1, 'horse': 'Vivienne V.',  'odds': 614},
   {'number': 5, 'horse': 'Enjoy The Moment',  'odds': 784},
   {'number': 4, 'horse': 'MAS Imperial',  'odds': 88},
    ]

test_heat = pd.DataFrame(a)

test_heat['temperature'] = 19
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


#The betting thing
location_name = 'Kaustinen, Nikula'
#Lähtö 2
heat_num = 2
blooded = 1


a=[{'number': 1, 'horse': 'Kokardi', 'odds': 23},
   {'number': 2, 'horse': 'Pystin Virkku', 'odds': 160},
   {'number': 3, 'horse': 'Makkonen', 'odds': 78},
   {'number': 4, 'horse': 'Lavjuu', 'odds': 363},
   {'number': 6, 'horse': 'Villi-Riikka', 'odds': 329},
   {'number': 7, 'horse': 'Söören', 'odds': 79},
   {'number': 8, 'horse': 'Tuisku-Taika', 'odds': 30},
    ]

test_heat = pd.DataFrame(a)

test_heat['temperature'] = 21
test_heat['heat_num'] = heat_num
test_heat['track_condition'] = 'Hyvä kesärata'
test_heat['location'] = location_name
test_heat['blooded'] = blooded

test_heat['location_id'] = find_location_id(test_heat['location'], heppas)
test_heat['horse_id'] = find_horse_id(test_heat['horse'], heppas)
horse_counts = heppas['horse'].value_counts()
test_heat['occurences'] = test_heat['horse'].map(horse_counts).fillna(0).astype(int)
test_heat['track_condition'] = test_heat['track_condition'].astype('category')

test_heat['pred_score'] = model.predict(test_heat[features])
test_heat = test_heat.sort_values(by='pred_score', ascending=False).reset_index(drop=True)
test_heat['predicted_placing'] = test_heat.index + 1
print("-------------------------------------------------")
print(test_heat[['horse', 'pred_score', 'predicted_placing', 'odds', 'occurences']])



