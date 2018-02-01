#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 18:29:09 2018

"""

"""
This is an upgraded version of Ceshine's LGBM starter script
Added more statistical features and categorical features
"""
from datetime import date, timedelta
import calendar as ca
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
print('Loading Data')
df_train = pd.read_csv(
    '/Users/kehanliu/Downloads/shopsales/train/train.csv', usecols=[1, 2, 3, 4, 5],
    dtype={'onpromotion': bool},
    converters={'unit_sales': lambda u: np.log1p(
        float(u)) if float(u) > 0 else 0},
    parse_dates=["date"],
    skiprows=range(1, 66458909)  # 2016-01-01
)
#extract data relevant to test set
item_nbr_u = df_train.loc[df_train.date>pd.datetime(2017,8,10),'item_nbr']
#item_nbr_u = df_train[df_train.date>pd.datetime(2017,8,10)].item_nbr.unique()
df_test = pd.read_csv(
    "/Users/****/Downloads/shopsales/test/test.csv", usecols=[0, 1, 2, 3, 4],
    dtype={'onpromotion': bool},
    parse_dates=["date"]  # , date_parser=parser
).set_index(
    ['store_nbr', 'item_nbr', 'date']
)

items = pd.read_csv(
    "/Users/****/Downloads/shopsales/items/items.csv",
).set_index("item_nbr")
stores=pd.read_csv(
    "/Users/****/Downloads/shopsales/stores/stores.csv",
).set_index("store_nbr")

#only takes 2017's data for consideraton 
df_2017 = df_train.loc[df_train.date>=pd.datetime(2017,1,1)]
#keep info of items sold before
df_2017=df_2017[df_2017.item_nbr.isin(item_nbr_u)]
result = df_2017.join(items, on='item_nbr')
result = result.join(stores, on='store_nbr')

del df_train


oil= pd.read_csv(
    "/Users/*****/Downloads/shopsales/oil/oil.csv",parse_dates=["date"]
)

promo_2017_train = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(
        level=-1).fillna(False)
promo_2017_train.columns = promo_2017_train.columns.get_level_values(1)
promo_2017_test = df_test[["onpromotion"]].unstack(level=-1).fillna(False)
promo_2017_test.columns = promo_2017_test.columns.get_level_values(1)
promo_2017_test = promo_2017_test.reindex(promo_2017_train.index).fillna(False)
promo_2017 = pd.concat([promo_2017_train, promo_2017_test], axis=1)
del promo_2017_test, promo_2017_train

df_2017 = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(
        level=-1).fillna(0)

df_2017cat = result.set_index(
    ["store_nbr", "item_nbr", "date"])[["family","class","perishable","city","state","type","cluster" ]].unstack(
        level=-1).fillna(0)

df_2017.columns = df_2017.columns.get_level_values(1)

items = items.reindex(df_2017.index.get_level_values(1))

def get_timespan(df, dt, minus, periods, freq='D'):
    return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]

def get_nearwd(date,b_date):
    date_list = pd.date_range(date-timedelta(140),periods=21,freq='7D').date
    result = date_list[date_list<=b_date][-1]
    return result
def prepare_dataset(t2017, is_train=True):
    X = pd.DataFrame({
        #past sales
        "day_1_2017": get_timespan(df_2017, t2017, 1, 1).values.ravel(),
        "mean_3_2017": get_timespan(df_2017, t2017, 3, 3).mean(axis=1).values,
        "mean_7_2017": get_timespan(df_2017, t2017, 7, 7).mean(axis=1).values,
        "mean_14_2017": get_timespan(df_2017, t2017, 14, 14).mean(axis=1).values,
        "mean_30_2017": get_timespan(df_2017, t2017, 30, 30).mean(axis=1).values,
        "mean_60_2017": get_timespan(df_2017, t2017, 60, 60).mean(axis=1).values,
        "mean_140_2017": get_timespan(df_2017, t2017, 140, 140).mean(axis=1).values,
        "promo_14_2017": get_timespan(promo_2017, t2017, 14, 14).sum(axis=1).values,
        "promo_60_2017": get_timespan(promo_2017, t2017, 60, 60).sum(axis=1).values,
        "promo_140_2017": get_timespan(promo_2017, t2017, 140, 140).sum(axis=1).values,
                                                                           
                                      
        "median_3_2017": get_timespan(df_2017, t2017, 3, 3).median(axis=1).values,
        "median_7_2017": get_timespan(df_2017, t2017, 7, 7).median(axis=1).values,
        "median_14_2017": get_timespan(df_2017, t2017, 14, 14).median(axis=1).values,
        "median_30_2017": get_timespan(df_2017, t2017, 30, 30).median(axis=1).values,
        "median_60_2017": get_timespan(df_2017, t2017, 60, 60).median(axis=1).values,
        "median_140_2017": get_timespan(df_2017, t2017, 140, 140).median(axis=1).values,
        "promomedian_14_2017": get_timespan(promo_2017, t2017, 14, 14).median(axis=1).values,
        "promomedian_60_2017": get_timespan(promo_2017, t2017, 60, 60).median(axis=1).values,
        "promomedian_140_2017": get_timespan(promo_2017, t2017, 140, 140).median(axis=1).values,
                                      
        "max_3_2017": get_timespan(df_2017, t2017, 3, 3).max(axis=1).values,
        "max_7_2017": get_timespan(df_2017, t2017, 7, 7).max(axis=1).values,
        "max_14_2017": get_timespan(df_2017, t2017, 14, 14).max(axis=1).values,
        "max_30_2017": get_timespan(df_2017, t2017, 30, 30).max(axis=1).values,
        "max_60_2017": get_timespan(df_2017, t2017, 60, 60).max(axis=1).values,
        "max_140_2017": get_timespan(df_2017, t2017, 140, 140).max(axis=1).values,
        "promomax_14_2017": get_timespan(promo_2017, t2017, 14, 14).max(axis=1).values,
        "promomax_60_2017": get_timespan(promo_2017, t2017, 60, 60).max(axis=1).values,
        "promomax_140_2017": get_timespan(promo_2017, t2017, 140, 140).max(axis=1).values,
                                      
        "min_3_2017": get_timespan(df_2017, t2017, 3, 3).min(axis=1).values,
        "min_7_2017": get_timespan(df_2017, t2017, 7, 7).min(axis=1).values,
        "min_14_2017": get_timespan(df_2017, t2017, 14, 14).min(axis=1).values,
        "min_30_2017": get_timespan(df_2017, t2017, 30, 30).min(axis=1).values,
        "min_60_2017": get_timespan(df_2017, t2017, 60, 60).min(axis=1).values,
        "min_140_2017": get_timespan(df_2017, t2017, 140, 140).min(axis=1).values,
        "promomin_14_2017": get_timespan(promo_2017, t2017, 14, 14).min(axis=1).values,
        "promomin_60_2017": get_timespan(promo_2017, t2017, 60, 60).min(axis=1).values,
        "promomin_140_2017": get_timespan(promo_2017, t2017, 140, 140).min(axis=1).values,
                                      
        "std_3_2017": get_timespan(df_2017, t2017, 3, 3).std(axis=1).values,
        "std_7_2017": get_timespan(df_2017, t2017, 7, 7).std(axis=1).values,
        "std_14_2017": get_timespan(df_2017, t2017, 14, 14).std(axis=1).values,
        "std_30_2017": get_timespan(df_2017, t2017, 30, 30).std(axis=1).values,
        "std_60_2017": get_timespan(df_2017, t2017, 60, 60).std(axis=1).values,
        "std_140_2017": get_timespan(df_2017, t2017, 140, 140).std(axis=1).values,
        "std_14_2017": get_timespan(promo_2017, t2017, 14, 14).std(axis=1).values,
        "std_60_2017": get_timespan(promo_2017, t2017, 60, 60).std(axis=1).values,
        "std_140_2017": get_timespan(promo_2017, t2017, 140, 140).std(axis=1).values,
                                      
        "unpromo_16aftsum_2017":(1-get_timespan(promo_2017, t2017+timedelta(16), 16, 16)).iloc[:,1:].sum(axis=1).values}) 
                                

    for i in range(16):# next 16 days
        X["promo_{}".format(i)] = promo_2017[
            t2017 + timedelta(days=i)].values.astype(np.uint8)#no prom after the later days
        for j in [14,60,140]:#
            X["aft_promo_{}{}".format(i,j)] = (promo_2017[
                t2017 + timedelta(days=i)]-1).values.astype(np.uint8)
            X["aft_promo_{}{}".format(i,j)] = X["aft_promo_{}{}".format(i,j)]\
                                        *X['promo_{}_2017'.format(j)]#prom after/prom before 
        if i ==15:
            X["bf_unpromo_{}".format(i)]=0
        else:
            X["bf_unpromo_{}".format(i)] = (1-get_timespan(
                    promo_2017, t2017+timedelta(16), 16-i, 16-i)).iloc[:,1:].sum(
                            axis=1).values / (15-i) * X['promo_{}'.format(i)] #before unprom/how many days left

    for i in range(7):
        #dow is day of week
        X['mean_4_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 28-i, 4, freq='7D').mean(axis=1).values
        #X['mean_12_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 84-i, 12, freq='7D').mean(axis=1).values
        X['mean_8_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 56-i, 4, freq='7D').mean(axis=1).values
        X['mean_12_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 84-i, 4, freq='7D').mean(axis=1).values
        X['mean_16_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 112-i, 4, freq='7D').mean(axis=1).values  
        X['mean_20_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 140-i, 20, freq='7D').mean(axis=1).values        
        
          
        X['median_4_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 28-i, 4, freq='7D').median(axis=1).values
        #X['mean_12_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 84-i, 12, freq='7D').mean(axis=1).values
        X['median_8_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 56-i, 4, freq='7D').median(axis=1).values
        X['median_12_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 84-i, 4, freq='7D').median(axis=1).values
        X['median_16_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 112-i, 4, freq='7D').median(axis=1).values  
        X['median_20_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 140-i, 20, freq='7D').median(axis=1).values        
        
        X['max_4_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 28-i, 4, freq='7D').max(axis=1).values
        #X['mean_12_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 84-i, 12, freq='7D').mean(axis=1).values
        X['max_8_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 56-i, 4, freq='7D').max(axis=1).values
        X['max_12_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 84-i, 4, freq='7D').max(axis=1).values
        X['max_16_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 112-i, 4, freq='7D').max(axis=1).values  
        X['max_20_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 140-i, 20, freq='7D').max(axis=1).values        
        
        X['min_4_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 28-i, 4, freq='7D').min(axis=1).values
        #X['mean_12_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 84-i, 12, freq='7D').mean(axis=1).values
        X['min_8_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 56-i, 4, freq='7D').min(axis=1).values
        X['min_12_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 84-i, 4, freq='7D').min(axis=1).values
        X['min_16_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 112-i, 4, freq='7D').min(axis=1).values  
        X['min_20_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 140-i, 20, freq='7D').min(axis=1).values        
        
        X['std_4_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 28-i, 4, freq='7D').std(axis=1).values
        #X['mean_12_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 84-i, 12, freq='7D').mean(axis=1).values
        X['std_8_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 56-i, 4, freq='7D').std(axis=1).values
        X['std_12_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 84-i, 4, freq='7D').std(axis=1).values
        X['std_16_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 112-i, 4, freq='7D').std(axis=1).values  
        X['std_20_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 140-i, 20, freq='7D').std(axis=1).values        
         
        date = get_nearwd(t2017+timedelta(i),t2017)#which day in last week
        ahead = (t2017-date).days# exceed how many days
        if ahead!=0:
            X['ahead0_{}'.format(i)] = get_timespan(df_2017, date+timedelta(ahead), ahead, ahead).mean(axis=1).values
            X['ahead7_{}'.format(i)] = get_timespan(df_2017, date+timedelta(ahead), ahead+7, ahead+7).mean(axis=1).values
        X["day_1_2017_{}1".format(i)]= get_timespan(df_2017, date, 1, 1).values.ravel()
        X["day_1_2017_{}2".format(i)]= get_timespan(df_2017, date-timedelta(7), 1, 1).values.ravel()
        for m in [3,7,14,30,60,140]:
            X["mean_{}_2017_{}1".format(m,i)]= get_timespan(df_2017, date,m, m).\
                mean(axis=1).values
            X["mean_{}_2017_{}2".format(m,i)]= get_timespan(df_2017, date-timedelta(7),m, m).\
                mean(axis=1).values
    if is_train:
        y = df_2017[
            pd.date_range(t2017, periods=16)
        ].values
        return X, y
    return X
    

print("Preparing dataset...")

t2017 = date(2017, 6, 14)
X_l, y_l = [], []
for i in range(6):
    delta = timedelta(days=7 * i)
    X_tmp, y_tmp = prepare_dataset(
        t2017 + delta
    )
    X_l.append(X_tmp)
    y_l.append(y_tmp)
X_train = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)
del X_l, y_l
X_val2,y_val2 = prepare_dataset(date(2017, 7, 12))
X_val, y_val = prepare_dataset(date(2017, 7, 26))
X_test = prepare_dataset(date(2017, 8, 16), is_train=False)
X_train.fillna(0)

print("Training and predicting models...")
params = {
  
    'num_leaves': 31,
    'objective': 'regression',
    'min_data_in_leaf': 300,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'metric': 'l2_root',
    'num_threads': 4
}

MAX_ROUNDS = 500
val_pred = []
test_pred = []
cate_vars = []

for i in range(16):
    print("=" * 50)
    print("Step %d" % (i+1))
    print("=" * 50)
    dtrain = lgb.Dataset(
        X_train, label=y_train[:, i],
        categorical_feature=cate_vars,
        weight=pd.concat([items["perishable"]] * 6) * 0.25 + 1
    )
    dval2 = lgb.Dataset(
        X_val2, label=y_val2[:, i], reference=dtrain,
        weight=items["perishable"] * 0.25 + 1,
        categorical_feature=cate_vars)
        
    dval = lgb.Dataset(
        X_val, label=y_val[:, i], reference=dtrain,
        weight=items["perishable"] * 0.25 + 1,
        categorical_feature=cate_vars)
    bst = lgb.train(
        params, dtrain, num_boost_round=MAX_ROUNDS,
        valid_sets=[dtrain, dval, dval2], early_stopping_rounds=50, verbose_eval=100
    )
    print("\n".join(("%s: %.2f" % x) for x in sorted(
        zip(X_train.columns, bst.feature_importance("gain")),
        key=lambda x: x[1], reverse=True
    )))
    val_pred.append(bst.predict(
        X_val, num_iteration=bst.best_iteration or MAX_ROUNDS))
    test_pred.append(bst.predict(
        X_test, num_iteration=bst.best_iteration or MAX_ROUNDS))

print("Validation mse:", mean_squared_error(
    y_val, np.array(val_pred).transpose())**0.5)

print("Making submission...")
y_test = np.array(test_pred).transpose()
df_preds = pd.DataFrame(
    y_test, index=df_2017.index,
    columns=pd.date_range("2017-08-16", periods=16)
).stack().to_frame("unit_sales")
df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)

#output important features
importance = bst.feature_importance()
names = bst.feature_name()
with open('...../feature_importance.txt', 'w+') as file:
for index, im in enumerate(importance):
string = names[index] + ', ' + str(im) + 'n'
file.write(string)


#retrain the model with top 300 features


submission = df_test[["id"]].join(df_preds, how="left").fillna(0).reset_index()
submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 10000)
submission.loc[~submission.item_nbr.isin(item_nbr_u),'unit_sales']=0
del item_nbr_u
submission[['id','unit_sales']].to_csv('lgb.csv', float_format='%.4f', index=None)