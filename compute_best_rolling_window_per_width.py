                        #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 12:22:36 2025

@author: ashwinsamuel
"""

from datetime import datetime, timedelta, time
import gc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import pytz
import sys

import lightgbm as lgb

PATH_FOR_DATASET = '/Users/ashwinsamuel/Documents/Kaggle/closing_auction_prediction/202503_imbalance/'
PATH_TO_READ_SAVE_FEATURES = '/Users/ashwinsamuel/Documents/Kaggle/closing_auction_prediction/saved_features/'
PATH_TO_READ_SAVE_MODELS = '/Users/ashwinsamuel/Documents/Kaggle/closing_auction_prediction/models/n1000_False_with_mul'
PATH_TO_SAVE_RESULTS = '/Users/ashwinsamuel/Documents/Kaggle/closing_auction_prediction/results/n1000_False_with_mul'
CLASSIFICATION = False
WINDOWS_MULTIPLIER = 10
USE_LAST_W_SECONDS = False

def load_data(path):

    start_date = datetime(2025, 3, 3)
    end_date = datetime(2025, 3, 31)
    all_data = []

    current_date = start_date
    while current_date <= end_date:
        print(f'Getting data for {current_date.date()}')
        if current_date.weekday() < 5:
            filename = current_date.strftime("%Y%m%d") + ".csv"
            df = pd.read_csv(path+filename)
            all_data.append(df)

        current_date += timedelta(days=1) 

    # convert to pandas datetime for fast vector ops
    compiled_df = pd.concat(all_data, ignore_index=True)
    compiled_df['local_time'] = pd.to_datetime(compiled_df['local_time'], utc=True).dt.tz_convert("America/New_York")
    compiled_df['date'] = compiled_df['local_time'].dt.day

    # filter out datapoint after 4PM
    cutoff_time = time(16,0,1)
    compiled_df = compiled_df[ compiled_df['local_time'].dt.time < cutoff_time ]

    return compiled_df


def clean_data(df):

    zero_ref_prices = df[df['ref_price']==0]
    zeros_per_symbol_date = zero_ref_prices.groupby(['symbol', 'date'], as_index=False).size()
    df_clean = df.merge(zeros_per_symbol_date[['symbol', 'date']], on=['symbol', 'date'], how='left', indicator=True)
    df_clean = df_clean[df_clean['_merge']=='left_only'].drop(columns='_merge')
    zero_far_prices = df_clean['far_price']==0
    zero_near_prices = df_clean['near_price']==0
    df_clean.loc[zero_far_prices, 'far_price'] = np.nan
    df_clean.loc[zero_near_prices, 'near_price'] = np.nan

    return df_clean


def compute_shifts(width):
    
    set_of_shifts = set()
    
    time = 55*60 - width + 10
    while time<55*60:
        minute = time//60
        seconds = time%60
    
        secs_to_55 = (55 - minute - 1)*60 + 60 - seconds
        secs_after_55 = width - secs_to_55
        to_shift = int(secs_after_55 + secs_to_55/10)
        set_of_shifts.add(to_shift)
    
        time+=10

    return set_of_shifts


def get_target(df, width, use_last_w_seconds):

    df['stock_id'] = pd.factorize(df['symbol'])[0]
    df['seconds'] = df['local_time'].dt.second
    df['minute'] = df['local_time'].dt.minute

    df.sort_values(by=['symbol', 'local_time'], inplace=True)
    
    df[f'ref_price_shift-{width}'] = df['ref_price'].shift(-width) #all
    df[f'ref_price_shift-{int(width/10)}'] = df['ref_price'].shift(-int(width/10)) #cutoff2
    df['next_ref_price'] = df[f'ref_price_shift-{width}']
    
    # other shifts required for this width
    for shift in compute_shifts(width):
        df[f'ref_price_shift-{shift}'] = df['ref_price'].shift(-shift)

    cutoff_time_1 = time(15,55,0)
    condition_1 = df['local_time'].dt.time < cutoff_time_1

    time_in_seconds = 55*60 - width
    while time_in_seconds<55*60:
        m = time_in_seconds//60
        s = time_in_seconds%60
        secs_to_55 = (55 - m - 1)*60 + 60 - s
        secs_after_55 = width - secs_to_55
        to_shift = int(secs_after_55 + secs_to_55/10)

        condition_2 = (df['minute']==m) & (df['seconds']==s)
        df.loc[(condition_1) & (condition_2), 'next_ref_price'] = df.loc[(condition_1) & (condition_2), f'ref_price_shift-{to_shift}']
        print(f'For width={width}: variable shift for {m}m{s}sec done')
        time_in_seconds+=10

    cutoff_time_2 = (datetime(2025,3,3,15,55,0) - timedelta(seconds=width)).time()
    condition_2 = df['local_time'].dt.time < cutoff_time_2
    df.loc[condition_2, 'next_ref_price'] = df.loc[condition_2, f'ref_price_shift-{int(width/10)}']

    cutoff_datetime_3 = datetime(2025,3,3,16,0,0) - timedelta(seconds=width)
    final_cross = df.groupby(['symbol', 'date'], as_index=False)['cross'].last().rename(columns={'cross': 'final_cross'})
    df = df.merge(final_cross, on=['symbol', 'date'], how='left')

    if use_last_w_seconds:    
        condition_3 = df['local_time'].dt.time > cutoff_datetime_3.time()
        df.loc[condition_3, 'next_ref_price'] = df.loc[condition_3, 'final_cross']
        df.dropna(subset='ref_price', inplace=True)
    else:

        m,s = cutoff_datetime_3.minute, cutoff_datetime_3.second
        last_valid_rows = (df['minute']==m) & (df['seconds']==s)
        df.loc[last_valid_rows, 'next_ref_price'] = df.loc[last_valid_rows, 'final_cross']

        cutoff_datetime_3 = cutoff_datetime_3 + timedelta(seconds=1)
        condition_3 = df['local_time'].dt.time > cutoff_datetime_3.time()
        df = df[~condition_3]

    df.sort_values(by='local_time', inplace=True)
    df=df.copy()

    return df['next_ref_price']


def generate_features(df, window_multiplier):
    
    features = ['local_time', 'date', 'symbol','seconds','unpaired_shares','imbalance_side',
               'ref_price','paired_shares','far_price','near_price','bid','bid_qty',
                'ask','ask_qty', 'auc_bid_qty','auc_ask_qty', 'cross', 'secs_to_close']
    

    df['stock_id'] = pd.factorize(df['symbol'])[0]
    df['seconds'] = df['local_time'].dt.second
    df['minute'] = df['local_time'].dt.minute
    df.rename(columns={'shares': 'unpaired_shares'}, inplace=True)
    df.sort_values(by='local_time', inplace=True)


    # 1 - Basic features
    size_col = ['unpaired_shares','paired_shares','bid_qty','ask_qty']
    for _ in size_col:
        df[_] = df[_] / df['adv']
        df[f"scale_{_}"] = df[_] / df.groupby(['stock_id'])[_].transform('median')
        features.append(f"scale_{_}")

    df['secs_to_close'] = (60 - df['minute'] - 1)*60 + (60-df['seconds'])
    df['imbalance_side'] = np.where( df['side']=='BUY', 1, -1 )
    df['auc_bid_qty'] = df['paired_shares']
    df['auc_ask_qty'] = df['paired_shares']
    df.loc[df['imbalance_side']==1,'auc_bid_qty'] += df.loc[df['imbalance_side']==1,'unpaired_shares']
    df.loc[df['imbalance_side']==-1,'auc_ask_qty'] += df.loc[df['imbalance_side']==-1,'unpaired_shares']

    df["ask_money"] = df["ask_qty"] * df["ask"]
    df["bid_money"] = df["bid_qty"] * df["bid"]
    df["ask_qty_all"] = df["ask_qty"] + df["auc_ask_qty"]
    df["bid_qty_all"] = df["bid_qty"] + df["auc_bid_qty"]
    df["volumn_size_all"] = df["ask_qty_all"] + df["bid_qty_all"]
    df["ask_auc_money"] = df["ref_price"] * df["auc_ask_qty"]
    df["bid_auc_money"] = df["ref_price"] * df["auc_bid_qty"]
    df["volume_money"] = df["ask_money"] + df["bid_money"]
    df["volume_cont"] = df["ask_qty"] + df["bid_qty"]
    df["diff_ask_bid_qty"] = df["ask_qty"] - df["bid_qty"]
    df["volumn_auc"] = df["unpaired_shares"] + 2 * df["paired_shares"]
    df["volumn_auc_money"] = df["volumn_auc"] * df["ref_price"]
    df["mid_price"] = (df["ask"] + df["bid"]) / 2
    df["mid_price_near_far"] = (df["near_price"] + df["far_price"]) / 2
    df["price_diff_ask_bid"] = df["ask"] - df["bid"]
    df["price_div_ask_bid"] = df["ask"] / df["bid"]
    df["flag_scale_unpaired_shares"] = df["imbalance_side"] * df["scale_unpaired_shares"]
    df["flag_unpaired_shares"] = df["imbalance_side"] * df["unpaired_shares"]
    df["div_flag_unpaired_shares_2_balance"] = (df["unpaired_shares"] / df["paired_shares"]) * df["imbalance_side"]
    df["price_pressure"] = df["price_diff_ask_bid"] * df["unpaired_shares"]
    df["price_pressure_v2"] = df["price_pressure"] * df["imbalance_side"]
    df["depth_pressure"] = (df["ask_qty"] - df["bid_qty"]) / (df["far_price"] - df["near_price"])
    df["div_bid_qty_ask_qty"] = df["bid_qty"] / df["ask_qty"]

    features.extend(['ask_money', 'bid_money', 'ask_auc_money','bid_auc_money',"ask_qty_all","bid_qty_all","volumn_size_all",
                      'volume_money','volume_cont',"volumn_auc","volumn_auc_money","mid_price",
                      'mid_price_near_far','price_diff_ask_bid',"price_div_ask_bid","flag_unpaired_shares","div_flag_unpaired_shares_2_balance",
                     "price_pressure","price_pressure_v2","depth_pressure","flag_scale_unpaired_shares","diff_ask_bid_qty"])

    print("1 - Basic features done")


    # 2 - Various ratios
    # Improve microscopically
    ratio_pairs = [
        ("unpaired_shares", "bid_qty"),
        ("unpaired_shares", "ask_qty"),
        ("paired_shares", "bid_qty"),
        ("paired_shares", "ask_qty"),
        ("unpaired_shares", "volume_cont"),
        ("paired_shares", "volume_cont"),
        ("auc_bid_qty", "bid_qty"),
        ("auc_ask_qty", "ask_qty"),
        ("bid_auc_money", "bid_money"),
        ("ask_auc_money", "ask_money"),
    ]
    for col1, col2 in ratio_pairs:
        df[f"div_{col1}_2_{col2}"] = df[col1] / df[col2]
        features.append(f"div_{col1}_2_{col2}")

    print("2 - Ratio features done")


    # 3 - Imbalanced Features
    # non-price related
    imbalance_pairs = [
        ('ask_qty', 'bid_qty'),
        ('ask_money', 'bid_money'),
        ('volume_money', 'volumn_auc_money'),
        ('volume_cont', 'volumn_auc'),
        ('unpaired_shares', 'paired_shares'),
        ('auc_ask_qty', 'auc_bid_qty'),
        ("ask_qty_all", 'bid_qty_all')
    ]
    for col1, col2 in imbalance_pairs:
        df[f"imb1_{col1}_{col2}"] = (df[col1] - df[col2]) / (df[col1] + df[col2])
        features.append(f"imb1_{col1}_{col2}")

    # price related
    price_cols = ["ref_price", "far_price", "near_price", "ask", "bid", "mid_price"]
    for i,c1 in enumerate(price_cols):
        for j,c2 in enumerate(price_cols):
            if i<j:
                df[f"imb1_{c1}_{c2}"] = (df[c1] - df[c2]) / (df[c1] + df[c2])
                features.append(f"imb1_{c1}_{c2}")

    # cumulative imbalance features
    df["market_urgency_v2"] = (df["imb1_ask_qty_bid_qty"] + 2) * (df["imb1_ask_bid"] + 2) * (df["imb1_auc_ask_qty_auc_bid_qty"] + 2)
    df["market_urgency"] = df["price_diff_ask_bid"] * df["imb1_ask_qty_bid_qty"]
    df["market_urgency_v3"] = df["imb1_ask_bid"] * df["imb1_ask_qty_bid_qty"]
    features.extend(["market_urgency_v2", "market_urgency", "market_urgency_v3"])

    print("3 - Imbalance features done")


    # fancy features
    all_new_features = []
    
    
    # 4 - rolling mean and std features
    rolling_cols = [
        "bid_auc_money", "bid_qty_all",
        "imb1_auc_ask_qty_auc_bid_qty", "div_flag_unpaired_shares_2_balance",
        "imb1_ask_qty_all_bid_qty_all", "flag_unpaired_shares", "imb1_ref_price_mid_price" 
    ]
    for col in rolling_cols:
        for window in [3*window_multiplier, 6*window_multiplier, 12*window_multiplier, 18*window_multiplier]:
            s1 = df.groupby(['stock_id', 'date'])[col].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            s1.name = f"rolling{window}_mean_{col}"

            s2 = df.groupby(['stock_id', 'date'])[col].transform(lambda x: x.rolling(window=window, min_periods=1).std())
            s2.name = f"rolling{window}_std_{col}"

            all_new_features.extend([s1,s2])
            features.extend([s1.name, s2.name])

    print("4 - Rolling mean and std features done")


    # select few features 2


    # Miscellaneous
    s1 = df.groupby(['stock_id', 'date'])["flag_unpaired_shares"].diff()
    s1.name = "imbalance_momentum_unscaled"
    s2 = df.groupby(['stock_id', 'date'])["price_diff_ask_bid"].diff()
    s2.name = "spread_intensity"
    df["imbalance_momentum"] = s1/df["paired_shares"]
    all_new_features.extend([s1,s2])
    features.extend(["imbalance_momentum_unscaled", "spread_intensity", "imbalance_momentum"])


    # 5 - Diff features
    diff_cols = [
        "ask", "bid", "imb1_ref_price_near_price", "bid_qty",
        "scale_bid_qty", "mid_price", "ask_qty", "price_div_ask_bid",
        "div_bid_qty_ask_qty", "market_urgency", "imbalance_momentum" 
    ]
    for col in diff_cols:
        for window in [3*window_multiplier, 6*window_multiplier, 12*window_multiplier, 18*window_multiplier]:
            s1 = df.groupby(['stock_id', 'date'])[col].diff(periods=window)
            s1.name = f"{col}_diff_{window}"
            all_new_features.append(s1)
            features.append(f"{col}_diff_{window}")
    
    print("5 - Diff features done")


    # 6 - Prev ref prices rolling window
    df['ref_price_shift10'] = df['ref_price'].shift(10)
    df['ref_price_shift30'] = df['ref_price'].shift(30)
    df['ref_price_shift60'] = df['ref_price'].shift(60)
    prev_prices = [ "ref_price_shift10", "ref_price_shift30", "ref_price_shift60"]
    for col in prev_prices:
        for window in [3*window_multiplier, 6*window_multiplier, 12*window_multiplier, 18*window_multiplier]:
            s1 = df.groupby(['stock_id', 'date'])[col].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            s1.name = f"rolling{window}_mean_{col}"
            all_new_features.append(s1)
            features.append(f"rolling{window}_mean_{col}")
    print("6 - Prev ref_prices features done")
    
    # 7 - change(division and difference) compared to prev imbalances
    imbalances = [ "imb1_auc_ask_qty_auc_bid_qty","imbalance_side","price_pressure_v2","scale_paired_shares"]
    for col in imbalances:
        for window in [3*window_multiplier, 6*window_multiplier, 12*window_multiplier, 18*window_multiplier]:
            shifted = df.groupby(['stock_id', 'date'])[col].shift(window)
            s1 = df[col] / shifted
            s1.name = f'div_shift{window}_{col}'
            s2 = df[col] - shifted
            s2.name = f'diff_shift{window}_{col}'
            all_new_features.extend([s1,s2])
            features.extend([f'div_shift{window}_{col}', f'diff_shift{window}_{col}'])
    print("7 - Changes compared to prev imbalance features done")

    # 8 - MACD
    macd_features = []
    rsi_cols = ["mid_price_near_far", "near_price"]
    for col in rsi_cols:
        for window_size in [3*window_multiplier, 6*window_multiplier, 12*window_multiplier, 18*window_multiplier]:
            s1 = df.groupby(['stock_id', 'date'])[col].transform(lambda x: x.ewm(span=window_size, adjust=False).mean())
            s1.name = f"rolling_ewm_{window_size}_{col}"
            macd_features.append(s1)        
            features.append(f"rolling_ewm_{window_size}_{col}")            
    df = pd.concat([df] + macd_features, axis=1)
    
    macd_features = []
    for col in rsi_cols:
        for w1, w2 in zip((3*window_multiplier, 6*window_multiplier, 12*window_multiplier), (6*window_multiplier, 12*window_multiplier, 18*window_multiplier)):
            s1 = df[f"rolling_ewm_{w1}_{col}"] - df[f"rolling_ewm_{w2}_{col}"]
            s1.name = f"dif_{col}_{w1}_{w2}"
            macd_features.append(s1)
            features.append(f"dif_{col}_{w1}_{w2}")
    df = pd.concat([df] + macd_features, axis=1)

    macd_features = []    
    for col in rsi_cols:
        for w1, w2 in zip((3*window_multiplier, 6*window_multiplier, 12*window_multiplier), (6*window_multiplier, 12*window_multiplier, 18*window_multiplier)):
            s1 = df.groupby(['stock_id', 'date'])[f"dif_{col}_{w1}_{w2}"].transform(lambda x: x.ewm(span=9, adjust=False).mean())
            s1.name = f"dea_{col}_{w1}_{w2}"
            macd_features.append(s1)
            features.append(f"dea_{col}_{w1}_{w2}")
    df = pd.concat([df] + macd_features, axis=1)
    
    for col in rsi_cols:
        for w1, w2 in zip((3*window_multiplier, 6*window_multiplier, 12*window_multiplier), (6*window_multiplier, 12*window_multiplier, 18*window_multiplier)):
            s1 = df[f"dif_{col}_{w1}_{w2}"] - df[f"dea_{col}_{w1}_{w2}"]
            s1.name = f"macd_{col}_{w1}_{w2}"
            all_new_features.append(s1)
            features.append(f"macd_{col}_{w1}_{w2}")
    print("8 - MACD features done")

    # 9 - rolling window of target 'x' days back 
    # for days in [1,2,3]:
    #     for window_size in [3*window_multiplier,6*window_multiplier,12*window_multiplier,18*window_multiplier]:
    #         shifted = df.groupby(['stock_id', 'seconds'])['next_ref_price'].shift(days)
    #         df[f'rolling_mean_{window_size}_{days}_days'] = shifted.groupby(df['stock_id']).transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
    #         df[f'rolling_std_{window_size}_{days}_days'] = shifted.groupby(df['stock_id']).transform(lambda x: x.rolling(window=window_size, min_periods=1).std())
    #         features.extend([f'rolling_mean_{window_size}_{days}_days', f'rolling_std_{window_size}_{days}_days'])
    
    
    print(f"We got total {len(features)} Features")
    df = pd.concat([df] + all_new_features, axis=1)
    df = df[features]
    
    return df


if __name__=="__main__":
    
    
    compiled_df = load_data(PATH_FOR_DATASET)

    split = round(0.8*len(compiled_df))
    compiled_df = compiled_df.iloc[split:]
    compiled_df2 = compiled_df.dropna(subset='ref_price').copy()

    gc.collect()

    # os.chdir(PATH_TO_READ_SAVE_FEATURES')
    backtest_features = generate_features(compiled_df2, window_multiplier=WINDOWS_MULTIPLIER)
    # features.index = pd.RangeIndex(len(features))
    # features.to_feather(f"features_251_mul{WINDOWS_MULTIPLIER}.feather")
    # features = pd.read_feather(f"features_251_mul{WINDOWS_MULTIPLIER}.feather")
    
    columns_to_remove = ['symbol', 'local_time', 'date', 'cross', 'secs_to_close']   
    backtest_features = backtest_features.dropna(subset='ref_price')

    os.chdir(PATH_TO_READ_SAVE_MODELS)
    pnl_per_w = {}
    for width in [50,100,150,250] + list(range(20,301,20)):
        model = lgb.Booster(model_file=f"model_w{width}_False_mul{WINDOWS_MULTIPLIER}.txt")
        target = get_target(compiled_df[['date', 'symbol', 'local_time', 'ref_price', 'cross']].copy(), width=width, use_last_w_seconds=USE_LAST_W_SECONDS)
    
        if not USE_LAST_W_SECONDS:
            cutoff_datetime = datetime(2025,3,3,16,0,0) - timedelta(seconds=width-1)
            condition = backtest_features['local_time'].dt.time > cutoff_datetime.time()
            features2 = backtest_features[~condition]
        else:
            features2 = backtest_features
    
        X = features2.drop(columns=columns_to_remove).values
        Y = target.values


        prediction = model.predict(X)
        buy_sell = np.where(prediction>features2['ref_price'], 1, -1)
        pnls = (Y-features2['ref_price'])*buy_sell
        pnl_per_w[width] = pnls.sum()

        
    os.chdir(PATH_TO_SAVE_RESULTS)    
    with open(f'dict_mul{WINDOWS_MULTIPLIER}_pnl.pkl', 'wb') as f:
        pickle.dump(pnl_per_w, f)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    