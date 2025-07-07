#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 02:17:42 2025

@author: ashwinsamuel
"""

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle

PATH_TO_READ_SAVE_MODELS = '/Users/ashwinsamuel/Documents/Kaggle/closing_auction_prediction/models/n1000_False_with_mul'
PATH_TO_SAVE_RESULTS = '/Users/ashwinsamuel/Documents/Kaggle/closing_auction_prediction/results/n1000_False_with_mul'


def plot_pnl_across_thresholds(multiplier, difference_array, pnl_array, num_bins):

    ref_prices = np.load('ref_prices.npy')
    
    difference_array = difference_array[ref_prices!=0]
    pnl_array = pnl_array[ref_prices!=0]
    ref_prices = ref_prices[ref_prices!=0]

    diff_percentage = difference_array/ref_prices*100
    
    sorted_diffs = np.sort(diff_percentage)
    quantile_edges = np.linspace(0, 1, num_bins + 1)
    bin_edges = np.quantile(sorted_diffs, quantile_edges)
    
    threshold_pnl = {}
    for i in range(11):
        threshold = bin_edges[i]
        pnl = np.sum(pnl_array[diff_percentage>threshold])
        threshold_pnl[i]=pnl
    
    plt.plot(threshold_pnl.keys(), threshold_pnl.values())
    plt.xlabel("Threshold slice")
    plt.ylabel("Pnl")
    plt.title(f"Pnl Across different thresholds when multiplier={multiplier}")
    plt.grid(True)
    plt.show()
    

def plot_test_errors(df, top_ones):
    
    fig, ax = plt.subplots()

    for key, grp in df.groupby("rolling_window_size"):
        if top_ones and key in [1,2,3,9,10]:
            continue
        ax.plot(grp["width"], grp["MAE"], label=f"rolling window={key}")
    
    ax.set_xlabel("Width in seconds")
    ax.set_ylabel("Mean Absolute Error on unseen data")
    if top_ones:
        ax.set_title("Mean Absolute Error grouped by rolling window size (top ones)")
    else:
        ax.set_title("Mean Absolute Error grouped by rolling window size (all windows)")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__=='__main__':
    
    
    # PLOT 1 - Mean Absolute Error in predictions made on the 20% split test set
    
    os.chdir(PATH_TO_READ_SAVE_MODELS)
    df = pd.read_csv('mae.csv')
    df.columns = [ 'rolling_window_size', 'width', 'MAE']
    
    plot_test_errors(df, False)
    plot_test_errors(df, True)
        
    
    # PLOT 2 - Pnl for different rolling windows per width
    
    os.chdir(PATH_TO_SAVE_RESULTS)

    pnl_per_w, index = [],[]
    for multiplier in range(1,11):
        with open(f'dict_mul{multiplier}_pnl.pkl', 'rb') as f:
            my_dict = pickle.load(f)
        pnl_per_w.append(my_dict)
        index.append(multiplier)
    
    all_pnls = pd.DataFrame(pnl_per_w, index=index)
    
    sorted_columns = sorted(all_pnls.columns)
    all_pnls = all_pnls[sorted_columns]
    
    max_pnl_per_width = all_pnls.max()
    best_multipliers = all_pnls.idxmax()
    
    plt.figure(figsize=(10, 5))
    plt.plot(sorted_columns, max_pnl_per_width.values, marker='o')
    

    for i, width in enumerate(sorted_columns):
        pnl = max_pnl_per_width[width]
        mul = best_multipliers[width]
        plt.text(width, pnl, f'mul {mul}', ha='center', va='bottom')
    
    plt.title('Max Pnl per Width with corresponding rolliwng window multiplier')
    plt.xlabel('Width (in seconds)')
    plt.ylabel('Pnl')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
      
    with open('best_rolling_windows.pkl', 'wb') as f:
        pickle.dump(best_multipliers, f)


    # PLOT 3 - Pnl for different frequency strategies per rolling window size
    
    pnls,diffs={},{}
    freq_list = [20,50,60,100,150]
    best_pnl={}
    for mul in range(1,11):
        current_best_pnl = -float('inf')
        best_pnl[mul]=0
        for freq in freq_list:
            pnls[(mul,freq)] = np.load(f'pnls_mul{mul}_freq{freq}.npy')
            diffs[(mul,freq)] = np.load(f'diffs_mul{mul}_freq{freq}.npy').ravel()
            
            pnl = np.sum(pnls[(mul,freq)])
            if pnl>current_best_pnl:
                current_best_pnl=pnl
                best_pnl[mul]=freq
    
    
    fig, ax = plt.subplots()
    for freq in freq_list:
        pnl = defaultdict(list)
        for mul in range(1,11):
            pnl[mul].append(np.sum(pnls[(mul,freq)]))
        
        ax.plot(pnl.keys(), pnl.values(), label=f'frequency = {freq}')
        
    ax.set_xlabel('window multipliers')
    ax.set_ylabel('Pnl')
    ax.set_title('Pnl for different frequencies')
    ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
    # PLOT 4 - Which Th works for each (mul,freq) pair
    
    for multiplier in range(1,11):
        best_freq_for_multiplier = best_pnl[multiplier]
        
        difference_array = diffs[(multiplier, best_freq_for_multiplier)]
        pnl_array = pnls[(multiplier, best_freq_for_multiplier)]
        plot_pnl_across_thresholds(multiplier, difference_array, pnl_array, num_bins=10)

    
        
        
            
    
    
    
    
    
            
    
    
    
    
    # Enseble of best window per width -> need first plot 1 for accuracy. then normal pnl/accuracy of backtest
    
    

    # max possible PNL possible Max=90.5k    
    # os.chdir('/Users/ashwinsamuel/Documents/Kaggle/closing_auction_prediction/results/n1000_False_with_mul')
    # pnls = np.load('pnls_mul6_freq50.npy')
    # print(np.sum(pnls))
    # print(np.sum(np.abs(pnls)))
