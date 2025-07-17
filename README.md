# Auction Crossing Price Prediction

## 1 - Loading and cleaning data

* Dataset contains exchange messages after 4 PM too, so filtered out those. It was imp to take into account daylight savings time that starts from 10th March thats whay Ive used timezone aware pandas datetime type. This also got rid of the extra messages for WBA (5.40-5.45 PM) and AAOI (4.30 PM - 4.35 PM) which led to a ~30% jump in the stock due to the post-market deal announcement with Amazon on 13th March.

* Convert timestamp column to pandas datetime. Important for fast vector operations for a large dataset like this


* <u>Clean:</u> Grouping by symbol and date, there are ***only*** 331 or 332 rows in the dataset (no duplicates or missing rows). This comes from:
        
        (i) 30 points from 3.50.0 PM - 3.54.50 PM every 10 seconds
        
        (ii) 300 points from 3.55.0 PM - 3.59.59 PM every 1 second
        
        (iii) 1 point at 4.0.0 PM (not always there - thats why some cases have 331 and others have 332 points)
        
        (iv) 1 special exchange message with ref_price=nan and giving the cross price for that date and symbol


* <u>Clean:</u> Count of rows with ref_prices=nan is 10500 exactly equal to the unique (symbol,date) pairs. So, only the special exchange messages at 4PM with the cross price are the ones with nan ref_prices. No extra unexpected nan ref_prices.

* <u>Not Clean:</u> Some ref_prices are =0. In most cases, they are 0 for the entire day for a symbol (when the row count is 330 or 331). 1 less than 331/332 because 1 point is the special exchange message with the cross price and ref_price=nan. There are 8 cases with some (not all) ref_prices=0 for a day. We're going to filter those days out as its a negligible proportion of the dataset, so not alot to lose and we get the advantage of clean non-zero ref_prices which will help the model to learn better and not skew decisions based on a few 0s.


* <u>Not clean (other columns):</u> Before above cleaning, there were many problematic columns. But after, there are 2 types of problematic columns left:
        
        (i) 4 bid/ask(qty) columns are the only ones with Nans. -> I've left them as nans because tree models are robust in handling nans and number of rows are also very less. cross is nan for all exchange messages except the special mesg at 4PM
        
        (ii) The columns with zeros. Most are fine. shares and paired_shares can be 0 naturally. open, we dont care about. far_price and near_price are only sent from 3.55 onwards. The 3.50 - 3.55 ones are the 0 ones along with some other random 0s. I have replaced them with nans, as tree models can handle it but 0 can throw the model off.


## 2 - Deciding Target variable (Preprocessing)
* I've modelled the problem as ref_price prediction WIDTH secs later with WIDTH=10,20,...300 secs. This will help us in making an informed decision at all times during the auction period.

* Here, it was important to handle the logic in 4 different time windows for a particular WIDTH=w. It's required because messages from 3.50 - 3.55 are 10 secs apart compared to 1 sec apart for 3.55 - 4.
        
        (i) Time window 1: 3.50 - (3.55-w) -> target_price is simply w/10 rows ahead
        
        (ii) Time window 2: (3.55-w) - 3.55 -> target_price is variable rows ahead (computed by compute_shifts())
        
        (iii) Time window 3: 3.55 - (4.00-w) -> target price is simply w rows ahead
        
        (iv) Time window 4: (4.00-w) - 4.00 -> target price is simply the crossing price at 4.00. There are 2 types of models. Type 1 doesnt use this 4th time window for training and Type 2 does (denoted by bool flag use_last_w_seconds).


* Variables introduced:

        (i) WIDTH - for each exchange message, model is trying to predict the ref_price WIDTH seconds later
        
        (ii) USE_LAST_W_SECONDS - flag denoting whther to use the last WIDTH seconds or not for training



## 3 - Feature Engineering
* I used a total of 8 different kind of features for a total of 251 features - 
    
    (1) Basic features - total amount, combining auction and limit order book quantities, buy vs sell pressure/difference, scaling the quantities per stock
    
    (2) Ratios - different auction quantities and money vs. limit order book quantities
    
    (3) Imbalance features - most imp features denoting the skew for different types of bid and ask quantities, pair and unpaired, price related. Also includes the urgency/pressure denoting features. Higher the skew between prices and quantities, higher are the urgency features.
    
    (4) Rolling mean and std - for different imbalance features with imbalance sign
    
    (5) Diff features - Difference between current and (3/6/12/18)*WINDOW_MULLTIPLIER secs before, per stock and date, for various price, quantity and imbalance features
    
    (6) Previous ref_prices - Rolling mean of 10/30/60 secs old ref price. 
    
    (7) Change of different imbalance features in terms of division and difference between 30/60/120/180 secs ago features
    
    (8) MACD features - for capturing the difference in limit order book vs auction prices
    
* Variable introduced:
    
    (1) WINDOW_MULTIPLIER - Rolling window size multiplier. Rolling window size used for different rolling mean/std are (3,6,12,18)*multiplier. Hence, if multiplier of 2 is used, that means for rolling features, a mean/std of last (6,12,24,36) entries or seconds are used as separate features.



## 4 - Modelling (What all I tried)

* Normalization of all features (0-1) and Oulier scrapping: Didnt provide any improvements. Tree models are scale invariant. Since the model works on threshold for each dependent variable that minimizes impurity, it doesnt matter if the variable is between 10 and 10,000 or 0.2 and 5, the threshold just adjusts accordingly. All normalization can do is speed up the training process but the disadvantage is that it adds an extra step for analysis so I chose to ignore it. Similarly for outliers, tree models dont pull the model in a certain direction, like in linear regression, due to the splitting nature of these models.

* PCA - Didnt provide any improvements with the added disadvantage of replacing normal features with meaningless transformations of the original features. So chose to not use it. Generally, tree models handle collinearity internally so the added advantage of PCA is moot most of the times. 

* XGBoost, CatBoost, LightGBM - Tried with normalization and no-normalization, L1 and L2 optimization with all 3 models. L1 optimization with no normalization with Lgbm gave the best results.

* CV - 5 fold CV were all leading to the same exact prediction so I ended up using the entire 80% split to train the model

* USE_LAST_W_SECONDS = True/False - Time window 4 discussed above in part 2. Unsurprisingly, False gave better results, in fact for larger WIDTH models, there were drastic differences between the False and the True cases in terms of Mean Absolute Error on test set. Makes sense because for the last WIDTH seconds, we dont have a reference price WIDTH seconds later as the model expects. Setting the target variable as the crossing price for this time window (for a lack of better target orice) is a source of noise.

* I also tried modelling it as a classification problem with target variable being +1(0) for price increase(decrease) by 4PM. Features used were the same along with an extra feature for distance to close (in seconds). This simplified the model massively since no parameter WIDTH was required. It also made sense since the final decision per exchange message is ultimately a classification problem. But the performance was underwhelming with a final Pnl of -2.2K on backtesting with WINDOW_MULTIPLIER=4. The model also early stopped within the first 5 iterations suggesting that a more nuanced way of predicting prices WIDTH seconds later and using an esnemble of models(different WIDTH models) is more promising. 1 possible future step I'd like to implement: Modelling the problem as a classifcation problem using the parameter WIDTH and hence, using an ensemble of models just like the regression modelling done in this submission.

* Training classification model file attached, classification.py -> after that just run backtest.py with global variable CLASSIFICATION set = True



## 5 - Backtest

<u>Note for reader:</u>
Above code trains only 1 model. We need a set of models with different WIDTHs for backtesting done below. To reproduce my results, clone my github repo containing the saved models used for the bellow analysis. Make sure to:

    (1) Set PATH_TO_READ_SAVE_MODELS to where the models are saved
    (2) Set PATH_FOR_DATASET to where the daily March csv files are there
    (2) Set CLASSIFICATION=False ofcourse
    (3) Run backtest.py with arguments like "python backtest.py 10 20" -> This runs the backtest for windows_multiplier=10 and frequency=20 (Parameters explained below)
    (4) Please make sure that all the frequency WIDTH models (till 300) are present in the Models folder i.e. if frequency=100, trained models with width=100,200,300 should be there.
    (5) If run successfully, each (multiplier, frequency) pair run will save 2 numpy arrays in the PATH_TO_SAVE_RESULTS folder - 
        (i) Pnl array - storing the pnl at every exchange message
        (ii) Difference array - storing the absolute difference between predicted price and the current price (useful for threshold strategy analysis done in Plot 4 in the next section)
    

* Variables used:
    
    (1) WINDOWS_MULTIPLIER - Same as before. Multiplier for deciding rolling window size used as features.

    (2) Frequency - This denotes the frequency of models used from 0 to 300 secs before the market close. For ex- if frequency is 50. Then models with WIDTH=50,100,150,200,250,300 will be used to make a prediction at any time between 3.50-4.00
    
* Backtest makes a prediction for the crossing price at every instant it gets an exchange message. Features for every exchange message can be calculated as done for training ofc. This helps us in making a prediction based on our trained models.

* At any time, the crossing price prediction is the weighted mean of the predictions of the closest 2 WIDTHS models based on the frequency. For ex- 
    
        params: if backtesting is done with freq=60, multiplier=8
        
        situation: At 3.57.30 -> halfway between 120secs and 180secs from the close
        
        models used: model1(width=120, multiplier=8) + model2(width=180, multiplier=8)
        
        prediction: since it is midway, prediction will be simply (prediction1 + prediction2)/2
    
        If time was 3.57.40 -> then predition = (2xprediction1 + prediction2)/3



* Predictions for greater than 300 secs away from close is made by the model 300 secs away simply. No weighted mean.

* After the prediction is made, strategy simply buys 1 share if prediction is that the price goes up. Else, sells. (This is ofc very simple just for the purpose of the assignment. Real-life strategy will require the use of confidence intervals)

* This strategy hence results in a pnl for every exchange message. Strat is backtested on the 20% unseen split = 693247 exchange message from 25th March 3.58PM - 31st March 4PM

* As specified above, it saves the 2 output numpy array of (1) pnl and (2) diff array - difference between prediction and current price.

* Code shared as part of sumbmission backtest.py. With the saved models, it takes about 7 mins to run without a GPU.



## 6 - Analysis

* There are 3 main variables that are constantly used in this analysis - multiplier, frequency & width. Imp to understand:-
    * Each multiplier leads to a unique feature set -> hence their own model
    * A (multiplier, width) pair defines a model -> it has an associated mean absolute error
    * A (multiplier, frequency) pair defines a unique strategy. 2 types of strategies possible:
        * ***10 Vanilla strats*** of using a single multiplier across the board for differeing distances from the close. Hence, this'll lead to 10 different optimal strategies. 1 best frequency per multiplier strategy.
        * ***1 Hybrid strat*** - Use a combination of multipliers depending on how far away from the close the mssg is. Essentially, each width will have an optimal multiplier. For ex- when we need a prediction from width=20, we use a (multiplier=6, width=20) trained model and another (multiplier=8,width=40) trained model. The idea is simple, depending on how far into the future is the price we're trying to predict, we might need a different rolling windows to capture temporal patterns of a time series effectively. Now, this optimal multiplier set, will have a different pnl for each frequency. Choose the one with the best frequency.
    * Threshold analysis - Each of the 10 optimal vanilla strats will lead to a different Pnl if we use an aggresiveness factor aka Threshold. Logic - simply enter positions for every exchange positions only if difference between prediction_price and current_price. Idea is that it might be better to not enter into a position for all exchange messages, only if prediction is significantly higher or lower.



* <u>Caution:</u> All the plots below are based on a very small test set. For real life strategies, will need a larger dataset or live testing with minimal sizes to come to a conclusion of which strategy to use use (if such execution were possible in real-life ofc). A reasonable conclusion for now would be that this strat might be biased to the test set used.

* Plots below are based on backtesting for (10 multipliers x 5 frequency)= 50 strategies based on 180 (10 multipliers x 18 widths) trained models for 1000 iterations.

* <u>Plot 1:</u> Mean Absolute Error for different rolling windows per width

    * Note- To run this, it requires a file mae.csv, that I have as a result of training (10 multipliers x 18 widths) = 180 models. Please save that in the PATH_TO_READ_SAVE_MODELS folder
    * <u>The purpose of this plot is to evaluate the performance of models trained on features with different rolling windows sizes</u>. For each rolling window, we can see the mean absolute error across different widths. This is important because for a vanilla strat rolling window (or multiplier) model to lead to a good pnl, it should perform well across all widths (depending on frequency chosen)
    * <u>How its calculated:</u> There are total (10 rolling windows x 18 different widths{20,40,...300 and 50,150,250}) = 180 different (rolling window, frequency) pairs. Each pair means that the features were calculated using the given rolling window size and an ensemble of these models (models with different widths and given rolling window multiplier) are used to predict the closing price just as described in 5, using weighted mean.
    * The general trend of increasing MAE makes sense -> as future time period aka Width increases, it becomes harder to predict.
    * Its not that clear, but 4,5,6,7,8 seem overall better performers. Note the 2nd plot of only the top ones (4,5,6,7,8)

![Plot: MAE per Width](plots/mae1.png)
![Plot: MAE per Width](plots/mae2.png)

* <u>Plot 2:</u> (Hybrid strategy) Pnl for different rolling windows per width

    * Note- Before running this. Need to run the file compute_best_rolling_window_per_width.py 10 times (once for each multiplier). Or save the dicts "dict_mul{multiplier}_pnl.pkl" I'll share in the PATH_TO_SAVE_RESULTS folder.
    * The purpose of this plot, similar to the previous plot, is to determine the rolling window size most suitable for a particular width model. In other words, <u>which rolling window size is the most useful in predicting the ref_price width seconds later.</u>
    * <u>How its calculated:</u> Taken care by the py script mentioned, for the 20% split exchange messages, I use multiplier=mul to predict prices for a width model (ref_prices width seconds later). This prediction is used in the same way except for the weighted mean. If predicted price higher, buy. The key difference between this way of calculating pnl and backtest.py is that the pnl locked in is due to the ref_price width seconds later instead of the cross price. This leads to a pnl for different rolling windows per width model.
    * The plot depicts the best pnl acheived per width model and which rolling multiplier was behind it.
    * Nice and large Pnls. This strat could be promising compared to the vanilla strats.

![Plot: Max Pnl per Width](plots/max_pnl_per_width.png)

* <u>Plot 3:</u> (Vanilla strategies) Pnl for different frequency strategies per rolling window size

    * Note- To run this, the corresponding (multiplier,freq) trained models should be saved in the PATH_TO_READ_SAVE_MODELS folder
    * As can be seen from the Plot 1, there is no clear multiplier(rolling window size) strategy that is a better performer so we <u>use this plot to determine the best performing frequency for each rolling window sized model used</u>
    * As can be seen from the plot, best Pnl is acheived with the (multiplier=6, frequency=100) pair. Pnl = 5223.6 (logs attached)
    * Overall, This way of making decisions for every exchange message looks like a granular approach. Using models for predicting every 20 secs aren't performing well as compared to frequency= 50 and 100.Freq=60 seems like the worst performer among them consistently across multipliers.

![Plot: Pnl for different frequencies](plots/pnl_for_freq.png)

* <u>Plot 4:</u> Pnl across thresholds for the best (mul,freq) pairs

    * This plot <u>helps us visualize an alternate way to use the same strategy adding another parameter Threshold</u> to a (multiplier, frequency) pair vanilla strategy.
    * Till now, we entered into a position for every exchange message. The idea of using a threshold to only enter into a position if the (predicted_price-current_price) > Threshold, naive way to model confidence in our model.
    * So we have 10 plots for each of the vanilla strategies i.e. 10 (multiplier, best_freq for that multiplier) pairs with different thresholds on the x-axis
    * Surprisingly, the best Pnl is acheived by taking a threshold of 0 consistently for all multipliers. Essentially, the threshold approach doesnt seem promising.

![Plot: Pnl across thresholds](plots/pnl_across_th1.png)
![Plot: Pnl across thresholds](plots/pnl_across_th10.png)


## 7 - Winning Strategy - Hybrid Strat
* Now for the winning strategy. As we saw the best pnl we get from vanilla multiplier strategies is 5.2K at best with the (multiplier=6, frequency=100) pair.
* But, the hybrid strategy showed alot of promise from plot 2 when testing different rolling windows per width. Best Pnls are pretty high with each width:

* Same file is used to test the backtest as before. Instructions to test-
    * Step 1: Run compute_best_rolling_window_per_width.py used for plot 2. This saves all the different Pnls for each (multiplier, width) pair.
    * Step 2: Run plotting.py to save the best_rolling_windows.pkl dictionary mapping width -> best multiplier. This file used in step 3.
    * Step 3: Run Backtest.py taking care of frequency (the 2nd argument) and setting HYBRID_STRAT=True in the beginning of the file.
        * Set HYBRID_STAT global variable to True
        * Make sure PATH_TO_READ_SAVE_MODELS has all the required models
        * Will need to run backtest.py as usual with 2 arguments. 1st argument for multiplier is meaningless as this is a hybrid strategy. 2nd argument is for the frequency.

* <u>Plot 5:</u> Pnl is considerably higher than the vanilla variants: Best Pnl ov=bserved with (hybrid optimal multiplier set, freq=100) pair = 13.2K (Log file shared: hybrid_100.log)

![Plot: Hybrid strat Pnl](plots/hybrid_strat_pnl.png)

## 8 - Possible Future Steps for Improvement

* All the models were trained on 1000 iterations. I did test briefly with 10,000 iterations for some variants. Mostly models were early stopping at <=5k but to maximize the capabilities of this strat/models, all the models should be retrained for larger number of iterations.
* I was only able to use freq=20,50,60,100,150. Due to limited resources, I was only able to train the models with width%20 or width%50=0 models for the 10 multipliers. Atleast 5-10 more should be investigated but the best model, strategy should still be in the ballpark of the above analysis.
* Some expensive stocks showed a very high MAE for their models compared to others. There should be better features to capture their variations, needs to be investigated.
* I wanted to test a more nuanced way of selecting features. Incrementally evaluate model performance improvement with chunks of feature groups added iteratively. Would be helpful to eliminate redundant and collinear features because code is somewhat computationally heavy currently. (Cant count the number of times my system crashed)
