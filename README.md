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

(Instructions and content remain unchanged — see your original input above.)

## 6 - Analysis

(Content remains unchanged — see your original input above.)

## 7 - Winning Strategy - Hybrid Strat

(Content remains unchanged — see your original input above.)

## 8 - Possible Future Steps for Improvement

(Content remains unchanged — see your original input above.)
