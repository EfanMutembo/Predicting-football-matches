# Football Win Predictor
### Overview
This project uses Machine Learningn to predict a team winning a footbal match in the English Premier Leauge
Using the trained ML model I paired odds data to backtest strategies, trying to fish out one that produces positive ROI
### Data Sources
For the project I decided on the EPL(englis premier leauge) 2019 - 2022 seasons to get a quick idea on the project succes
I ended up scraping the data from the 2019 - 2022 season from Fbref

Match Data: https://fbref.com/en/comps/9/2022-2023/schedule/2022-2023-Premier-League-Scores-and-Fixtures

Odds Data: https://www.football-data.co.uk/data.php


### Project Plan

My project involves training the ML model and joining betting data to strategise a betting algorithm that produces a positive ROI
1. Data Preprocessing
2. Feature Enineering
3. Train and Predict
   
4. Backtest

## 1. Data Preprocessing
Most of ym data preprocessing was done in the ingestion stage after the webscrape. 
Data preview:
| index | date       | time   | comp             | round                    | day   | venue   | result   | gf     | ga     | opponent             |    xg |   xga |   poss |   attendance | captain                   | formation   | referee                 | match report   | notes                                                    |   sh |   sot |   dist |   fk |   pk |   pkatt |   season | team                     |
|-------|------------|--------|------------------|--------------------------|-------|---------|----------|--------|--------|---------------------|------:|------:|-------:|-------------:|---------------------------|-------------|------------------------|----------------|----------------------------------------------------------|------:|------:|--------:|-----:|-----:|--------:|---------:|--------------------------|
|     0 | 2021-08-07 | 17:15  | Community Shield | FA Community Shield      | Sat   | Neutral | L        | 0      | 0      | nan                  |   0   |   0   |      65 |        58262 | Fernandinho               | 4-3-3       | Anthony Taylor          | Match Report   | nan                                                      |   12 |     3 |   nan   |  nan |    0 |       0 |     2022 | Manchester City          |
|     1 | 2021-08-15 | 16:30  | Premier League   | Matchweek 1              | Sun   | Away    | L        | 0      | 1      | Tottenham            |   1.8 |   1   |      65 |        58262 | Fernandinho               | 4-3-3       | Anthony Taylor          | Match Report   | nan                                                      |   18 |     4 |   17.3 |    1 |    0 |       0 |     2022 | Manchester City          |
|     2 | 2021-08-21 | 15:00  | Premier League   | Matchweek 2              | Sat   | Home    | W        | 5      | 0      | Norwich City         |   2.6 |   0.1 |      67 |        51437 | İlkay Gündoğan            | 4-3-3       | Graham Scott            | Match Report   | nan                                                      |   16 |     4 |   18.5 |    1 |    0 |       0 |     2022 | Manchester City          |
|     3 | 2021-08-28 | 12:30  | Premier League   | Matchweek 3              | Sat   | Home    | W        | 5      | 0      | Arsenal              |   4.4 |   0.2 |      80 |        52276 | İlkay Gündoğan            | 4-3-3       | Martin Atkinson         | Match Report   | nan                                                      |   25 |    10 |   14.8 |    0 |    0 |       0 |     2022 | Manchester City          |
|     4 | 2021-09-11 | 15:00  | Premier League   | Matchweek 4              | Sat   | Away    | W        | 1      | 0      | Leicester City       |   2.8 |   0.6 |      61 |        32087 | İlkay Gündoğan            | 4-3-3       | Paul Tierney            | Match Report   | nan                                                      |   25 |     8 |   14.3 |    0 |    0 |       0 |     2022 | Manchester City          |
|     5 | 2021-09-15 | 20:00  | Champions Lg     | Group stage              | Wed   | Home    | W        | 6      | 3      | de RB Leipzig        |   2.3 |   1.3 |      50 |        38062 | Rúben Dias                | 4-3-3       | Serdar Gözübüyük        | Match Report   | nan                                                      |   15 |     7 |   16.8 |    0 |    1 |       1 |     2022 | Manchester City          |
|     6 | 2021-09-18 | 15:00  | Premier League   | Matchweek 5              | Sat   | Home    | D        | 0      | 0      | Southampton          |   1   |   0.4 |      64 |        52698 | Fernandinho               | 4-3-3       | Jonathan Moss           | Match Report   | nan                                                      |   16 |     1 |   16.4 |    1 |    0 |       0 |     2022 | Manchester City          |
|     7 | 2021-09-21 | 19:45  | EFL Cup          | Third round              | Tue   | Home    | W        | 6      | 1      | Wycombe              | nan   | nan   |      79 |        30959 | Kevin De Bruyne           | 4-3-3       | Robert Jones            | Match Report   | nan                                                      |   26 |    14 |  nan   |  nan |    0 |       0 |     2022 | Manchester City          |

The steps I took to clean the data were:
- Filter to only see Premier Leauge rows
- Encode Categorical columns
- Eliminate nulls
- Filter out draws

I had to filter out the draws as my classification model is only concered with predicting wins at this moment. 
This is a concious decision because my model doesnt have enough data or complex features to effectively learn and categorize Loss Win and Draw as of yet.
Dropping the null values was due to no important columns containng nulls.

## 2. Feature Engineering
To capture a footballs team form I calulated the rolling average for the most influential shooting and scoring statistics. This helped boost the models precision when predicting wins by 10%

|   gf_rolling |   ga_rolling |   sh_rolling |   sot_rolling |   dist_rolling |   fk_rolling |   pk_rolling |   pkatt_rolling |
|-------------:|-------------:|-------------:|--------------:|---------------:|-------------:|-------------:|----------------:|
|     1.33333  |     1.33333  |     10.6667  |      4.66667  |        17.2    |     0.333333 |     0        |        0        |
|     2        |     2        |     14.6667  |      5.66667  |        18.2    |     1        |     0.333333 |        0.333333 |
|     1.66667  |     1.66667  |     13.6667  |      3.33333  |        17.0667 |     1        |     0.333333 |        0.333333 |
|     1.33333  |     1        |     13.6667  |      3.33333  |        17      |     1        |     0.333333 |        0.333333 |
|     0.333333 |     1        |      9.66667 |      2        |        16.6333 |     0.333333 |     0        |        0     


## 3. Train and Predict
I trained 3 classification models to decide on which model performs best 
I used the Sci-Kit Learn library and Python to quickly train the models

| model                      |   Precision Wins |   Precision Loss |   Accuracy |
|:---------------------------|-----------------:|-----------------:|-----------:|
| GradientBoostingClassifier |         0.637389 |         0.624194 |   0.632257 |
| LogisticRegression         |         0.616503 |         0.568976 |   0.591161 |
| RandomForestClassifier     |         0.633202 |         0.578754 |   0.600473 |

- Gradient Boosting Classifier
  <ul style="list-style-type: none;">
  <li>  It is an ensemble learning method that builds multiple decision trees sequentially, with each tree correcting the errors of the previous one.It combines the predictions of weak learners (usually shallow decision trees) to create a strong predictive model.</li>
  <li> Gave us the highest Precision and Accuracy meaning when the model predicted a win it was correct 63.7% of the time</li>
</ul>

- Logistic Regression
  <ul style="list-style-type: none;">
  <li> Despite its name, logistic regression is a linear model for binary classification.It estimates probabilities using the logistic function, which maps any real-valued input into the range [0, 1].It's simple, interpretable, and efficient for large datasets but assumes a linear relationship between features and the log-odds of the target variable.</li>
  <li> Came in Last due to the poorest accuracy</li>
</ul>

- Random Forest
   <ul style="list-style-type: none;">
  <li> It is an ensemble learning method that builds a collection of decision trees and combines their predictions.</li>
  <li> Close second to GBC but marginally does worse when predicting Loss i.e mis labels lossing games</li>
</ul>

If I were to improve model performence further I would perform a extensive Grid Search and teak the model parameters to improve model performence. Sicne we are after a quick answer I have skipped this stage.

Calculating the feature importance helps us converge on important columns. As seen below the model is curretly making its decisions from 4 features (sh_rolling ,opponent_code,dist_rolling,ga_rolling). To ensure a better prediction I must feature engineer to help the model learn over a more broad set of features.

Feature          |Feature Importance|
|:--------------:|-----------------------
sh_rolling:     |0.23413043361703154|
opponent_code:  |0.22871934706564198|
dist_rolling:   |0.1567954916313238
ga_rolling:     |0.09179432614925116
venue_code:     | 0.06937108238348512
gf_rolling:     | 0.0598946381628045
sot_rolling:    | 0.04954476975104505
fk_rolling:     |0.0436001284724147
month_code:     |0.02824945757327016
hour:           |0.010919854985633156
pkatt_rolling:  |0.010586125467140535
day_code:       |0.009287825612504371
pk_rolling:     |0.007106519128454021



## 4. Backtest (Fun Part!!)
Now for the fun. Backtesting betting strategies. After research I found 3 main strategies used in sport betting :
- Expected Value betting
  <ul style="list-style-type: none;"> 
     <li>Expected Value is a measure of how statistacly profitable a bet is i.e after we plug in our P(Win)(percentage chance of a win) the equation spits out a positive or negative number. If the number is postive we should place the bet. </li>
  <li> Calculating the EV using EV = P(win) x potential_winnings - P(Loss) x potential_loss.</li>
  <li> Risk: if the percentage by our model isn`t accurate we will be betting on negative EV bets when we believe we are only betting on positive ones </li>
</ul>

- Betting a percentage of account value
  <ul style="list-style-type: none;">
  <li> Bet 10% of your account value each bet</li>
  <li>Risk: This method has high volitility</li>
   <li>Opportunity: The strategy attempts to minimize losses on losing streaks and maximize wins on winning streaks. </li>
</ul>

- Betting a flat value
  <ul style="list-style-type: none;">
  <li> Bet a consistent amount of £10 each time</li>
  <li>Risk: This method has a consistent risk and low volitility</li>
</ul>

![Backtest Strategies](images/strategy_plots(corrected).png)

### Outcome 
The graph shows that the best strategies use **variable stake sizes** .This doesnt come for free. The addedd volatility increases with account size as seen in the major dips in account balance.

|            Strategy           |   ROI  | Win Rate | Max Drawdown | Peak Profit |
|:-----------------------------:|:------:|:--------:|:------------:|:-----------:|
|         Raw Strategy          | -11.4  |   49.1   |    -140.40   |     0.00    |
|      Percentage Strategy      | 144.2  |   58.9   |    -446.36   |   2333.22   |
|           EV Strategy         |  16.9  |   61.2   |     -51.80   |    188.30   |
| EV and Percentage Bet Strategy| 127.4  |   58.0   |    -452.92   |   2104.63   |

The KPI table helps us identify 2 top strategies by diferent metrics

**Top ROI**
1.Percentage Strategy
The top ROI of 144% it lets the winners run and culls the losers

Investor Psychometric Analysis: Suited to growing an account and is extreemly emotionally stable and a big apetite for risk , able to stomach the 30% plus swings in account balance

**Minimum Drawdown**
1. EV Strategy Minimum Drawdown
Focused on minimizing loss are we shocked that the EV performed so well on this metric.

Investor Psychometric Analysis:  Such a strategy would be suited to a large bankroll who is risk averse
### Expected Value
The use of expected value was underwhelming when coupled with variable betting. 
It did not yield better returns than a basic percentage strategy alone. 
There are a few reasons for this:

- Un-Precisise ability to predict Probability of a win. This lead to bet placing on games that donn`t truly have positive EV
- EV calcualted with variable stake. This increased the likelyhood of positive EV bets being placed with higher account balances

## Next Steps
A precision of 64% is indicative of a primitive model. The next steps are aimed at increasing this metric to cerca 80% for a reliable model.

### Player Data
For those of you that explored the FBref pge you will have seen the vast amount of player data available. I would like to incorporate player ratings into the model to increase the models ability to predict from form. As well as looking at current player ratings understanding injuries and suspensions would be a useful column.

### Experiment With the label
- Multi Class Win Draw Loss:
  
One of the major drawbacks to this model is grouping Losses and draws together to focus only on wins. This means the model is unaware of possible relationships leading to a draw and is misslabeling Draws as Losses. For the purpose of this quick test with a primative model irts ok. In order to scale the model I will need to take the draws into consideration.

- Expected Goals:
  
  After some research into Algorithmic sport betting a popular method emerged going by the ame of Expegted Goals. This method relies on creating a modle to predict amount of goals scored in a game and finding opportunities at the Bookmakers where the odds dont relfect this. This is a proven strategy that would be interesting to try.

  
- Betting a flat value
  <ul style="list-style-type: none;">
  <li> Bet 10% of your account value each bet</li>
  <li>Risk is consistent and managed with this strategy as the stake is consitent</li>
</ul>

## Have we hit the jackpot!!

ROI- 144.2%

Peak Profit £2333.22

Sure its all well mucking about with numbers in python, but what is the reality. After taking into account transaction fees, ROI drops to cerca 130%. But then we must remember bookmakers are in the business of seeing you lose money, not make it! And are quick to cull accounts showing signs of consistent profitable behaviour. Sure it could take a few months and you could make a few dollars before being shut down but nothing that will have you sipping cocktails on a beach in the short term. 

That said the project has been successfull in showing how simple it is to use ML and develop winning strategies from the comfort of your home.
