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

## 4. Backtest
Now for the fun bit. Backtesting betting strategies. After research I found 3 main strategies used in sport betting :
- Expected Vlaue betting
  Calculating the EV using EV = P(win) x potential_winnings - P(Loss) x potential_loss
- Betting a percentage of account value
  Bet 10% of your account valueeach time
  - Pros cons # todo 
- Betting a flat value
- pros cons #todo
