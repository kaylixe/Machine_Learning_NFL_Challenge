# Applied Machine Learning NFL Challenge Project

> Can Machine Learning Beat the Vegas Spread for NFL Games?

> By: Shivam Kapadia & Peter Russell

# Topic Motivation

For Las Vegas, sports wagering is a lucrative business. In 2017, Nevada sportsbooks took in over $ \$ $ 4.8 billion of wagers and won a record $ \$ $ 248.7 million. This cleared the previous record by $17 million [1]. These profits did not come from taking the money of unlucky or unskilled gamblers, but instead comes largely from pairing up opposing bets and collecting the spread between the two offsetting bets. As a result, it is in the sportsbook best interest to find the most fair line that generates the largest volume of bets on both sides. Few are more adept at determining a fair line for a sports game than Vegas sportsbook for this very reason. As a testament to their skill, over the last 15 regular seasons, NFL favorites have gone 1,859-1,860 against the spread, with 111 pushes, which is an amazingly symmetrical distribution [2].

The NFL is the most wagered sport in the country, responsible for 36\% of all bets in Nevada. Baseball is the next closest at 23\% [3]. The United States is currently undergoing to a dramatic transformation as it relates to sports betting. The Supreme Court has decided that states outside of Nevada will be allowed to offer sports betting, which is a windfall that is expected to add $\$ $2.3 billion to the NFL and will make the sportsbook landscape even more competitive [4]. With this in mind, we are interested in applying the developments in machine learning and techniques learned throughout the semester to see if we can build a sports betting model that will not only beat the baseline success rate of 50\%, but also the spread between opposing wagers' payouts of roughly 10 \%. Namely, we hope to build a model that will have a success rate of 60 \% for pure classification predictions of "cover the spread" or "not cover the spread." Additionally, as a stretch goal, we would like to weight our predictions by our confidence level in the predictions for optimal money wagering management.

1. http://www.espn.com/chalk/story/_/id/22273982/record-amounts-money-bet-lost-nevada-2017
2. https://abcnews.go.com/Sports/betting-nfl-season/story?id=57617459
3. https://www.sportsbusinessdaily.com/Journal/Issues/2018/04/16/World-Congress-of-Sports/Research.aspx
4. https://www.legalsportsreport.com/23596/nfl-sports-betting-revenue-survey/

## Data Description

**Source:** Kaggle (public) https://www.kaggle.com/tobycrabtree/nfl-scores-and-betting-data

**NFL Stadium Information (100 x 15):** Name, Location, Open/Close, Type, Address, Weather Station Code, Weather Type, Capacity, Surface, Longitude, Latitude

**NFL Teams (41 x 8):** Name (Short and Long), Conference, Division, Conference (pre-2002), Division (pre-2002)

**Scores (12.4k x 17):** Date, Season, Week, Team Home, Team Away, Stadium, Favorite, Spread (Favorite), Over/Under Line, Weather, Wind, Humidity, Score, Stadium, Playoff/Regular Season

Our data is largely limited to the conditions of the game being played versus the characteristics of the team. We will try to use the dataset to infer features that incorporate the momentum of a team (eg. points scored over the last 3 games, points given up) to see if they hold importance in the model. In some ways though, having previous season data would not be expected be tremendously helpful with the exception of a few teams (eg. Patriots) as roster turnover is large in the NFL and team strength can vary significantly year-to-year. Having week-by-week statistics ('Points Scored per Game','Opponent Points Scored per Game','Giveaways per Game','Takeaways per Game','Yards per Game','Opponent Yards per Game','Penalty Yards per Game') for a team would be help, but the data is difficult to obtain. We have put a request out to a provider of the data and are waiting to hear back (https://www.teamrankings.com/nfl). This would add potentially 6 new features with each having approximately 255 observations (15 years of data available for 17 weeks each).
We have found aggregated season data if we choose to incorporate this, perhaps as a dummy variable if the team was a top 4 finisher at the end of the previous season (https://www.kaggle.com/farmander/nfl-statistics).

## Model Choice

To begin, we will be working with binary classifiers to fit our model where the binary outcomes will be 'covered spread' and 'did not cover spread'. We will be interested in the point spread and the over/under totals for measuring against the spread.
The models that we will be exploring will be Random Forest Classifier to help the model decide which data elements have critical break points that will help us determine if the point spread will be covered. Additionally, we will look at Logistic Regression, KNN and SVM as our other binary classifiers as benchmarks.
Time permitting, we will look at Linear Regression model to predict the point spread and use this difference from the actual point spread to weight the size of our bet. As of now, this is a stretch goal.

## Metrics for Success
Since we are concerned with whether or not our model can successfully predict the correct spread outcome, we will be using accuracy as our scoring mechanism in the binary classifier.
If we are able to reach our stretch goal, we will use mean square error as the metric we wish to minimize for the most accurate spread prediction that we can then compare against the actual spread level for our wager weighting.

## Pipeline Description
Our pipeline will require the integration of both numerical data (team statistics, spreads) along with categorical data (type of stadium, weather conditions). Within the pipeline, we will standardize the team statistics across the entire dataset. We will ultimately lean on this pipeline in our grid search in tuning the hyperparameters to find the best model.