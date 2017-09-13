# Scripts:
- scripts/epl_new.py: 
    + The main script that implements the learning classifiers, runs the parameter tunings and performance metrics.
- scripts/helpers.py:
    + A set of helper functions for importing data from the source dataset/database, handling data cleanup and encoding, and adding additional features based on the desired analysis
- scripts/process_538_predictions.py:
    + a script that uses data in FiveThirtyEight2016-17predictions.xlsx to get performance metrics for the FiveThirtyEight benchmark
- scripts/process_lawro_predicts.py
    + a script that generates performance metrics for Mark lawrenson predictions for the 2015/16 EPL season. This is another benchmark
- 538scraper.py
    + a script use to web scrape the FiveThirtyEight match predictions from https://projects.fivethirtyeight.com/soccer-predictions/premier-league/


# Inputs
- input/database.sqlite:
    + This is the source dataset obtained https://www.kaggle.com/hugomathien/soccer. It is a large sqlite database which can be downloaded as a zip and uncompressed. It has to be placed in a folder named 'input' outside the scripts folder
    + processing of this file is performed in the helpers.py script
- FiveThirtyEight2016-17predictions.xlsx:
    + contains the FiveThirtyEight 2016/17 EPL season predictions
    + this is used as a benchmark
    + run the process_538_predictions.py script to process this and generate F1 scores and confusion matrix
    + scraped from https://projects.fivethirtyeight.com/soccer-predictions/premier-league/
- Lawro2015-16predictions.xlsx:
    + contains the predictions from BBC pundit Mark lawrenson
    + this is used as a benchmark
    + run the process_lawro_predicts.py scrip to process and generate both F1 scores and a confusion matrix
    + data is scraped from link below and then processed externally in excel
    + http://www.myfootballfacts.com/Mark_Lawrenson_Predictions_2015-16.html

# How to:
- the code was tested using an Anaconda environment with Python 3.4.5
- it uses Sklearn, pandas, matplotlib and numpy
- Running the code:
    + run the scripts/epl_new.py an integer command line argument
        * example epl_new.py 1
    + the script takes a single integer argument. integers can range from 1 to 5
    + Each integer runs a section of code (analysis) with descriptions below
    + during processing the script may generate additional output files launch matplotlib figures
    - each section can be run out of sequence

- Command line arguments and corresponding functionality:
    + 1:
        * Generates training and test scores for each classifier
        * Output used for Tables 1 & 2 in the project report
        * Does not use cross-validation 
        * Uses only features in existing dataset
        * It loops over all classifiers, fits a model and uses it to compute training and test scores for selected metrics
    + 2:
        * Analysis 2:
        * Uses training data to get the performance scores using k-folds cross-validation with a k=5
        * Output used for Table 3 in the project report
        * This uses the preprocessed data with only existing features
        * it loops over all classifiers and reports back the cross-validation scores for a number of metrics
    
    + 3:
        Explores the impact of the additional features:
        * The team form features
                + This loops over multiple windows and computes scores for each classifier 
                + Change variable in this section of code to: 
                        - compute_form = True,  
                        - exclude_firstn = True, 
                        - home_advantage = None, 
                + Output used to plot Figures 7 & 8 in the project report
        * The home team advantage features
                + This resets the team form window to 0 and adds a set of new features to capture the home advantage. 
                + There are two ways to compute the home form: goals or points
                + Change variable in this section of code to:
                        - compute_form = False,  
                        - exclude_firstn = False, 
                        - home_advantage = 'points' or 'goals'
                + Output use for Tables 4 & 5 in the project report
    + 4:
        * This section does the parameter tuning for a subset of algorithms
        * Assumes a fixed home form window based on the results of analysis 3
        * Output used for Table 6 in the project report
    + 5
        * This section computes the test results for the Adaboost and SGDC classifiers
        * It also generates the ROC curves using Matplotlib (plot windows do open) and confusion matrices
        * Output used for Tables 7 - 9, and Figures 9 - 10 in the project report 


# Obtaining the European soccer data set
- The dataset can be obtained by downloading a zipped sqlite file from:
    https://www.kaggle.com/hugomathien/soccer/downloads/soccer.zip
- the SQlite database file is named database.sqlite
- The file has the following tables and fields
###Player_Attributes	
    	id, player_fifa_api_id, player_api_id, date, overall_rating, potential, preferred_foot, attacking_work_rate, defensive_work_rate, crossing, finishing, heading_accuracy, short_passing, volleys, dribbling, curve, free_kick_accuracy, long_passing, ball_control, acceleration, sprint_speed, agility, reactions, balance, shot_power, jumping, stamina, strength, long_shots, aggression, interceptions, positioning, vision, penalties, marking, standing_tackle, sliding_tackle, gk_diving, gk_handling, gk_kicking, gk_positioning, gk_reflexes
###Player	
        id, player_api_id, player_name, player_fifa_api_id, birthday, height, weight
###Match	
        id, country_id, league_id, season, stage, date, match_api_id, home_team_api_id, away_team_api_id, home_team_goal, away_team_goal, home_player_X1, home_player_X2, home_player_X3, home_player_X4, home_player_X5, home_player_X6, home_player_X7, home_player_X8, home_player_X9, home_player_X10, home_player_X11, away_player_X1, away_player_X2, away_player_X3, away_player_X4, away_player_X5, away_player_X6, away_player_X7, away_player_X8, away_player_X9, away_player_X10, away_player_X11, home_player_Y1, home_player_Y2, home_player_Y3, home_player_Y4, home_player_Y5, home_player_Y6, home_player_Y7, home_player_Y8, home_player_Y9, home_player_Y10, home_player_Y11, away_player_Y1, away_player_Y2, away_player_Y3, away_player_Y4, away_player_Y5, away_player_Y6, away_player_Y7, away_player_Y8, away_player_Y9, away_player_Y10, away_player_Y11, home_player_1, home_player_2, home_player_3, home_player_4, home_player_5, home_player_6, home_player_7, home_player_8, home_player_9, home_player_10, home_player_11, away_player_1, away_player_2, away_player_3, away_player_4, away_player_5, away_player_6, away_player_7, away_player_8, away_player_9, away_player_10, away_player_11, goal, shoton, shotoff, foulcommit, card, cross, corner, possession, B365H, B365D, B365A, BWH, BWD, BWA, IWH, IWD, IWA, LBH, LBD, LBA, PSH, PSD, PSA, WHH, WHD, WHA, SJH, SJD, SJA, VCH, VCD, VCA, GBH, GBD, GBA, BSH, BSD, BSA
###League	
        id, country_id, name
###Country	
        id, name
###Team	
        id, team_api_id, team_fifa_api_id, team_long_name, team_short_name
###Team_Attributes
        id, team_fifa_api_id, team_api_id, date, buildUpPlaySpeed, buildUpPlaySpeedClass, buildUpPlayDribbling, buildUpPlayDribblingClass, buildUpPlayPassing, buildUpPlayPassingClass, buildUpPlayPositioningClass, chanceCreationPassing, chanceCreationPassingClass, chanceCreationCrossing, chanceCreationCrossingClass, chanceCreationShooting, chanceCreationShootingClass, chanceCreationPositioningClass, defencePressure, defencePressureClass, defenceAggression, defenceAggressionClass, defenceTeamWidth, defenceTeamWidthClass, defenceDefenderLineClass

- For my analysis, I merged the first eleven columns of the the Match table with the Team attributes table using the home_team_api_id and away_team_api_id fields.