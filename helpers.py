'''
###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
#import warnings
#warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
###########################################
'''

import itertools
import matplotlib
matplotlib.use('TkAgg') #"Qt5Agg")
import matplotlib.pyplot as plt
import sqlite3 as sql
import numpy as np
import pandas as pd


def preprocess_matches_for_season(seasons):
    '''
    do all the preprocessing and return a matches dataframe ready 
    for learning
    '''
    # if seasons:
    #     # get matches for the season
    #     matches = get_matches_for_season(season) 
    #     # We only need the first 11 columns for this analysis. the other data is betting data
    #     matches = matches[matches.columns[:11]]

    #     # get attributes
    #     team_attributes = get_attributes_for_seasons(season)

    #     # get teams
    #     teams = get_teams()

    #     # do merges of matches with teams and team_attributes
    #     matches = merge_matches_teams(matches, teams)
    #     matches = merge_matches_attributes(matches, team_attributes)
    # else:
    matches = get_all_seasons_data(seasons) #matches, team_attributes)

    #assert(1==-1)

    return matches


def get_teams():
    '''
    return the dataframe of teams
    '''
    con = None
    con = sql.connect('../input/database.sqlite')
    # create the cursor
    cur= con.cursor()
    query = "SELECT * FROM Team"
    return pd.read_sql(query,con=con)

def get_team_id(team_name):
    '''
    Get the id for a given team
    '''
    con = None
    con = sql.connect('../input/database.sqlite')
    # create the cursor
    cur= con.cursor()
    query = "SELECT team_api_id FROM Team where team_long_name like '%{}%'".format(team_name)

    return pd.read_sql(query,con=con)

def get_season_as_date(season):
    '''
    Returns a formatted date from a season
    '''
    return "{}-07-01 00:00:00".format(season)

def get_matches_for_season(season=None):
    '''
        Get matches for each season
    '''
    con = None
    con = sql.connect('../input/database.sqlite')
    #cur = con.cursor()
    query = "select * from League where name like '%England%'"
    eplinfo = pd.read_sql(query, con=con)

    if (season is not None) and len(season)==1:
        query = "Select * from Match \
                where league_id = {} and season='{}'".format(eplinfo['id'][0], season[0])
    else:
        query = "Select * from Match where league_id = {} ".format(eplinfo['id'][0])
    matches = pd.read_sql(query, con=con)

    return matches

def get_matches_for_seasons(seasons):
    '''
    Get matches for multiple seasons
    '''
    if seasons is None:
        return get_matches_for_season(season = None)
    start_season = seasons[0]
    matches = get_matches_for_season(start_season)
    for s in range(1,len(seasons)):
        matches = matches.append(get_matches_for_season(seasons[s]), ignore_index=True)
        #print(matches.shape)
    return matches

def get_attributes_for_seasons(seasons):
    '''
    Get matches for multiple seasons
    '''
    if seasons is None:
        return get_attributes_for_season(season=None)
    start_season = seasons[0]
    attrs = get_attributes_for_season(start_season)
    #print(attrs.columns.T)
    #print("attributes shape {}".format(*attrs.shape))
    for s in range(1,len(seasons)):
        attrs = attrs.append(get_attributes_for_season(seasons[s]), ignore_index=True)
        #print("attributes shape {}".format(*attrs.shape))
    return attrs

def get_attributes_for_season(season=None):
    '''
    Get team attributes data from the database
    '''
    con = None
    con = sql.connect('../input/database.sqlite')

    if season:
        [sstart, ssend] = season.split('/')
        query = "SELECT * FROM Team_Attributes where date >= '{}' and date <='{}'".format(
                get_season_as_date(sstart), get_season_as_date(ssend))
    else:
        query = "SELECT * FROM Team_Attributes"
    
    team_attributes = pd.read_sql(query, con=con)
    return team_attributes

def merge_matches_teams(matches, teams):
    '''
    prepare the season data by doing merges and joins for matches and teams
    '''
    matches = pd.merge(left=matches, right=teams, how='left', left_on='home_team_api_id',
                    right_on='team_api_id')
    matches = matches.drop(['country_id','league_id','id_y','team_api_id',
                    'team_fifa_api_id','team_short_name'], axis=1)

    matches.rename(columns={'id_x':'match_id','date':'match_date','team_long_name':'home_team'}, 
                inplace=True)
    matches = pd.merge(left=matches, right=teams, how='left', left_on='away_team_api_id', 
                right_on='team_api_id')
    matches = matches.drop(['id', 'match_api_id', 'team_fifa_api_id', 'team_short_name'], axis=1)

    matches.rename(columns={'team_long_name':'away_team'}, inplace=True)
    matches.head()

    return matches

def merge_matches_attributes(matches, team_attributes):
    '''
    Merge the matches and team attributes data
    '''
    matches = pd.merge(left=matches, right=team_attributes, how='left', left_on='home_team_api_id', right_on='team_api_id')
    matches = matches.drop(['id', 'team_fifa_api_id', 'team_api_id_x','team_api_id_y','date'], axis=1)
    matches.rename(columns={'buildUpPlaySpeed':'home_buildUpPlaySpeed','buildUpPlaySpeedClass':'home_buildUpPlaySpeedClass',
                        'buildUpPlayDribbling':'home_buildUpPlayDribbling',
                        'buildUpPlayDribblingClass':'home_buildUpPlayDribblingClass',
                        'buildUpPlayPassing':'home_buildUpPlayPassing','buildUpPlayPassingClass':'home_buildUpPlayPassingClass',
                        'buildUpPlayPositioningClass':'home_buildUpPlayPositioningClass',
                        'chanceCreationPassing':'home_chanceCreationPassing','chanceCreationPassingClass':'home_chanceCreationPassingClass',
                        'chanceCreationCrossing':'home_chanceCreationCrossing',
                        'chanceCreationCrossingClass':'home_chanceCreationCrossingClass','chanceCreationShooting':'home_chanceCreationShooting',
                        'chanceCreationShootingClass':'home_chanceCreationShootingClass','chanceCreationPositioningClass':'home_chanceCreationPositioningClass','defencePressure':'home_defencePressure',
                        'defencePressureClass':'home_defencePressureClass','defenceAggression':'home_defenceAggression',
                        'defenceAggressionClass':'home_defenceAggressionClass','defenceTeamWidth':'home_defenceTeamWidth',
                        'defenceTeamWidthClass':'home_defenceTeamWidthClass','defenceDefenderLineClass':'home_defenceDefenderLineClass'}, inplace=True)

    matches = pd.merge(left=matches, right=team_attributes, how='left', left_on='away_team_api_id', right_on='team_api_id')
    matches = matches.drop(['id', 'team_fifa_api_id', 'team_api_id', 'date'], axis=1)

    matches.rename(columns={'buildUpPlaySpeed':'away_buildUpPlaySpeed','buildUpPlaySpeedClass':'away_buildUpPlaySpeedClass',
                        'buildUpPlayDribbling':'away_buildUpPlayDribbling',
                        'buildUpPlayDribblingClass':'away_buildUpPlayDribblingClass',
                        'buildUpPlayPassing':'away_buildUpPlayPassing','buildUpPlayPassingClass':'away_buildUpPlayPassingClass',
                        'buildUpPlayPositioningClass':'away_buildUpPlayPositioningClass',
                        'chanceCreationPassing':'away_chanceCreationPassing','chanceCreationPassingClass':'away_chanceCreationPassingClass',
                        'chanceCreationCrossing':'away_chanceCreationCrossing',
                        'chanceCreationCrossingClass':'away_chanceCreationCrossingClass','chanceCreationShooting':'away_chanceCreationShooting',
                        'chanceCreationShootingClass':'away_chanceCreationShootingClass','chanceCreationPositioningClass':'away_chanceCreationPositioningClass','defencePressure':'away_defencePressure',
                        'defencePressureClass':'away_defencePressureClass','defenceAggression':'away_defenceAggression',
                        'defenceAggressionClass':'away_defenceAggressionClass','defenceTeamWidth':'away_defenceTeamWidth',
                        'defenceTeamWidthClass':'away_defenceTeamWidthClass','defenceDefenderLineClass':'away_defenceDefenderLineClass'}, inplace=True)
    #print(matches.columns)

    ## create the output variables
    # matches['home_team_points'] = 3*(matches['home_team_goal'] > matches['away_team_goal']) + 1*(matches['home_team_goal'] == matches['away_team_goal'])
    # matches['home_team_outcome'] = 'draw'
    # matches.loc[matches['home_team_goal'] > matches['away_team_goal'],['home_team_outcome']] = 'win'
    # matches.loc[matches['home_team_goal'] < matches['away_team_goal'],['home_team_outcome']] = 'lose'


    #print win_idx
    #matches = matches.lambda(:)
    #    matches['home_team_outcome']= 'win'
    #if matches['home_team_goal'] < matches['away_team_goal']:
    #    matches['home_team_outcome']= 'lose'
    
    #matches['home_team_points'] = 1*(matches['home_team_goal'] == matches['away_team_goal'])
    #print(matches.shape)
    #matches.head(10)

    return matches

def clean_up_matches(matches):
    '''
    clean up matches dataframe by removing nulls
    also drop some additioonal columns: either ids or 
    '''
    matches.index = matches['match_id']
    #print(matches.shape)
    # then drop the match_id and also drop stage for now
    to_drop = [ 'match_id', 'stage', 'match_date','home_team_api_id',
            'away_team_api_id','home_team', 'away_team','season',
            'home_buildUpPlayDribbling','away_buildUpPlayDribbling']  
    #'home_team_goal', 'away_team_goal',
    # make a copy of the matches dataframe and drop the appropriate fields while deleting the unneeded features
    matches = matches.drop(to_drop, axis =1)
    #print(matches.shape)

    return matches

def encode_matches(matches):
    '''
    encode category columns using the dummies to create a column per option
    '''
    # get categorical data ...
    cat_list= matches.select_dtypes(include=['object']).columns.tolist()
    # then encode those columns ...
    matches = pd.get_dummies(matches, prefix=cat_list)

    return matches
    
def get_all_seasons_data(seasons): #matches,tattr):
    '''
    get the number of unique seasons in the matches dataframe
    '''
    matches = get_matches_for_seasons(seasons) # if seasons else get_matches_for_season(seasons)   
    matches = matches[matches.columns[:11]]

    # get attributes
    tattr = get_attributes_for_seasons(seasons)

    if seasons is None:
        seasons = matches['season'].unique()
    
    teams = get_teams()
    # create an empty array to store our data
    newmatches = pd.DataFrame()

    for e in seasons:

        # get the corresponding matches
        print(e)
        m = matches[matches['season'] == e]
        #print(m.shape)

        # get the corresponding attributes
        [sstart, ssend] = e.split('/')
        sstartdt = get_season_as_date(sstart)
        ssenddt = get_season_as_date(ssend)
        t = tattr[(tattr['date'] >= get_season_as_date(sstart)) & 
                (tattr['date'] < get_season_as_date(ssend))]

        # # do the on the home team
        # m = pd.merge(left=m, right=t, how='left', left_on='home_team_api_id',right_on='team_api_id')
        # #print m.info()
        # dropcols = ['country_id','league_id','id_y','team_api_id','team_fifa_api_id','team_short_name']
        # dropcols = [c for c in dropcols if c in m.index]
        # m = m.drop(dropcols, axis=1)
        # m.rename(columns={'id_x':'match_id','date':'match_date', 'team_long_name':'home_team'}, 
        #                 inplace=True)

        # # then merge again on the away team
        # m = pd.merge(left=m, right=t, how='left', left_on='away_team_api_id', right_on='team_api_id')
        # dropcols = ['id', 'match_api_id', 'team_fifa_api_id', 'team_short_name']
        # dropcols = [c for c in dropcols if c in m.index]
        # m = m.drop(dropcols, axis=1)
        # m.rename(columns={'team_long_name':'away_team'}, inplace=True)

        m = merge_matches_teams(m, teams)
        m = merge_matches_attributes(m, t)

        # then append to newmatches
        #if newmatches.shape[0] == 0:
        #    newmatches = m.copy()
        #    #print newmatches.shape
        #else:
        newmatches = newmatches.append(m, ignore_index=False)
        #print(newmatches.shape)
        #    print m.shape
        #    print newmatches.shape
        #print m
    

    return newmatches

#print(matches.shape)
#matches = get_all_season_data(matches,team_attributes,'2008/2009')

#nm = nm.drop(['id', 'match_api_id', 'team_fifa_api_id', 'team_short_name'], axis=1)
#nm.rename(columns={'team_long_name':'away_team'}, inplace=True)
#print(nm.shape)
#nm.head(15)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    this is pulled from:
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#
        sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    np.set_printoptions(precision=2)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # import seaborn as sns; sns.set()
    # ax = sns.heatmap(cm)
    # ax.ylabel('True label')
    # ax.xlabel('Predicted label')
