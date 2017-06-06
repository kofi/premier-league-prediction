'''
###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
#import warnings
#warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
###########################################
'''

import itertools
import pickle
import matplotlib
matplotlib.use('TkAgg') #"Qt5Agg")
import matplotlib.pyplot as plt
import sqlite3 as sql
import numpy as np
import pandas as pd
#import re


def import_datasets():
    '''
        Get matches for each season
    '''
    #print('importing datasets ...')
    con = None
    con = sql.connect('../input/database.sqlite')
    # get the leagues
    try:
        pd.read_pickle('all_leagues.p')
    except Exception:
        query = "select * from League"
        all_leagues = pd.read_sql(query, con=con)
        all_leagues.to_pickle('all_leagues.p')
    # get the matches
    try:
        pd.read_pickle('all_matches.p')
    except Exception:
        query = "Select * from Match"
        all_matches = pd.read_sql(query, con=con)
        all_matches.to_pickle('all_matches.p')
    # get the teams
    try:
        pd.read_pickle('all_teams.p')
    except Exception:
        query = "Select * from Team"
        all_teams = pd.read_sql(query, con=con)
        all_teams.to_pickle('all_teams.p')
    # get the team attributes
    try:
        pd.read_pickle('all_team_attributes.p')
    except Exception:
        query = "Select * from Team_Attributes"
        all_team_attributes = pd.read_sql(query, con=con)
        all_team_attributes.to_pickle('all_team_attributes.p')
    #print('... completing the data import')

def preprocess_matches_for_season(seasons, compute_form = False, window=3):
    '''
    do all the preprocessing and return a matches dataframe ready
    for learning
    '''
    #print(seasons)
    matches = get_all_seasons_data(seasons) #matches, team_attributes)
    #print(matches.shape)
    if compute_form:
        matches = compute_all_forms(matches,window=window)

    return matches

def get_all_matches():
    '''
    return the dataframe of leagues
    '''
    try:
        matches = pd.read_pickle('all_matches.p')
    except Exception as e:
        import_datasets()
        matches = pd.read_pickle('all_matches.p')
    return matches

def get_all_leagues():
    '''
    return the dataframe of leagues
    '''
    try:
        leagues = pd.read_pickle('all_leagues.p')
    except Exception as e:
        import_datasets()
        leagues = pd.read_pickle('all_leagues.p')
    return leagues

def get_all_teams():
    '''
    return the dataframe of teams
    '''
    try:
        teams = pd.read_pickle('all_teams.p')
        teams.shape
    except Exception as e:
        import_datasets()
        teams = pd.read_pickle('all_teams.p')
    return teams

def get_all_team_attributes():
    '''
    return the dataframe of team_attributes
    '''
    try:
        team_attributes = pd.read_pickle('all_team_attributes.p')
    except Exception as e:
        import_datasets()
        team_attributes = pd.read_pickle('all_team_attributes.p')
    return team_attributes

def get_team_id(team_name):
    '''
    Get the id for a given team
    '''
    teams = get_all_teams()
    if team_name is None:
        return None
    my_team = teams[teams['team_long_name'].str.contains(team_name)]
    return my_team

def get_league_id(league_name):
    '''
    Get the id for a given league
    '''

    league = get_all_leagues()
    my_league = league[league['name'].str.contains(league_name)]['id']
    return my_league.values[0]

def get_season_as_date(season):
    '''
    Returns a formatted date from a season
    '''
    return "{}-07-01 00:00:00".format(season)

def get_matches_for_season(season=None,league_name='England'):
    '''
        Get matches for each season
    '''
    league_id = get_league_id(league_name=league_name)
    matches = get_all_matches()
    matches = matches[matches['league_id']==league_id]
    if season is not None:
        matches = matches[matches['season'].str.contains(season)]

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

    return matches

def get_attributes_for_seasons(seasons):
    '''
    Get matches for multiple seasons
    '''
    if seasons is None:
        return get_attributes_for_season(season=None)
    start_season = seasons[0]
    attrs = get_attributes_for_season(start_season)

    for s in range(1,len(seasons)):
        attrs = attrs.append(get_attributes_for_season(seasons[s]), ignore_index=True)

    return attrs

def get_attributes_for_season(season=None):
    '''
    Get team attributes data from the database
    '''
    team_attributes = get_all_team_attributes()
    if season is not None:
        [sstart, ssend] = season.split('/')
        sstart = get_season_as_date(sstart)
        ssend= get_season_as_date(ssend)
        query = 'date >= @sstart and date <= @ssend'
        team_attributes = team_attributes.query(query)
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
    matches = pd.merge(left=matches, right=team_attributes, how='left',
                       left_on='home_team_api_id', right_on='team_api_id')
    matches = matches.drop(['id', 'team_fifa_api_id', 'team_api_id_x',
                            'team_api_id_y','date'], axis=1)
    matches.rename(columns={'buildUpPlaySpeed':'home_buildUpPlaySpeed',
                        'buildUpPlaySpeedClass':'home_buildUpPlaySpeedClass',
                    'buildUpPlayDribbling':'home_buildUpPlayDribbling',
                    'buildUpPlayDribblingClass':'home_buildUpPlayDribblingClass',
                    'buildUpPlayPassing':'home_buildUpPlayPassing',
                    'buildUpPlayPassingClass':'home_buildUpPlayPassingClass',
                    'buildUpPlayPositioningClass':'home_buildUpPlayPositioningClass',
                    'chanceCreationPassing':'home_chanceCreationPassing',
                    'chanceCreationPassingClass':'home_chanceCreationPassingClass',
                    'chanceCreationCrossing':'home_chanceCreationCrossing',
                    'chanceCreationCrossingClass':'home_chanceCreationCrossingClass',
                    'chanceCreationShooting':'home_chanceCreationShooting',
                    'chanceCreationShootingClass':'home_chanceCreationShootingClass',
                    'chanceCreationPositioningClass':'home_chanceCreationPositioningClass',
                    'defencePressure':'home_defencePressure',
                    'defencePressureClass':'home_defencePressureClass',
                    'defenceAggression':'home_defenceAggression',
                    'defenceAggressionClass':'home_defenceAggressionClass',
                    'defenceTeamWidth':'home_defenceTeamWidth',
                    'defenceTeamWidthClass':'home_defenceTeamWidthClass',
                    'defenceDefenderLineClass':'home_defenceDefenderLineClass'},
               inplace=True)

    matches = pd.merge(left=matches, right=team_attributes, how='left',
                       left_on='away_team_api_id', right_on='team_api_id')
    matches = matches.drop(['id', 'team_fifa_api_id', 'team_api_id', 'date'],
                           axis=1)

    matches.rename(columns={'buildUpPlaySpeed':'away_buildUpPlaySpeed',
                    'buildUpPlaySpeedClass':'away_buildUpPlaySpeedClass',
                'buildUpPlayDribbling':'away_buildUpPlayDribbling',
                'buildUpPlayDribblingClass':'away_buildUpPlayDribblingClass',
                'buildUpPlayPassing':'away_buildUpPlayPassing',
                'buildUpPlayPassingClass':'away_buildUpPlayPassingClass',
                'buildUpPlayPositioningClass':'away_buildUpPlayPositioningClass',
                'chanceCreationPassing':'away_chanceCreationPassing',
                'chanceCreationPassingClass':'away_chanceCreationPassingClass',
                'chanceCreationCrossing':'away_chanceCreationCrossing',
                'chanceCreationCrossingClass':'away_chanceCreationCrossingClass',
                'chanceCreationShooting':'away_chanceCreationShooting',
                'chanceCreationShootingClass':'away_chanceCreationShootingClass',
                'chanceCreationPositioningClass':'away_chanceCreationPositioningClass',
                'defencePressure':'away_defencePressure',
                'defencePressureClass':'away_defencePressureClass','defenceAggression':'away_defenceAggression',
                'defenceAggressionClass':'away_defenceAggressionClass',
                'defenceTeamWidth':'away_defenceTeamWidth',
                'defenceTeamWidthClass':'away_defenceTeamWidthClass',
                'defenceDefenderLineClass':'away_defenceDefenderLineClass'},
                   inplace=True)

    return matches

def clean_up_matches(matches):
    '''
    clean up matches dataframe by removing nulls
    also drop some additioonal columns: either ids or
    '''
    matches.index = matches['match_id']
    # then drop the match_id and also drop stage for now
    to_drop = [ 'match_id','stage', 'match_date','home_team_api_id',
            'away_team_api_id','home_team', 'away_team','season',
            'home_buildUpPlayDribbling','away_buildUpPlayDribbling']
    #'home_team_goal', 'away_team_goal',
    # make a copy of the matches dataframe and drop the appropriate fields while deleting the unneeded features
    matches = matches.drop(to_drop, axis =1)
    #print("Matches shape after clean up {}".format(matches.shape))

    return matches

def encode_matches(matches):
    '''
    encode category columns using the dummies to create a column per option
    '''
    # get categorical data ...
    cat_list= matches.select_dtypes(include=['object']).columns.tolist()
    #print("cat list: {}".format(cat_list))
    # then encode those columns ...
    matches = pd.get_dummies(matches, prefix=cat_list)
    #print("Matches shape after encode up {}".format(matches.shape))
    return matches

def get_all_seasons_data(seasons): #matches,tattr):
    '''
    get the number of unique seasons in the matches dataframe
    '''
    matches = get_matches_for_seasons(seasons)
    #print("matches shape 1 {}".format(matches.shape) )
    matches = matches[matches.columns[:11]]
    #print("matches shape 2 {}".format(matches.shape))
    # get attributes
    tattr = get_attributes_for_seasons(seasons)


    if seasons is None:
        seasons = matches['season'].unique()

    teams = get_all_teams()
    # create an empty array to store our data
    newmatches = pd.DataFrame()

    for e in seasons:

        # get the corresponding matches
        m = matches[matches['season'] == e]
        # get the corresponding attributes
        [sstart, ssend] = e.split('/')
        sstartdt = get_season_as_date(sstart)
        ssenddt = get_season_as_date(ssend)
        t = tattr[(tattr['date'] >= get_season_as_date(sstart)) &
                (tattr['date'] < get_season_as_date(ssend))]

        m = merge_matches_teams(m, teams)
        m = merge_matches_attributes(m, t)

        newmatches = newmatches.append(m, ignore_index=False)

    return newmatches


def compute_all_forms(matches,window=3):
    #print(matches.info())
    #print(matches.columns.T)
    #sorted_matches = matches.sort_values(by=['match_id'],axis=0)
    print(matches.shape)
    unique_teams = matches['home_team_api_id'].unique()
    nmatches = matches.shape[0]
    matches['this_team_win_average'] = 0.0
    matches['this_team_draw_average'] = 0.0
    matches['this_team_lose_average'] = 0.0
    matches['home_team_win_average'] = 0.0
    matches['home_team_draw_average'] = 0.0
    matches['home_team_lose_average'] = 0.0
    matches['away_team_win_average'] = 0.0
    matches['away_team_draw_average'] = 0.0
    matches['away_team_lose_average'] = 0.0


    matches['home_team_won'] = 1*(matches['home_team_goal'] > matches['away_team_goal'])
    matches['home_team_drew'] = 1*(matches['home_team_goal'] == matches['away_team_goal'])
    matches['home_team_lost'] = 1*(matches['home_team_goal'] < matches['away_team_goal'])
    matches['away_team_won'] = 1*(matches['home_team_goal'] < matches['away_team_goal'])
    matches['away_team_lost'] = 1*(matches['home_team_goal'] > matches['away_team_goal'])
    matches['away_team_drew'] = 1*(matches['home_team_goal'] == matches['away_team_goal'])

    in_cols = ['home_team_won','home_team_drew','home_team_lost',
               'away_team_won','away_team_drew','away_team_lost']
    all_cols = in_cols.append(['this_team_win_average','this_team_lose_average',
               'this_team_draw_average',
               'home_team_goal','away_team_goal','match_id'])
    roll_cols = ['this_team_win_average','this_team_lose_average',
                 'this_team_draw_average']

    #print("Matches shape D {}".format(matches.shape))
    for t in unique_teams:
        print(t)
        matches_home = matches[matches['home_team_api_id'] == t][all_cols]
        matches_home['this_team_won'] = matches_home['home_team_won']
        matches_home['this_team_drew'] = matches_home['home_team_drew']
        matches_home['this_team_lost'] = matches_home['home_team_lost']
        matches_away = matches[matches['away_team_api_id'] == t][all_cols]
        matches_away['this_team_won'] = matches_home['away_team_won']
        matches_away['this_team_drew'] = matches_home['away_team_drew']
        matches_away['this_team_lost'] = matches_home['away_team_lost']
        matches_t = matches_home.append(matches_away)
        matches_t.fillna(0, inplace=True)

        matches_t['this_team_win_average'] = matches_t['this_team_won'].rolling(window,
                                        win_type='triang').sum() / (1.0*window)
        matches_t['this_team_draw_average'] = matches_t['this_drew_won'].rolling(window,
                                        win_type='triang').sum() / (1.0*window)
        matches_t['this_team_lose_average'] = matches_t['this_lost_won'].rolling(window,
                                        win_type='triang').sum() / (1.0*window)
        print(matches_t)
        #matches_t.set_index('match_id',inplace = True)
        #matches_t.sort_values(by=['match_id'],axis=0, inplace=True)
        #print(matches_t.head(15).T)
        #curr_form = matches_t[in_cols].rolling(window,win_type='triang').sum() / (1.0*window)
        #mloc_away = matches.loc[(matches['home_team_api_id'] == t)]

        assert(-1==1)

        # # get the indexes
        # mloc = matches[(matches['home_team_api_id'] == t)  | (matches['away_team_api_id'] == t)].index
        # #print(mloc) #matches.loc[mloc])
        # #assert(-1==1)
        # mloc_away =  matches.loc[(matches['home_team_api_id'] == t)]
        # mloc_home =  matches.loc[(matches['away_team_api_id'] == t)]#.index
        # tmatched = matches.loc[mloc]
        # #print(tmatched)
        # curr_form = tmatched[in_cols].rolling(window,win_type='triang').sum() / (1.0*window)
        # curr_form.fillna(0,inplace= True)
        # #curr_form.reset_index(inplace= True)
        # #print(curr_form)
        # for l in mloc_away:
        #     #print(l)
        #     matches.loc[l,'away_team_win_average'] = curr_form.ix[l]['away_team_won']
        #     matches.loc[l,'away_team_draw_average'] = curr_form.ix[l]['away_team_drew']
        #     matches.loc[l,'away_team_lose_average'] = curr_form.ix[l]['away_team_lost']
        #     #matches.loc[l,'away_form'] = 3*curr_form.ix[l]['away_team_won'] + 1curr_form.ix[l]['away_team_lost']
        #
        # for l in mloc_home:
        #     matches.loc[l,'home_team_win_average'] = curr_form.ix[l]['home_team_won']
        #     matches.loc[l,'home_team_draw_average'] = curr_form.ix[l]['home_team_drew']
        #     matches.loc[l,'home_team_lose_average'] = curr_form.ix[l]['home_team_lost']


    matches = matches.drop(in_cols, axis =1)

    return matches


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
