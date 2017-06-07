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
    matches = get_all_seasons_data(seasons) #matches, team_attributes)

    if compute_form:
        matches = merge_matches_with_form(matches=matches,
                        seasons=seasons,window=window)
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
        #print("shouldnt be here")
        return get_matches_for_season(season = None)
    start_season = seasons[0]
    matches = get_matches_for_season(start_season)

    for s in range(1,len(seasons)):
        #print("or here yet")
        matches = matches.append(get_matches_for_season(seasons[s]),
                                 ignore_index=True)

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
        attrs = attrs.append(get_attributes_for_season(seasons[s]),
                             ignore_index=True)

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

def merge_matches_with_form(matches, seasons=None, window=3):
    '''
        Generate the matches with form dataframe
    '''
    avg_cols =['match_id','home_team_win_average','home_team_draw_average',
               'home_team_lose_average', 'away_team_win_average',
               'away_team_draw_average','away_team_lose_average']

    # get the matches with form
    try:
        mwf = pd.read_pickle('matches_with_form.p')
    except Exception as e:
        mwf = get_all_seasons_data(None)
        mwf = compute_all_forms(mwf,window=window)
        mwf.to_pickle('matches_with_form.p')

    # filter the season ...
    if seasons:
        mwf = mwf[mwf['season'].isin(seasons)]
    # then only filter the specified columns
    mwf = mwf[avg_cols]

    #print(mwf.columns.T)
    #print()
    #print(matches.columns.T)
    #idxs = matches[matches['match_id'].isin(matches_away['match_id'].values)].index
    matches = pd.merge(left=matches, right=mwf, how='left',
                        left_on='match_id', right_on='match_id')
    #print()
    #print(matches.columns.T)
    #assert(-1==1)

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
    #print(matches.shape)
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
    avg_cols =['home_team_win_average','home_team_draw_average',
               'home_team_lose_average', 'away_team_win_average',
               'away_team_draw_average','away_team_lose_average']

    matches['home_team_won'] = 1*(matches['home_team_goal'] > matches['away_team_goal'])
    matches['home_team_drew'] = 1*(matches['home_team_goal'] == matches['away_team_goal'])
    matches['home_team_lost'] = 1*(matches['home_team_goal'] < matches['away_team_goal'])
    matches['away_team_won'] = 1*(matches['home_team_goal'] < matches['away_team_goal'])
    matches['away_team_lost'] = 1*(matches['home_team_goal'] > matches['away_team_goal'])
    matches['away_team_drew'] = 1*(matches['home_team_goal'] == matches['away_team_goal'])

    in_cols = ['home_team_won','home_team_drew','home_team_lost',
               'away_team_won','away_team_drew','away_team_lost']
    all_cols = in_cols
    all_cols.extend(['home_team_goal','away_team_goal','match_id'])
    #this_team_win_average','this_team_lose_average','this_team_draw_average','
    roll_cols = ['match_id','this_team_win_average','this_team_lose_average',
                 'this_team_draw_average']

    # get teams
    teams = get_all_teams()

    #print("Matches shape D {}".format(matches.shape))
    #matches_form = pd.DataFrame()
    to_rename = {'this_team_win_average_x':'this_team_win_average',
        'this_team_draw_average_x':'this_team_draw_average',
        'this_team_lose_average_x':'this_team_lose_average'}

    for t in unique_teams:
        #print(t)
        team_name = teams[teams['team_api_id']==t]['team_long_name']
        print("Team_name {}".format(team_name.values[0]))
        # get the home teams for this team
        dm = matches[(matches['home_team_api_id'] == t) |
                    (matches['away_team_api_id'] == t)]
        #print("dm size {}".format(dm.shape))

        matches_home = matches[matches['home_team_api_id'] == t].copy()
        matches_home = matches_home[all_cols]
        # if we won then assign this teams win, lose, draw stats
        matches_home['this_team_won'] = matches_home['home_team_won']
        matches_home['this_team_drew'] = matches_home['home_team_drew']
        matches_home['this_team_lost'] = matches_home['home_team_lost']
        #print("matches_home size {}".format(matches_home.shape))
        # ... repeat this for the away team
        matches_away = matches[matches['away_team_api_id'] == t].copy()
        matches_away = matches_away[all_cols]
        matches_away['this_team_won'] = matches_away['away_team_won']
        matches_away['this_team_drew'] = matches_away['away_team_drew']
        matches_away['this_team_lost'] = matches_away['away_team_lost']
        #print("matches_away size {}".format(matches_away.shape))

        matches_t = matches_home.append(matches_away)
        matches_t.fillna(0, inplace=True)
        #print("matches_t size {}".format(matches_t.shape))

        # compute the rolling statistics
        matches_t.loc[:,'this_team_win_average'] = matches_t['this_team_won'].rolling(window,
                                        win_type='triang').sum() / (1.0*window)
        matches_t.loc[:,'this_team_draw_average'] = matches_t['this_team_drew'].rolling(window,
                                        win_type='triang').sum() / (1.0*window)
        matches_t.loc[:,'this_team_lose_average'] = matches_t['this_team_lost'].rolling(window,
                                        win_type='triang').sum() / (1.0*window)
        # fill in Nans for the first window -1 matches
        matches_t.fillna(0, inplace=True)
        #print("matches_t size {}".format(matches_t.shape))

        # then reassign the computed average
        matches_t=matches_t[roll_cols]
        matches_home = pd.merge(left=matches_home, right=matches_t, how="left",
                            left_on = 'match_id', right_on='match_id')
        # #print(matches_home.columns.T)
        # matches_home['home_team_win_average'] = matches_home['this_team_win_average']
        # matches_home['home_team_draw_average'] = matches_home['this_team_draw_average']
        # matches_home['home_team_lose_average'] = matches_home['this_team_lose_average']
        # matches_home.drop(['this_team_win_average','this_team_draw_average',
        #                     'this_team_lose_average'], axis=1,inplace=True)
        #print(matches_home.columns.T)
        #print(matches_home)

        #print("matches_home size {}".format(matches_home.shape))

        matches_away = pd.merge(left=matches_away, right=matches_t, how="left",
                            left_on = 'match_id', right_on='match_id')
        # matches_away['away_team_win_average'] = matches_away['this_team_win_average']
        # matches_away['away_team_draw_average'] = matches_away['this_team_draw_average']
        # matches_away['away_team_lose_average'] = matches_away['this_team_lose_average']
        # matches_away.drop(['this_team_win_average','this_team_draw_average',
        #                     'this_team_lose_average'], axis=1,inplace=True)
        #print("matches_away size {}".format(matches_away.shape))
        #print(matches_away.columns.T)
        # matches_form = matches_form.append(matches_away, ignore_index=False)
        # matches_form = matches_form.append(matches_home, ignore_index=False)
        # print("matches_form size {}".format(matches_form.shape))
        # print("")

        matches_away = matches_away[roll_cols]
        #print("matches_away size {}".format(matches_away.shape))
        matches_home = matches_home[roll_cols]
        #print("matches_home size {}".format(matches_home.shape))

        #print("matches size {}".format(matches.shape))
        matches = pd.merge(left=matches, right=matches_home, how='left',
                            left_on='match_id', right_on='match_id')
        #print(matches.columns.T)
        #assert(-1==1)
        idxs = matches[matches['match_id'].isin(matches_home['match_id'].values)].index
        #print(idxs)
        #print(len(idxs))
        matches.loc[idxs,'home_team_win_average'] = matches['this_team_win_average_y']
        matches.loc[idxs,'home_team_draw_average'] = matches['this_team_draw_average_y']
        matches.loc[idxs,'home_team_lose_average'] = matches['this_team_lose_average_y']
        #print("matches size {}".format(matches.shape))
        matches.drop(['this_team_win_average_y','this_team_draw_average_y',
                            'this_team_lose_average_y'], axis=1,inplace=True)
        matches.rename(columns=to_rename, inplace=True)
        #print(matches.columns.T)
        #print("matches size {}".format(matches.shape))

        matches = pd.merge(left=matches, right=matches_away, how='left',
                            left_on='match_id', right_on='match_id')
        idxs = matches[matches['match_id'].isin(matches_away['match_id'].values)].index
        matches.loc[idxs,'away_team_win_average'] = matches['this_team_win_average_y']
        matches.loc[idxs,'away_team_draw_average'] = matches['this_team_draw_average_y']
        matches.loc[idxs,'away_team_lose_average'] = matches['this_team_lose_average_y']
        #print("matches size {}".format(matches.shape))
        #print(matches.columns.T)
        matches.drop(['this_team_win_average_y','this_team_draw_average_y',
                            'this_team_lose_average_y'], axis=1,inplace=True)
        matches.rename(columns=to_rename, inplace=True)
        #print("matches size {}".format(matches.shape))
        #print("")
        #assert(-1==1)


    # then merge back into the matches
    #allnas = matches.isnull().any()
    #print(matches.info())
    #print("Matches size {}".format(matches.shape))
    #print("Forms size {}".format(matches_form.shape))
    #matches.drop(in_cols, axis =1,inplace=True)
    #print(matches[avg_cols].tail(12))
    #assert(-1==1)
    #matches = pd.merge(left=matches_t, right=matches_t, how="left",
    #                    left_on = 'match_id', right_on='match_id')

    #matches = matches.drop(in_cols, axis =1)
    #print(matches.columns.T)
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
