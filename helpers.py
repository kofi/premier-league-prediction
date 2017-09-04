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
import matplotlib.cm as cm
from matplotlib.pyplot import show, draw
import sqlite3 as sql
import numpy as np
import pandas as pd
import pprint 
#import re


def import_datasets(league_name="England"):
    '''
        Get matches for each season
    '''
    #print('importing datasets ...')
    con = None
    con = sql.connect('../input/database.sqlite')
    # get the leagues
    try:
        pd.read_pickle('all_leagues_{}.p'.format(league_name))
    except Exception:
        query = "select * from League"
        all_leagues = pd.read_sql(query, con=con)
        all_leagues.to_pickle('all_leagues_{}.p'.format(league_name))
    # get the matches
    try:
        pd.read_pickle('all_matches_{}.p'.format(league_name))
    except Exception:
        query = "Select * from Match"
        all_matches = pd.read_sql(query, con=con)
        all_matches.to_pickle('all_matches_{}.p'.format(league_name))
    # get the teams
    try:
        pd.read_pickle('all_teams_{}.p'.format(league_name))
    except Exception:
        query = "Select * from Team"
        all_teams = pd.read_sql(query, con=con)
        all_teams.to_pickle('all_teams_{}.p'.format(league_name))
    # get the team attributes
    try:
        pd.read_pickle('all_team_attributes_{}.p'.format(league_name))
    except Exception:
        query = "Select * from Team_Attributes"
        all_team_attributes = pd.read_sql(query, con=con)
        all_team_attributes.to_pickle('all_team_attributes_{}.p'.format(league_name))
    #print('... completing the data import')

def preprocess_matches_for_season(seasons, compute_form = False,
            window=3, exclude_firstn=True, home_advantage=None,league_name="England"):
    '''
    do all the preprocessing and return a matches dataframe ready
    for learning
    '''
    matches = get_all_seasons_data(seasons,league_name=league_name) #matches, team_attributes)

    if home_advantage:
        if home_advantage == 'points':
            matches = compute_point_based_home_advantage(matches)
        if home_advantage == 'goals':
            matches = compute_goal_based_home_advantage(matches)
        if home_advantage == 'both':
            matches = compute_point_based_home_advantage(matches,
                            column='home_advantage_points')
            matches = compute_goal_based_home_advantage(matches,
                            column='home_advantage_goals')

    if compute_form:
        matches = merge_matches_with_form(matches=matches,
                        seasons=seasons,window=window,league_name=league_name)
    return matches

def get_all_matches(league_name="England"):
    '''
    return the dataframe of leagues
    '''
    try:
        matches = pd.read_pickle('all_matches_{}.p'.format(league_name))
    except Exception as e:
        import_datasets()
        matches = pd.read_pickle('all_matches_{}.p'.format(league_name))

    #print("Matches Info:")
    #print(matches.info())
    #print(matches.head())
    return matches

def get_all_leagues(league_name="England"):
    '''
    return the dataframe of leagues
    '''
    try:
        leagues = pd.read_pickle('all_leagues_{}.p'.format(league_name))
    except Exception as e:
        import_datasets()
        leagues = pd.read_pickle('all_leagues_{}.p'.format(league_name))
    return leagues

def get_all_teams(league_name='England'):
    '''
    return the dataframe of teams
    '''
    try:
        teams = pd.read_pickle('all_teams_{}.p'.format(league_name))
        teams.shape
    except Exception as e:
        import_datasets()
        teams = pd.read_pickle('all_teams_{}.p'.format(league_name))
    #print("Team Info:")
    #print(teams.info())
    #print(teams.head())
    return teams

def get_all_team_attributes(league_name='England'):
    '''
    return the dataframe of team_attributes
    '''
    try:
        team_attributes = pd.read_pickle('all_team_attributes_{}.p'.format(league_name))
    except Exception as e:
        import_datasets()
        team_attributes = pd.read_pickle('all_team_attributes_{}.p'.format(league_name))
    #print("Team Attributes Info:")
    #print(team_attributes.info())
    #print(team_attributes.head())    
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
        Get matches for each season and league
    '''

    matches = get_all_matches()
    if league_name is not None:
        league_id = get_league_id(league_name=league_name)
        matches = matches[matches['league_id']==league_id]

    if season is not None:
        matches = matches[matches['season'].str.contains(season)]

    return matches

def get_matches_for_seasons(seasons,league_name="England"):
    '''
    Get matches for multiple seasons
    '''
    if seasons is None:
        #print("shouldnt be here")
        return get_matches_for_season(season = None,league_name=league_name)
    start_season = seasons[0]
    matches = get_matches_for_season(season=start_season,league_name=league_name)

    for s in range(1,len(seasons)):
        matches = matches.append(get_matches_for_season(season=seasons[s],
                                league_name=league_name), ignore_index=True)

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

def merge_matches_with_form(matches, seasons=None, window=3,
                            exclude_firstn=True,league_name="England"):
    '''
        Generate the matches with form dataframe
    '''
    avg_cols =['match_id','home_team_win_average','home_team_draw_average',
               'away_team_win_average','away_team_draw_average',
               'home_team_lose_average', 'away_team_lose_average']

    # get the matches with form
    pickname = 'matches_with_form_{}_{}.p'.format(window,league_name)
    try:

        mwf = pd.read_pickle(pickname)
    except Exception as e:
        mwf = get_all_seasons_data(None,league_name=league_name)
        mwf = compute_all_forms(mwf,window=window,league_name=league_name)
        mwf.to_pickle(pickname)

    # filter the season ...
    if seasons:
        mwf = mwf[mwf['season'].isin(seasons)]
    # then only filter the specified columns
    mwf = mwf[avg_cols]

    #idxs = matches[matches['match_id'].isin(matches_away['match_id'].values)].index
    matches = pd.merge(left=matches, right=mwf, how='left',
                        left_on='match_id', right_on='match_id')

    matches.sort_values(by=['match_date'],axis=0,inplace=True)
    cols = ['match_id', 'match_date','season','stage',
                'home_team_api_id'] #,'away_team_api_id']

    # if option is true then exclude the first N=window games for each team
    if exclude_firstn:
        # get the season's games
        for s in matches['season'].unique():
            #print("Shape before season {}, window {} drops: {}".format(s,
            #                        window, matches.shape))
            teams = matches['home_team_api_id'].unique()
            # loop over each team
            for t in teams:
                # get the teams matches
                match_slice= matches[(matches['home_team_api_id'] == t) |
                                     (matches['away_team_api_id'] == t)]
                match_slice = match_slice[match_slice['season']==s]

                matches.drop(match_slice.index[:window],inplace=True,axis=0)
            #print("Shape after season {}, window {} drops: {}".format(s,
            #                        window, matches.shape))
                #print(matches[matches['season']==s][cols].head(window*18))
        #assert(-1==1)
    return matches


def matches_home_away_diff(matches):
    '''
    Return dataset by taking a difference between home and away features
    corresponding
    '''
    cols = matches.columns
    home_cols = [x for x in matches.columns if 'home_' in x]
    away_cols = [x for x in matches.columns if 'away_' in x]
    #print(home_cols)
    cols = [x.replace('home_','') for x in home_cols]
    for c in cols:
        matches[c] = matches['home_'+c] - matches['away_'+c]
    #print("Shape of matches after new columns: {}".format(matches.shape))
    matches.drop(home_cols, axis=1, inplace=True)
    #print("Shape of matches after home drops: {}".format(matches.shape))
    matches.drop(away_cols,axis=1, inplace=True)
    #print("Shape of matches after away drops: {}".format(matches.shape))
    #print(matches.columns.T)
    #assert(1==-1)
    return matches

def clean_up_matches(matches, ignore_columns=None):
    '''
    clean up matches dataframe by removing nulls
    also drop some additioonal columns: either ids or
    '''
    matches.index = matches['match_id']
    # then drop the match_id and also drop stage for now
    to_drop = [ 'match_id','stage', 'match_date','home_team_api_id',
            'away_team_api_id','home_team', 'away_team','season',
            'home_buildUpPlayDribbling','away_buildUpPlayDribbling']
    if ignore_columns is not None:
        to_drop = [x for x in to_drop if x not in ignore_columns]
    #'home_team_goal', 'away_team_goal',
    # make a copy of the matches dataframe and drop the appropriate fields while deleting the unneeded features
    matches = matches.drop(to_drop, axis =1)
    #print("Matches shape after clean up {}".format(matches.shape))

    return matches

def encode_matches(matches, ignore_columns=None):
    '''
    encode category columns using the dummies to create a column per option
    '''
    # get categorical data ...
    cat_list= matches.select_dtypes(include=['object']).columns.tolist()
    if ignore_columns is not None:
        cat_list = [x for x in cat_list if x not in ignore_columns]
    #print("cat list: {}".format(cat_list))
    # then encode those columns ...
    matches = pd.get_dummies(matches, prefix=cat_list, columns=cat_list,drop_first=True)
    #print("Matches shape after encode up {}".format(matches.shape))
    return matches

def get_all_seasons_data(seasons,league_name="England"): #matches,tattr):
    '''
    get the number of unique seasons in the matches dataframe
    '''
    matches = get_matches_for_seasons(seasons=seasons,league_name=league_name)
    matches = matches[matches.columns[:11]]
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
    
    #print("Matches shape after clean up {}".format(newmatches.shape))

    return  newmatches #



def compute_point_based_home_advantage(matches,column='home_advantage'):
    unique_teams = matches['home_team_api_id'].unique()
    # get teams
    teams = get_all_teams()
    seasons = matches['season'].unique()

    matches['home_team_points'] = 3*(matches['home_team_goal'] > matches['away_team_goal']) + \
                1*(matches['home_team_goal'] == matches['away_team_goal'])
    matches['away_team_points'] = 3*(matches['home_team_goal'] < matches['away_team_goal']) + \
                1*(matches['home_team_goal'] == matches['away_team_goal'])

    matches[column] = None
    matches['home_points_cumsum'] = 0
    matches['away_points_cumsum'] = 0
    for team in unique_teams:

        team_name = teams[teams['team_api_id']== team]['team_long_name']
        #print("Team: {}".format(team_name.values[0]))
        matches.sort_values(by=['match_date'],axis=0,inplace=True)

        for season in seasons:
            # home matches
            cumsumh = matches.query('(season == @season) and (home_team_api_id == @team)')
            matches.loc[cumsumh.index,'home_points_cumsum'] =  cumsumh['home_team_points'].cumsum()
            # away matches
            cumsuma = matches.query('(season == @season) and (away_team_api_id == @team)')
            matches.loc[cumsuma.index,'away_points_cumsum'] =  cumsuma['away_team_points'].cumsum()
            # matches['home_points_cumsum'] = \
            #         matches.ix[(matches['season'] == season)]
            #[matches['home_team_api_id'] == team]['home_team_points'].cumsum()

            qry = '(season == @season) and ((home_team_api_id == @team) | (away_team_api_id == @team))'
            cumsumb = matches.query(qry)
            #print(cumsumb.index)
            for i in range(1,len(cumsumb.index)):
                # print(cumsumb.index[0:i])
                hmax = matches.loc[cumsumb.index[0:i],'home_points_cumsum'].max()
                amax = matches.loc[cumsumb.index[0:i],'away_points_cumsum'].max()
                # print("i, amax, hmax:[{},{},{}]".format(i,amax, hmax))
                # print(matches.loc[cumsumb.index[0:i],
                #         ['home_points_cumsum','away_points_cumsum','home_advantage']].head(i+1))
                #print()
                if amax != 0:
                    matches.loc[cumsumb.index[i],'advantage_home'] = hmax / (amax *1.)
                else:
                    matches.loc[cumsumb.index[i],'advantage_home'] = 0.0

    matches.drop(['home_points_cumsum','away_points_cumsum', 'home_team_points',
                  'away_team_points'], axis=1,inplace=True)
    
    matches['advantage_home'] = pd.to_numeric(matches['advantage_home'],errors='coerce')
    return matches

def compute_goal_based_home_advantage(matches,column='home_advantage'):
    unique_teams = matches['home_team_api_id'].unique()
    # get teams
    teams = get_all_teams()
    seasons = matches['season'].unique()

    matches[column] = None
    matches['home_goals_cumsum'] = 0
    matches['away_goals_cumsum'] = 0
    for team in unique_teams:

        team_name = teams[teams['team_api_id']== team]['team_long_name']
        #print("Team: {}".format(team_name.values[0]))
        matches.sort_values(by=['match_date'],axis=0,inplace=True)

        for season in seasons:
            # home matches
            cumsumh = matches.query('(season == @season) and (home_team_api_id == @team)')
            matches.loc[cumsumh.index,'home_goals_cumsum'] =  cumsumh['home_team_goal'].cumsum()
            # away matches
            cumsuma = matches.query('(season == @season) and (away_team_api_id == @team)')
            matches.loc[cumsuma.index,'away_goals_cumsum'] =  cumsuma['away_team_goal'].cumsum()

            qry = '(season == @season) and ((home_team_api_id == @team) | (away_team_api_id == @team))'
            cumsumb = matches.query(qry)

            for i in range(1,len(cumsumb.index)):
                # print(cumsumb.index[0:i])
                home_mean = matches.loc[cumsumb.index[0:i],'home_goals_cumsum'].max()
                away_mean = matches.loc[cumsumb.index[0:i],'away_goals_cumsum'].max()

                # get count of home game so far ...
                num_home_games = cumsumb.query('home_team_api_id == @team').shape[0]
                num_away_games = cumsumb.query('away_team_api_id == @team').shape[0]
                
                if (num_home_games != 0) &  (num_away_games != 0):
                    home_mean = home_mean / (1. * num_home_games)
                    away_mean = away_mean / (1. * num_away_games)
                    matches.loc[cumsumb.index[i],'advantage_home'] = home_mean - away_mean 

    matches.drop(['home_goals_cumsum','away_goals_cumsum'], axis=1,inplace=True)

    matches['advantage_home'] = pd.to_numeric(matches['advantage_home'],errors='coerce')

    return matches

def compute_all_forms(matches,window=3,league_name='England'):
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
                'away_team_win_average', 'away_team_draw_average',
              'home_team_lose_average', 'away_team_lose_average']

    matches['home_team_won'] = 1*(matches['home_team_goal'] > matches['away_team_goal'])
    matches['home_team_drew'] = 1*(matches['home_team_goal'] == matches['away_team_goal'])
    matches['home_team_lost'] = 1*(matches['home_team_goal'] < matches['away_team_goal'])
    matches['away_team_won'] = 1*(matches['home_team_goal'] < matches['away_team_goal'])
    matches['away_team_lost'] = 1*(matches['home_team_goal'] > matches['away_team_goal'])
    matches['away_team_drew'] = 1*(matches['home_team_goal'] == matches['away_team_goal'])

    in_cols = ['home_team_won','home_team_drew','home_team_lost',
               'away_team_won','away_team_drew','away_team_lost']
    all_cols = in_cols
    all_cols.extend(['home_team_goal','away_team_goal','match_id','match_date'])
    #this_team_win_average','this_team_lose_average','this_team_draw_average','
    roll_cols = ['match_id','this_team_win_average', 'this_team_lose_average',
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
        team_name = teams[teams['team_api_id']== t]['team_long_name']
        #print("Team: {}".format(team_name.values[0]))
        # get the home teams for this team
        #dm = matches[(matches['home_team_api_id'] == t) |
    #                (matches['away_team_api_id'] == t)]
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
        #### sort
        matches_t.sort_values(by=['match_date'],axis=0,inplace=True)
        #print("matches_t size {}".format(matches_t.shape))

        # compute the rolling statistics
        matches_t.loc[:,'this_team_win_average'] = \
                matches_t['this_team_won'].rolling(window).sum() / (1.0*window)
        matches_t.loc[:,'this_team_draw_average'] = \
                matches_t['this_team_drew'].rolling(window).sum() / (1.0*window)
        matches_t.loc[:,'this_team_lose_average'] = \
                matches_t['this_team_lost'].rolling(window).sum() / (1.0*window)
        # fill in Nans for the first window - 1 matches
        matches_t.fillna(0, inplace=True)
        for rc in roll_cols:
            if rc == 'match_id':
                continue
            matches_t[rc] = matches_t[rc].shift(1)
        #print("matches_t size {}".format(matches_t.shape))

        # then reassign the computed average
        matches_t=matches_t[roll_cols]
        matches_home = pd.merge(left=matches_home, right=matches_t, how="left",
                            left_on = 'match_id', right_on='match_id')
        #print("matches_home size {}".format(matches_home.shape))

        matches_away = pd.merge(left=matches_away, right=matches_t, how="left",
                            left_on = 'match_id', right_on='match_id')

        matches_away = matches_away[roll_cols]
        #print("matches_away size {}".format(matches_away.shape))
        matches_home = matches_home[roll_cols]
        matches_away.fillna(0, inplace=True)
        matches_home.fillna(0, inplace=True)
        #print(matches_away.tail())
        #assert(-1==1)

        #print("matches size {}".format(matches.shape))
        matches = pd.merge(left=matches, right=matches_home, how='left',
                            left_on='match_id', right_on='match_id')
        idxs = matches[matches['match_id'].isin(matches_home['match_id'].values)].index
        matches.loc[idxs,'home_team_win_average'] = matches['this_team_win_average_y']
        matches.loc[idxs,'home_team_draw_average'] = matches['this_team_draw_average_y']
        matches.loc[idxs,'home_team_lose_average'] = matches['this_team_lose_average_y']
        #print("matches size {}".format(matches.shape))
        matches.drop(['this_team_win_average_y','this_team_draw_average_y',
                      'this_team_lose_average_y'],axis=1,inplace=True)
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

    matches.sort_values(by=['match_date'],axis=0,inplace=True)
    matches.to_csv("matches_with_form_{}_{}.csv".format(window,league_name), encoding='utf-8')
    return matches


def subsample_matches(matches):
    '''
     Attempt to sub sample matches dataset to generate equal distributions of
     each class
    '''
    print('### Subsampling Data ####')
    draws = matches[matches['home_team_outcome'] == 'draw']
    wins = matches[matches['home_team_outcome'] == 'win']
    losses = matches[matches['home_team_outcome'] == 'lose']
    print("Losses:{}, draws:{}, wins:{}".format(len(losses), len(draws), len(wins)))

    # subsample losses
    are_draws_small = True if len(draws) < len(losses) else False

    if are_draws_small:
        percentage = len(draws)/float(len(losses))
        losses_sampled =  losses.sample(frac = percentage, random_state = 2)
        percentage = len(draws)/float(len(wins))
        wins_sampled = wins.sample(frac = percentage, random_state = 2)
        matches_sampled = draws.append(wins_sampled)
        matches_sampled = matches_sampled.append(losses_sampled)

        # print stats
        print("Percentage losses	:", len(losses_sampled)/float(len(matches_sampled)))
        print("Percentage draws		:", len(draws)/float(len(matches_sampled)))
    else:
        percentage = len(losses)/float(len(draws))
        draws_sampled =  draws.sample(frac = percentage, random_state = 2)
        percentage = len(losses)/float(len(wins))
        wins_sampled = wins.sample(frac = percentage, random_state = 2)
        matches_sampled = losses.append(wins_sampled)
        matches_sampled = matches_sampled.append(draws_sampled)

        #print stats
        print("Percentage draws	:", len(draws_sampled)/float(len(matches_sampled)))
        print("Percentage losses		:", len(losses)/float(len(matches_sampled)))

    print("Percentage wins		:", len(wins_sampled)/float(len(matches_sampled)))
    print("Total matches		:", len(matches_sampled))

    return matches_sampled


def pca_results(data, pca, columns):
    '''
    Create a DataFrame of the PCA results
    Includes dimension feature weights and explained variance
    Visualizes the PCA results
    '''
    # Dimension indexing
    dimensions =  ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]
    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns = columns) # data.keys())
    components.index = dimensions
    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
    variance_ratios.index = dimensions
    # Create a bar plot visualization
    fig, ax = plt.subplots(figsize = (14,8))
    # Plot the feature weights as a function of the components
    components.plot(ax = ax, kind = 'bar');
    #print("Components:")
    #pprint.pprint(components.T)
    ax.set_ylabel("Feature Weights")
    ax.set_xticklabels(dimensions, rotation=0)
    # Display the explained variance ratios
    for i, ev in enumerate(pca.explained_variance_ratio_):
        ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n%.4f"%(ev))
        #pprint.pprint("{} {}".format(i,ev))
        
    
    plt.show()
    # Return a concatenated DataFrame
    return pd.concat([variance_ratios, components], axis = 1)

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

