'''
###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
#import warnings
#warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
###########################################
'''


#import matplotlib.pyplot as plt
import sqlite3 as sql
import numpy as np
import pandas as pd


def get_season_as_date(season):
    '''
    Returns a formatted date from a season
    '''
    return "{}-07-01 00:00:00".format(season)

def get_matches_for_seasons(seasons):
    '''
    Get matches for multiple seasons
    '''
    start_season = seasons[0]
    matches = get_matches_for_season(start_season)
    for s in xrange(1,len(seasons)):
        matches = matches.append(get_matches_for_season(seasons[s]), ignore_index=True)
        #print(matches.shape)
    return matches

def get_attributes_for_seasons(seasons):
    '''
    Get matches for multiple seasons
    '''
    start_season = seasons[0]
    attrs = get_attributes_for_season(start_season)
    for s in xrange(1,len(seasons)):
        attrs = attrs.append(get_attributes_for_season(seasons[s]), ignore_index=True)
        #print(attrs.shape)
    return attrs

def get_matches_for_season(season=None):
    '''
        Get matches for each season
    '''
    con = None
    con = sql.connect('../input/database.sqlite')
    #cur = con.cursor()
    query = "select * from League where name like '%England%'"
    eplinfo = pd.read_sql(query, con=con)

    if season is not None:
        query = "Select * from Match \
                where league_id = {} and season='{}'".format(eplinfo['id'][0], season)
    else:
        query = "Select * from Match \
                where league_id = {} ".format(eplinfo['id'][0])
    matches = pd.read_sql(query, con=con)

    return matches

def get_attributes_for_season(season):
    '''
        Get team attributes data from the database
    '''
    con = None
    con = sql.connect('../input/database.sqlite')
    #cur = con.cursor()

    [sstart, ssend] = season.split('/')

    query = "SELECT * FROM Team_Attributes where date >= '{}' and date <='{}'".format(
        get_season_as_date(sstart), get_season_as_date(ssend))
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
    #print(matches.shape)

    matches.rename(columns={'id_x':'match_id','date':'match_date','team_long_name':'home_team'}, 
                inplace=True)
    matches = pd.merge(left=matches, right=teams, how='left', left_on='away_team_api_id', 
                right_on='team_api_id')
    matches = matches.drop(['id', 'match_api_id', 'team_fifa_api_id', 'team_short_name'], axis=1)

    matches.rename(columns={'team_long_name':'away_team'}, inplace=True)
    matches.head()
    #print(matches.shape)

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

    matches["home_team_points"] = 3*(matches["home_team_goal"] > matches["away_team_goal"]) + 1*(matches["home_team_goal"] == matches["away_team_goal"])
    #matches["home_team_points"] = 1*(matches["home_team_goal"] == matches["away_team_goal"])
    #print(matches.shape)
    #matches.head(10)

    return matches

def clean_up_matches(matches):
    '''
    clean up matches dataframe by removing nulls
    '''
    matches.index = matches['match_id']
    # then drop the match_id and also drop stage for now
    to_drop = ['match_id', 'stage',  'match_date','home_team_api_id',
            'away_team_api_id','home_team', 'away_team','season',
            'home_buildUpPlayDribbling','away_buildUpPlayDribbling']  
    #'home_team_goal', 'away_team_goal',
    # make a copy of the matches dataframe and drop the appropriate fields while deleting the unneeded features
    matches = matches.drop(to_drop, axis =1)
    print(matches.shape)

    return matches

def encode_matches(matches):
    '''
    encode category columns using the dummies to create a column per option
    '''
    cat_list= matches.select_dtypes(include=['object']).columns.tolist()
    matches = pd.get_dummies(matches, prefix=cat_list)

    return matches
    
def get_all_seasons_data(matches,tattr):
    '''
    get the number of unique seasons in the matches dataframe
    '''
    seasons = matches['season'].unique()
    #if season not in seasons:
    #    return None
    newmatches = pd.DataFrame()
    
    for e in seasons:
        [sstart, ssend] = e.split('/')
        # get the corresponding matches
        print e
        m = matches[matches['season'] == e]
        #print(m.shape)

        # get the corresponding attributes
        sstartdt = get_season_as_date(sstart)
        ssenddt = get_season_as_date(ssend)
        t = tattr[(tattr['date'] >= get_season_as_date(sstart)) & 
                (tattr['date'] < get_season_as_date(ssend))]

        # do the on the home team
        m = pd.merge(left=m, right=t, how='left', left_on='home_team_api_id',right_on='team_api_id')
        #print m.info()
        dropcols = ['country_id','league_id','id_y','team_api_id','team_fifa_api_id','team_short_name']
        dropcols = [c for c in dropcols if c in m.index]
        m = m.drop(dropcols, axis=1)
        m.rename(columns={'id_x':'match_id','date':'match_date', 'team_long_name':'home_team'}, 
                        inplace=True)

        # then merge again on the away team
        m = pd.merge(left=m, right=t, how='left', left_on='away_team_api_id', right_on='team_api_id')
        dropcols = ['id', 'match_api_id', 'team_fifa_api_id', 'team_short_name']
        dropcols = [c for c in dropcols if c in m.index]
        m = m.drop(dropcols, axis=1)
        m.rename(columns={'team_long_name':'away_team'}, inplace=True)

        # then append to newmatches
        #if newmatches.shape[0] == 0:
        #    newmatches = m.copy()
        #    #print newmatches.shape
        #else:
        newmatches = newmatches.append(m, ignore_index=False)
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
