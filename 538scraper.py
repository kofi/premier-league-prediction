'''
    Quick script to extract the results of the 2016/2017 EPL 
    predictions from the fivethirtyeight website.
    datasource:     https://projects.fivethirtyeight.com/soccer-predictions/

    - Match information is a table wrapped in a .match-container css div class
    - .match-top and .match-bottom css table row classes contain the two teams
        - td.date has the date
        - td.team has the team name in the "data-str" attribute
            - sub .team-div div class has two spans
                - .name for the team name
                - .score for the goals scored by the corresponding team
        - td.prob has the win probability
        - td.tie-prob has the tie probability (only in the .match-top)
        - 

'''
import os
from bs4 import BeautifulSoup
import requests
#import csv
import pandas as pd

predicts = 'predictions.html'
csv_file = 'epl_2017_predictions.csv'

with open(predicts) as fp:
    soup = BeautifulSoup(fp,"lxml")

#print(soup.prettify())
# Get all matches using the match-container div class
# total number should be 380 for the season
all_matches = soup.find_all("div", class_="match-container")
print("Total number of matches: {}".format(len(all_matches)))


# match counter
cnt = 0

# create the header
csv_string = ("Match,Date,HomeTeam,HomeGoals,HomeWinProb,TieProb,AwayTeam,AwayGoals,AwayProb\n")

# loop over the matches
for match in all_matches:
    # make sure we have content in the tag ...
    if match.contents[0] is None:
        continue

    #match_info  = {}
    # get the first/home team
    top_match = match.find_all("tr", class_="match-top")

    # skip if we don't have a top match
    if top_match is None:
        continue

    top_match = top_match[0]
    t_date = top_match.select_one(".date").string
    t_name = top_match.select_one(".team").select_one(".team-div").select_one(".name").string
    t_score = top_match.select_one(".team").select_one(".team-div").select_one(".score").string
    t_prob = top_match.select_one(".prob").string
    t_tie_prob = top_match.select_one(".tie-prob").string

    match_info = {'date':t_date, 'home_team':t_name,
                    'home_goals':t_score, 'home_win_prob': t_prob,
                    'tie_prob':t_tie_prob}



    
    # find the away team info
    bottom_match = match.find_all("tr", class_="match-bottom")
    # skip if we don't have a top match
    if bottom_match is None:
        continue

    bottom_match = bottom_match[0]
    t_name = bottom_match.select_one(".team").select_one(".team-div").select_one(".name").string
    t_score = bottom_match.select_one(".team").select_one(".team-div").select_one(".score").string
    t_prob = bottom_match.select_one(".prob").string

    # update the dictionary
    cnt = cnt + 1
    #match_info = match_info.update({'cnt':cnt, 'away_team':t_name, 'away_goals':t_score,
    #                                'away_win_prob': t_prob})
    
    match_info['cnt'] = cnt
    match_info['away_team'] = t_name
    match_info['away_goals'] = t_score
    match_info['away_win_prob'] = t_prob

    # then append the new row 
    csv_string = csv_string + ('{cnt},{date},{home_team},{home_goals},{home_win_prob}'
            ',{tie_prob},{away_team},{away_goals},{away_win_prob}\n'.format(**match_info))


# make sure to delete the file if it exists
try:
    os.remove(csv_file)
except OSError:
    pass

# write out the file into a csv file
with open(csv_file, 'w', encoding='utf-8') as c:
    # set up the csv
    #c = csv.writer(csvfile,  quoting=csv.QUOTE_MINIMAL)        
    #    c.writerows(csv_string) #.decode('utf-8').split(","))
    c.write(csv_string)


# quick check to make sure the data can be imported as a csv
data = pd.read_csv(csv_file)

data.info()

print("Head")
print(data.head().T)
print("Tail")
print(data.tail().T)
print("Fin!")

