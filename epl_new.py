
# coding: utf-8

# # Exploring how team attributes impact scores in the premier league
# ## Merge the match and (winning) team data to evaluate which team features most impact 

# In[1]:

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import sys
#from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton
#from PyQt5.QtGui import QIcon
import itertools
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sqlite3 as sql
import matplotlib
matplotlib.use('TkAgg') #"Qt5Agg")
import matplotlib.pyplot as plt


#******************************************************************
#from PyQt5 import QtCore
import seaborn as sbn 
sbn.set()
import helpers as h
import sys

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:

# create the connection to the database
con = None
con = sql.connect('../input/database.sqlite')
# create the cursor
cur= con.cursor()
# select only the information for the EPL
#   - changed from sqlite interface to use pd.read_sql
query = "select * from League where name like '%England%'"
eplinfo = pd.read_sql(query,con=con) 
# get the leagues
query = "select * from League"
leagues = pd.read_sql(query,con=con)
my_team = 'Liverpool'
my_team_info = h.get_team_id(my_team)
#print(my_team_info['team_api_id'])
my_team_id = my_team_info['team_api_id'][0]
#print(my_team_id)


# ### Get match information 
# filter on epl and get the actual team names
# also select date, season, date and time, home team goal, away team goals, home team wins?

# In[3]:

season = None #"2010/2011" #,"2015/201"]
matches = h.preprocess_matches_for_season(season)
# filter out only the matches with the team of interest
#matches = matches[(matches['home_team_api_id'] == my_team_id)  | (matches['away_team_api_id'] == my_team_id)]
# set the home status of the team of interest
#matches.loc[matches['home_team_api_id'] == my_team_id,'isteamhome'] = 1
#matches.loc[matches['home_team_api_id'] != my_team_id,'isteamhome'] = 0

matches = h.clean_up_matches(matches)
matches = h.encode_matches(matches)
matches.describe()
print(matches.columns.T)
#print(matches.shape)




# In[4]:

# create the output columns
matches['home_team_points'] = 3*(matches['home_team_goal'] > matches['away_team_goal']) + \
            1*(matches['home_team_goal'] == matches['away_team_goal'])

# define the output classes
output_class = np.array(['draw','lose','win'])
matches['home_team_outcome'] = 'draw'
matches.loc[matches['home_team_goal'] > matches['away_team_goal'],['home_team_outcome']] = 'win'
matches.loc[matches['home_team_goal'] < matches['away_team_goal'],['home_team_outcome']] = 'lose'
#matches.info()


# In[5]:

matches.head()
cols = ['home_team_goal','away_team_goal', 'home_team_outcome','home_team_points'] #,'isteamhome']
matches[cols].tail()


# In[6]:

# get some statistics
# home team win percentages
percent_home_win = np.sum(matches['home_team_points'] == 3)/(1. * np.max(matches.shape[0]))
percent_home_loss = np.sum(matches['home_team_points'] == 0)/(1. * np.max(matches.shape[0]))
percent_home_draw = np.sum(matches['home_team_points'] == 1)/(1. * np.max(matches.shape[0]))
print("Home team win percentage: {}".format(percent_home_win))
print("Home team loss percentage: {}".format(percent_home_loss))
print("Home team draw percentage: {}".format(percent_home_draw))
print()

# In[7]:

# Predicted home team goals analysis
#y = np.array(matches[ 'home_team_outcome']) #
#matches_hm_goals = matches.drop(['home_team_points','home_team_goal',
#                                    'away_team_goal','home_team_outcome'], axis=1)

# In[8]:

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# drop Nan rows
allnas = matches.isnull().any()
if (sum(allnas == True)):
    matches.dropna(inplace=True)

# define the output variable
y = np.array(matches['home_team_outcome'])

print("Unique Y ", matches['home_team_outcome'].unique())

# then delete columns
matches_hm_goals = matches.drop(['home_team_points','home_team_goal',
                                    'away_team_goal','home_team_outcome'], axis=1)
# finally transform the data and scale to normalize
X = np.array(scaler.fit_transform(matches_hm_goals)) 
print("shape of X: {}".format(X.shape))


# lets try with PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=28)
pca = pca.fit(X)
X = pca.transform(X)

#print("Percent explain variance")
#print(100*pca.explained_variance_ratio_)


#print(y.shape)

## algorithim by algorithm analysis
#from sklearn.svm import LinearSVR
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as kNN

from sklearn.metrics.pairwise import chi2_kernel
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

# first split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=14)

# get the baseline error/accuracy using a dummy classifier
clf = DummyClassifier(strategy='most_frequent',random_state=0)
clf.fit(X_train,y_train)
#print("Dummy Classifier Training score: {}".format(clf.score(X_train,y_train)))
#print("Dummy Classifier Test score: {}".format(clf.score(X_test,y_test)))
print("Dummy Classifier F1 score: {}".format( f1_score(y_test, clf.predict(X_test),average='weighted'))) 
print()
# then train
clf = SVC(kernel='linear')
clf.fit(X_train,y_train)

# ... get the training score
print("Linear SVC Training score: {}".format(clf.score(X_train,y_train)))
print("Linear SVC Training F1 score: {}".format( f1_score(y_train, clf.predict(X_train),average='weighted'))) #accuracy_score(y_train, clf.predict(X_train))))
# get the test error/score
print("Linear SVC Test score: {}".format(clf.score(X_test,y_test)))
print("Linear SVC Test F1 score: {}".format( f1_score(y_test, clf.predict(X_test), average='weighted')))
print()


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train,y_train)

y_test_pred = clf.predict(X_test)
print("DecisionTree F1 score: {}".format( f1_score(y_test, y_test_pred,average='weighted')))
print()


# then change our accuracy to an rbf kernel
clf = SVC(kernel='rbf')
clf.fit(X_train,y_train)
print("RBF SVC Training score: {}".format(clf.score(X_train,y_train)))
print("RBF SVC Training F1 score: {}".format( f1_score(y_train, clf.predict(X_train),average='weighted')))
print("RBF SVC Test score: {}".format(clf.score(X_test,y_test)))
y_test_pred = clf.predict(X_test)
print("RBF SVC Test F1 score: {}".format( f1_score(y_test, y_test_pred,average='weighted')))
print()

# get the confusion matrix and plot for the RBF
cnf_matrix = confusion_matrix(y_test, y_test_pred) #, labels=output_class)
print(np.sum(np.sum(cnf_matrix)))
#y_test

#print("Verification of Confusion matrix")
#for i in output_class:
#    for j in output_class:
#        matrix_val = np.dot((y_test==i)*1.,(y_test_pred == j)*1)
#        print("{} but predicts {}:  {}".format(i,j,matrix_val))

print()
print(cnf_matrix)

debug =False
if debug:
     np.set_printoptions(precision=2)
     plt.figure()
     h.plot_confusion_matrix(cnf_matrix, classes=output_class,
                      title='Confusion matrix, without normalization')

     plt.figure()
     h.plot_confusion_matrix(cnf_matrix, classes=output_class, normalize=True,
                      title='Normalized Confusion matrix')

     plt.show() 

#### Problem seems to be the model is overpredicting one classifier.
#### Could be a problem due to the unbalanced data set 
#### need to come up with an approach to oversample the smaller categories
#### and undersample the most frequent class
draws = matches[matches['home_team_outcome'] == 'draw']
wins = matches[matches['home_team_outcome'] == 'win']
losses = matches[matches['home_team_outcome'] == 'lose']

# subsample losses
percentage = len(draws)/float(len(losses))
losses_sampled =  losses.sample(frac = percentage, random_state = 2)
percentage = len(draws)/float(len(wins))
wins_sampled = wins.sample(frac = percentage, random_state = 2)

matches_sampled = draws.append(wins_sampled)
matches_sampled = matches_sampled.append(losses_sampled)

print("Percentage wins		:", len(wins_sampled)/float(len(matches_sampled)))
print("Percentage losses	:", len(losses_sampled)/float(len(matches_sampled)))
print("Percentage draws		:", len(draws)/float(len(matches_sampled)))
print("Total matches		:", len(matches_sampled))


# define the output variable
y = np.array(matches_sampled['home_team_outcome'])

print("Unique Y ", matches_sampled['home_team_outcome'].unique())

# then delete columns
matches_hm_goals = matches_sampled.drop(['home_team_points','home_team_goal',
                                    'away_team_goal','home_team_outcome'], axis=1)
# finally transform the data and scale to normalize
X = np.array(scaler.fit_transform(matches_hm_goals))
print("shape of X: {}".format(X.shape))
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=14)

## lets try with PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=28)
pca = pca.fit(X)
X = pca.transform(X)

#print("Percent explain variance")
#print(100*pca.explained_variance_ratio_)



# get the baseline error/accuracy using a dummy classifier
clf = DummyClassifier(strategy='most_frequent',random_state=0)
clf.fit(X_train,y_train)
print("Dummy Classifier Training score: {}".format(clf.score(X_train,y_train)))
print("Dummy Classifier Test score: {}".format(clf.score(X_test,y_test)))
print("Dummy Classifier F1 score: {}".format( f1_score(y_test, clf.predict(X_test),average='weighted')))
print()


# then change our accuracy to an rbf kernel
clf = SVC(kernel='rbf')
clf.fit(X_train,y_train)
print("RBF SVC Training score: {}".format(clf.score(X_train,y_train)))
print("RBF SVC Training F1 score: {}".format( f1_score(y_train, clf.predict(X_train),average='weighted')))
print("RBF SVC Test score: {}".format(clf.score(X_test,y_test)))
y_test_pred = clf.predict(X_test)
print("RBF SVC Test F1 score: {}".format( f1_score(y_test, y_test_pred,average='weighted')))
print()


# get the confusion matrix and plot for the RBF
cnf_matrix = confusion_matrix(y_test, y_test_pred) #, labels=output_class)
print(np.sum(np.sum(cnf_matrix)))
#y_test

print("Verification of Confusion matrix")
#for i in output_class:
#    for j in output_class:
#        matrix_val = np.dot((y_test==i)*1.,(y_test_pred == j)*1)
#        print("{} but predicts {}:  {}".format(i,j,matrix_val))
 
print()
#print(cnf_matrix)
cm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
print()
#print(cm)

# # then a chi2_kernel
# k_chi = chi2_kernel(X_train)
# clf = SVC(kernel='precomputed').fit(K,y_train)
# #clf.fit(X_train,y_train)
# print("Chi2 SVC Training score: {}".format(clf.score(X_train,y_train)))
# print("Chi2 SVC Test score: {}".format(clf.score(X_test,y_test)))

# then change our accuracy to an rbf kernel
clf =  GNB()
clf.fit(X_train,y_train)
print("GaussianNB Training score: {}".format(clf.score(X_train,y_train)))
print("GaussianNB Training F1 score: {}".format( f1_score(y_train, clf.predict(X_train),average='weighted')))
print("GaussianNB Test score: {}".format(clf.score(X_test,y_test)))
y_test_pred = clf.predict(X_test)
print("GaussianNB Test F1 score: {}".format( f1_score(y_test, y_test_pred,average='weighted')))
print()


#http://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_iris.html
from sklearn import linear_model
clf = linear_model.SGDClassifier(alpha=0.001, n_iter=100)
clf.fit(X_train,y_train)
print("SGD Training score: {}".format(clf.score(X_train,y_train)))
print("SGD Training F1 score: {}".format( f1_score(y_train, clf.predict(X_train),average='weighted')))
print("SGD Test score: {}".format(clf.score(X_test,y_test)))
y_test_pred = clf.predict(X_test)
print("SGD Test F1 score: {}".format( f1_score(y_test, y_test_pred,average='weighted')))
print()



# get the confusion matrix and plot for the RBF
cnf_matrix = confusion_matrix(y_test, y_test_pred) #, labels=output_class)
#print(np.sum(np.sum(cnf_matrix)))
#y_test


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train,y_train)

y_test_pred = clf.predict(X_test)
print("DecisionTree F1 score: {}".format( f1_score(y_test, y_test_pred,average='weighted')))
print()



sys.exit()

# then tune the model using gridsearchcv
# basically give a set of model parameters and cross validate on those



# then get the best estimator

# then get the test error of the best estimator



# maybe generate a plot


# repeat the above by pipelining PCA before using the algorithm




# In[9]:

print("Feature space holds %d observations and %d features" % X.shape)
print("Unique target space:", y.shape)
print("Unique target labels:", np.unique(y))


# In[10]:

from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
predictors = []
#clfs = ['SVR','LinearSVR']  #,'Lasso','RF','KNN']
nfolds = 5
def plot_matrix(y_pred, y, clfname, ax):
    #print np.sum(y!=y_pred)/(1. * len(y))
    ax.imshow(confusion_matrix(y_pred, y),
           cmap='Blues', interpolation='nearest')
    #ax.ylabel('{} true'.format(clfname))
    #ax.xlabel('{} predicted'.format(clfname))

def run_cv(X,y,clf_class,**kwargs):
    # Construct a kfolds object
    kf = KFold(len(y),n_folds=nfolds,shuffle=True)
    y_pred = y.copy()
    # Iterate through folds
    for train_index, test_index in kf:
        #print("train ={} test = {}".format(train_index, test_index))
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[test_index] = clf.predict(X_test)
        predictors.append(clf)
        #plot_matrix(clf.predict(X),y, '')
    return y_pred

def accuracy(y_true,y_pred):
    # NumPy interpretes True and False as 1. and 0.
    #print np.array([y_true.T, y_pred.T])
    #print len(y_true)
    return np.sum(y_true == y_pred)/(1. * len(y_true))



# In[11]:



print(X.shape)

# lets try with PCA
#from sklearn.decomposition import PCA
#pca = PCA() #n_components=12)
#pca = pca.fit(X)
#X = pca.transform(X)

#print 100*pca.explained_variance_ratio_

#print("Linear SVR:")
#print("%.3f" % accuracy(y, run_cv(X,y,LinearSVR)))
print("SVM:")
print("%.3f" % accuracy(y, run_cv(X,y,SVC,kernel='rbf')))
print("GaussianNB")
print("%.3f" % accuracy(y, run_cv(X,y,GaussianNB)))
print("Logistic")
print("%.3f" % accuracy(y, run_cv(X,y,LR)))
print("kNN")
print("%.3f" % accuracy(y, run_cv(X,y,kNN)))


# In[ ]:




# In[12]:

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# In[ ]:




# In[ ]:




# In[ ]:

#seasons = ['2008/2009','2009/2010','2010/2011']
#get_matches_for_season


# Set up the query to join the match information with the home and away team IDS
#query = "Select match_api_id, m.home_team_api_id, m.away_team_api_id, \
#        m.stage, m.date, t1.team_long_name as home_team, \
#        t2.team_long_name as away_team, m.home_team_goal, m.away_team_goal \
#        from Match as m join Team as t1 join Team as t2 \
#        on t1.team_api_id = m.home_team_api_id and t2.team_api_id = m.away_team_api_id  \
#        where m.league_id = {} and m.season='{}'".format(eplinfo.id[0],season)
#print(query)
# get matches
#m.id, m.match_api_id, m.home_team_api_id, m.away_team_api_id, \
#        m.stage, m.date, m.home_team_goal, m.away_team_goal from Match as'
#query = "Select * from Match \
#        where league_id = {} and season='{}'".format(eplinfo.id[0],season)
#queryall = "Select * from Match where league_id = {}".format(eplinfo.id[0])
#matches = h.get_matches_for_season(seasons) #h.get_matches_for_season(season) #pd.read_sql(query,con=con)
#allcols = matches.columns.tolist()
#for e in xrange(len(allcols)):
#    print "{} - {}".format(e,allcols[e])

#matches[matches.columns[77:85]].head(20)

#matches = matches[matches.columns[:11]]
#pd.concat([matches[matches.columns[:11]],matches[matches.columns[77:85]]])
# get teams
#query = "SELECT * FROM Team"
#teams = pd.read_sql(query,con=con)

#query = "SELECT * FROM Team_Attributes" #where date >= '2010-07-01 00:00:00' and date <='2011-06-0100:00:00'"
#team_attributes = h.get_attributes_for_season(season) #  pd.read_sql(query,con=con)
#print(team_attributes[(team_attributes["date"] >= '2010-07-01 00:00:00') & (team_attributes["date"] <= '2011-06-0100:00:00')])
#print(team_attributes.shape)
#team_attributes.head()
#matts = h.get_all_seasons_data(matches, team_attributes)
#print(matts.shape)
#print
#print(matches['season'].unique())
#print matches

#matches = h.merge_matches_teams(matches, teams)
#matches = h.merge_matches_attributes(matches, team_attributes)

#print(matches.shape)
## pulled from Pete Hodges's kernel 'https://www.kaggle.com/petehodge/d/hugomathien/soccer/epl-weekly-predicting
#matches = pd.merge(left=matches, right=teams, how='left', left_on='home_team_api_id', right_on='team_api_id')
#matches = matches.drop(['country_id','league_id','id_y','team_api_id','team_fifa_api_id','team_short_name'], axis=1)
#print(matches.shape)

#matches.rename(columns={'id_x':'match_id','date':'match_date','team_long_name':'home_team'}, inplace=True)
#matches = pd.merge(left=matches, right=teams, how='left', left_on='away_team_api_id', right_on='team_api_id')
#matches = matches.drop(['id', 'match_api_id', 'team_fifa_api_id', 'team_short_name'], axis=1)

#matches.rename(columns={'team_long_name':'away_team'}, inplace=True)
#matches.head()
#print(matches.shape)



# # do the same for the team attributes for the home and away teams
# #matches = pd.merge(left=matches, right=team_attributes, how='left', left_on='home_team_api_id', right_on='team_api_id')
# #matches = matches.drop(['id', 'team_fifa_api_id', 'team_api_id_x','team_api_id_y','date'], axis=1)
# #matches.rename(columns={'buildUpPlaySpeed':'home_buildUpPlaySpeed','buildUpPlaySpeedClass':'home_buildUpPlaySpeedClass',
#                         'buildUpPlayDribbling':'home_buildUpPlayDribbling',
#                         'buildUpPlayDribblingClass':'home_buildUpPlayDribblingClass',
#                         'buildUpPlayPassing':'home_buildUpPlayPassing','buildUpPlayPassingClass':'home_buildUpPlayPassingClass',
#                         'buildUpPlayPositioningClass':'home_buildUpPlayPositioningClass',
#                         'chanceCreationPassing':'home_chanceCreationPassing','chanceCreationPassingClass':'home_chanceCreationPassingClass',
#                         'chanceCreationCrossing':'home_chanceCreationCrossing',
#                         'chanceCreationCrossingClass':'home_chanceCreationCrossingClass','chanceCreationShooting':'home_chanceCreationShooting',
#                         'chanceCreationShootingClass':'home_chanceCreationShootingClass','chanceCreationPositioningClass':'home_chanceCreationPositioningClass','defencePressure':'home_defencePressure',
#                         'defencePressureClass':'home_defencePressureClass','defenceAggression':'home_defenceAggression',
#                         'defenceAggressionClass':'home_defenceAggressionClass','defenceTeamWidth':'home_defenceTeamWidth',
#                         'defenceTeamWidthClass':'home_defenceTeamWidthClass','defenceDefenderLineClass':'home_defenceDefenderLineClass'}, inplace=True)

# matches = pd.merge(left=matches, right=team_attributes, how='left', left_on='away_team_api_id', right_on='team_api_id')
# matches = matches.drop(['id', 'team_fifa_api_id', 'team_api_id', 'date'], axis=1)

# matches.rename(columns={'buildUpPlaySpeed':'away_buildUpPlaySpeed','buildUpPlaySpeedClass':'away_buildUpPlaySpeedClass',
#                         'buildUpPlayDribbling':'away_buildUpPlayDribbling',
#                         'buildUpPlayDribblingClass':'away_buildUpPlayDribblingClass',
#                         'buildUpPlayPassing':'away_buildUpPlayPassing','buildUpPlayPassingClass':'away_buildUpPlayPassingClass',
#                         'buildUpPlayPositioningClass':'away_buildUpPlayPositioningClass',
#                         'chanceCreationPassing':'away_chanceCreationPassing','chanceCreationPassingClass':'away_chanceCreationPassingClass',
#                         'chanceCreationCrossing':'away_chanceCreationCrossing',
#                         'chanceCreationCrossingClass':'away_chanceCreationCrossingClass','chanceCreationShooting':'away_chanceCreationShooting',
#                         'chanceCreationShootingClass':'away_chanceCreationShootingClass','chanceCreationPositioningClass':'away_chanceCreationPositioningClass','defencePressure':'away_defencePressure',
#                         'defencePressureClass':'away_defencePressureClass','defenceAggression':'away_defenceAggression',
#                         'defenceAggressionClass':'away_defenceAggressionClass','defenceTeamWidth':'away_defenceTeamWidth',
#                         'defenceTeamWidthClass':'away_defenceTeamWidthClass','defenceDefenderLineClass':'away_defenceDefenderLineClass'}, inplace=True)
# print(matches.columns)


# matches["home_team_points"] = 3*(matches["home_team_goal"] > matches["away_team_goal"]) + 1*(matches["home_team_goal"] == matches["away_team_goal"])
# #matches["home_team_points"] = 1*(matches["home_team_goal"] == matches["away_team_goal"])
# #print(matches.shape)
# matches.head(10)


# Set the match_id to be the dataset key.
# 
# It is redundant to keep the home and away teams goals in the analysis since the output (whether the home team wins, loses or draws) is directly dependent on both features. So we'll remove them from the data set or not add them into our feature set. Other fields to exclude from the feature set include the match date (for now), the names of the teams playing etc.
# 
# home_buildUpPlayDribbling and away_buildUpPlayDribbling have None entries
# 

# In[ ]:

#matches.index = matches['match_id']
# then drop the match_id and also drop stage for now
#to_drop = ['match_id', 'stage',  'match_date','home_team_api_id',
#           'away_team_api_id','home_team', 'away_team','season',
#           'home_buildUpPlayDribbling','away_buildUpPlayDribbling']  #'home_team_goal', 'away_team_goal',
# make a copy of the matches dataframe and drop the appropriate fields while deleting the unneeded features
#matches_ml = matches.drop(to_drop, axis =1)
#print(matches_ml.shape)


# Clean up the data by converting categorical data to their one-hot encoded forms

# In[ ]:

# Make a list of all columns with categorical data
#cat_list= matches_ml.select_dtypes(include=['object']).columns.tolist()
#print(cat_list)


# In[ ]:

##print(matches_ml.columns)
#from sklearn.preprocessing import OneHotEncoder 
#from sklearn.preprocessing import LabelEncoder

## pulled in from http://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn
#class MultiColumnLabelEncoder:
#    def __init__(self,columns = None):
#        self.columns = columns # array of column names to encode

#    def fit(self,X,y=None):
#        return self # not relevant here

#    def transform(self,X):
#        '''
#        Transforms columns of X specified in self.columns using
#        LabelEncoder(). If no columns specified, transforms all
#        columns in X.
#       '''
#        output = X.copy()
#        if self.columns is not None:
#            for col in self.columns:
#                #print(col)
#                output[col] = LabelEncoder().fit_transform(output[col])
#        else:
#            for colname,col in output.iteritems():
#                output[colname] = LabelEncoder().fit_transform(col)
#        return output

#    def fit_transform(self,X,y=None):
#        return self.fit(X,y).transform(X)


# In[ ]:

#MultiColumnLabelEncoder(columns = cat_list ).fit_transform(matches_ml)
#- Then convert those fields to encoded numeric forms
#http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html
#matches_ml = pd.get_dummies(matches_ml, prefix=cat_list)
#print(new_matches.columns)


# In[ ]:

#matches_ml = h.clean_up_matches(matches)
#matches_ml = h.encode_matches(matches_ml)

#matches_ml.head()
#for i in [cat_list]:
#    print(matches_ml.iloc[[1]][i])
#    #print("field: {}, value: {}".format(i,v))
    


# In[ ]:

# move the home team points to the target field
y = np.array(matches_ml['home_team_points'])
matches_ml.drop(['home_team_points'], axis=1)


# In[ ]:

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = np.array(scaler.fit_transform(matches_ml))   #np.array(matches_ml) 


# In[ ]:

print("Feature space holds %d observations and %d features" % X.shape)
print("Unique target space:", y.shape)
print("Unique target labels:", np.unique(y))


# In[ ]:

#for c in matches_ml.columns:
#    print(matches_ml[c].isnull().values.any())
#matches_ml.isnull().any()
#matches_ml['home_team_points'].isnull().any()
np.sum(np.isinf(X)*1)


# In[ ]:




# In[ ]:

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.metrics import average_precision_score




print("Logistiic Regression:")
print("%.3f" % accuracy(y, run_cv(X,y,LR)))
print("Gradient Boosting Classifier")
print("%.3f" % accuracy(y, run_cv(X,y,GBC)))
print("Support vector machines:")
print("%.3f" % accuracy(y, run_cv(X,y,SVC)))
print("Random forest:")
print("%.3f" % accuracy(y, run_cv(X,y,RF)))
print("K-nearest-neighbors:")
print("%.3f" % accuracy(y, run_cv(X,y,KNN)))

fig, ax = plt.subplots(2,3, sharex='col', sharey='row')
#for i in xrange(len(clfs)):
#    plot_matrix(predictors[i].predict(X),y, clfs[i],ax[i,np.mod(i,3)])
#plt.colorbar()
#plt.grid(False)


# DO a simple classification exercise and see how many classes are there

# In[ ]:

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
max_clust = 0
max_score = 0
for i in range(20) :
    n_clusters = i+2
    clusterer = GaussianMixture(n_components=n_clusters, random_state= 11 )
    clusterer = clusterer.fit(X)
    # TODO: Predict the cluster for each data point
    preds = clusterer.predict(X)
    # TODO: Find the cluster centers
    centers = clusterer.means_
    # TODO: Predict the cluster for each transformed sample data point
    #sample_preds = clusterer.predict(pca_samples)

    # TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
    from sklearn.metrics import silhouette_score
    score = silhouette_score(X,preds, random_state=10)

    print("For {} clusters , score is {}".format(n_clusters, score))
    if max_score < score:
        max_score = score
        max_clust = n_clusters
    
print("Maximum cluster size and score: *{}* and *{}* ".format(max_clust, max_score))

from sklearn import metrics
print("For DBScan")
db = DBSCAN(eps=0.1, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print (n_clusters)

print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(y, labels))
print("Completeness: %0.3f" % metrics.completeness_score(y, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(y, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(y, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(y, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))


# In[ ]:



