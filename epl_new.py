
# coding: utf-8

# # Exploring how team attributes impact scores in the premier league
# ## Merge the match and (winning) team data to evaluate which team features most impact

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
import sys
#from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton
#from PyQt5.QtGui import QIcon
import itertools
import pprint
import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import sqlite3 as sql
import matplotlib
matplotlib.use('TkAgg') #"Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#******************************************************************
#from PyQt5 import QtCore
import seaborn as sbn
sbn.set()
import helpers as h
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold #cross_validation import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss

from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics.pairwise import chi2_kernel
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle


import warnings
warnings.filterwarnings("ignore") #, category=DeprecationWarning)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter)
# will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
all_seasons=['2009/2010','2010/2011','2011/2012',
            '2013/2014','2014/2015','2015/2016'] # '2012/2013', '2008/2009',
training_seasons = ['2009/2010','2010/2011','2011/2012', '2013/2014','2014/2015'] 
test_seasons = ['2015/2016']
predictors = []
#clfs = ['SVR','LinearSVR']  #,'Lasso','RF','KNN']
# number of K-folds to run
nfolds = 10
# number of times to run k-folds
n_tests = 1
# the output classes for the matches
output_class = np.array(['draw','lose','win']) #np.array(['draw','lose','win'])

def print_spacer():
    print("-"*100)

def get_scores(clf, X, y):
    '''
        compute and return the scores a given classifier and dataset
    '''
    try:
        gen_score = clf.score(X,y)
    except Exception:
        y_pred = clf.predict(X)
        gen_score = (y_pred == y)*1./len(y)

    f1 = f1_score(y, clf.predict(X),average='weighted')
    ll = log_loss(y,clf.predict_proba(X))
    scores =  {'score': gen_score, 'f1_score': f1, 'log_loss': ll}
    #print("{0} score:{score}, f1 score:{f1_score},
    #logloss:{log_loss}".format(type(clf).__name__,**scores))
    return scores


def run_kfolds(X,y,clf_class,**kwargs):
    '''
        Run K-Folds for a given classifier and dataset
        Also computes relevant scores and returns them
    '''
    # Construct a kfolds object
    kf = StratifiedKFold(n_splits=nfolds,shuffle=True,random_state=50)
    #KFold(n_splits=nfolds,shuffle=True)
    y_pred = y.copy()
    scores = None
    cnf_matrix = None
    clf = None

    # iterate through number of tests to run
    for m in range(n_tests):
        # Iterate through folds
        for train_index, test_index in kf.split(X,y):
            # get the training and test sets
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Initialize a classifier with key word argument
            #if clf_class == OneVsRestClassifier:
            #    clf = clf_class(kwargs['estimator'](**kwargs['params']))
            #else:
            clf = clf_class(**kwargs)
            # then fit the training data fold ... 
            if ('kernel' in kwargs) and (kwargs['kernel'] == 'precomputed'):
                # if this is a gram kernel ...
                # https://stats.stackexchange.com/questions/92101/prediction-with-scikit-and-an-precomputed-kernel-svm
                gram = np.dot(X_train, X_train.T)
                clf.fit(gram,y_train)
                # get the scores
                local_score = get_scores(clf, np.dot(X_test, X_train.T), y_test)
            else:
                clf.fit(X_train,y_train)
                # get the scores
                local_score = get_scores(clf, X_test, y_test)
            if scores is None:
                # first time we run just set the value
                scores = local_score
                # create the confusion matrix
                if ('kernel' in kwargs) and (kwargs['kernel'] == 'precomputed'):
                     cnf_matrix = confusion_matrix(y_test, clf.predict(np.dot(X_test, X_train.T)))
                else:
                    cnf_matrix = confusion_matrix(y_test, clf.predict(X_test))
            else:
                # update the scores
                for k,v in scores.items():
                    scores[k] = scores[k] + local_score[k]
                # update the confusion matrix
                if ('kernel' in kwargs) and (kwargs['kernel'] == 'precomputed'):
                    cnf_matrix = cnf_matrix + confusion_matrix(y_test,
                                                        clf.predict(np.dot(X_test, X_train.T)))                
                else:
                    cnf_matrix = cnf_matrix + confusion_matrix(y_test,
                                                        clf.predict(X_test))

    #print(scores)
    # compute the average scores across all folds
    scores = {k: v/(nfolds*n_tests*1.0) for k,v in scores.items()}
    #print(scores)
    scores['cnf_matrix'] = cnf_matrix/(n_tests*1.0)
    scores['clf'] = clf.__class__.__name__
    if 'estimator' in kwargs: # 'OneVsRestClassifier':
        scores['clf'] = kwargs['estimator'].__class__.__name__
        # "{}_{}".format(scores['clf'], kwargs['estimator'].__class__.__name__)
    if 'base_estimator' in kwargs: # 'OneVsRestClassifier':
        scores['clf'] = kwargs['base_estimator'].__class__.__name__
        #"{}_{}".format(scores['clf'],kwargs['base_estimator'].__class__.__name__)
    if 'kernel' in kwargs:
        scores['clf'] = ("{}_{}".format(scores['clf'],kwargs['kernel']))
    return scores

def get_random_seasons(nseasons, istrain=False):
    '''
        Get a list of n_seasons random seasons
    '''
    seasons = [all_seasons[i] for i in
                        random.sample(range(len(all_seasons)),nseasons)]
    #print(seasons)
    #assert(-1==1)
    return seasons
def get_firstn_seasons(nseasons,istrain=False):
    '''
        Get a list of the first nseasons
    '''
    #print(all_seasons[0:nseasons])
    if nseasons >= len(all_seasons):
        return all_seasons[:]
    return all_seasons[0:nseasons] if not istrain else training_seasons[0:nseasons]

def get_all_seasons(nseasons,istrain=False):
    '''
        Get a list of the first nseasons
    '''
    return all_seasons if not istrain else training_seasons

def get_nth_seasons(nseasons,istrain=False):
    '''
        Get a list of the nseasons_th season
    '''
    #print(all_seasons[nseasons-1])
    return [all_seasons[nseasons-1] if not istrain else training_seasons[nseasons-1]]

def get_test_seasons(nseasons,istrain=False):
    '''
        Get a list of the nseasons_th season
    '''
    return test_seasons #[all_seasons[nseasons-1] if not istrain else training_seasons[nseasons-1]]

def perform_eda_for_matches_1():
    matches = h.preprocess_matches_for_season(None,
                compute_form=True)

    matches = h.clean_up_matches(matches)
    non_cat_list= matches.select_dtypes(exclude=['object']).columns.tolist()
    matches = matches[non_cat_list]

    matches.dropna(inplace=True)
    matches.drop(['home_team_goal','away_team_goal'], axis =1, inplace=True)

    home_columns = [x for x in matches.columns if 'home_' in x] # and 'average' in x ]
    away_columns = [x for x in matches.columns if 'away_' in x]
    # for c in matches.columns:
    #     sbn.pairplot(data=matches,y_vars=[c],
    #                  x_vars=[x for x in matches.columns if x != c])
    # seaborn plots
    # plot the home team data
    sbn.pairplot(matches[home_columns])
    print(matches[home_columns].columns.T)
    print(matches[home_columns].describe())
    sbn.plt.show()

    print(matches[away_columns].columns.T)
    print(matches[away_columns].describe())
    sbn.plt.show()

def perform_eda_for_matches_2():
    
    options = {'compute_form':False,
        'window':0,'exclude_firstn':False,'home_advantage':'goals'}
    matches = h.preprocess_matches_for_season(None,**options)
    matches = h.clean_up_matches(matches)

    matches['home_team_outcome'] = 'draw'
    matches.loc[matches['home_team_goal'] > matches['away_team_goal'],
                            ['home_team_outcome']] = 'win'
    matches.loc[matches['home_team_goal'] < matches['away_team_goal'],
                            ['home_team_outcome']] = 'lose'

    matches.dropna(inplace=True)

    cat_list= matches.select_dtypes(include=['object']).columns.tolist()
    non_cat_list= matches.select_dtypes(exclude=['object']).columns.tolist()
    print(non_cat_list)
    categorized = cat_list + non_cat_list
    n_plot_cols = 6
    n_plot_rows = int(len(categorized)/(n_plot_cols+1) )
    print((n_plot_cols,n_plot_rows))
    f, axes = plt.subplots(n_plot_rows, n_plot_cols)#, sharex=True, sharey=True) #, figsize=(7, 7)) #)

    for r in range(n_plot_rows):
        for c in range(n_plot_cols):
            col_index = n_plot_rows*r + c+ 1 

            #if col_index >= len(cat_list): #matches.columns):
            #    continue
            col =  categorized[col_index]  #matches.columns[col_index] #cat_list[col_index] 
            print('Columnn : {}'.format(col))
            axes[r,c].set_title(col, fontsize= 7)
            if col in cat_list:
                # Set the various categories
                values = matches[col].unique() 

                # Create DataFrame containing categories and count of each
                frame = pd.DataFrame(index = np.arange(len(values)), columns=(col,'Won','Lost','Drew'))
                for i, value in enumerate(values):
                    # print(i)
                    # print(value)
                    frame.loc[i] = [value, \
                        len(matches[(matches['home_team_outcome'] == 'win') & (matches[col] == value)]), \
                        len(matches[(matches['home_team_outcome'] == 'lose') & (matches[col] == value)]), \
                        len(matches[(matches['home_team_outcome'] == 'draw') & (matches[col] == value)])]
                
                # Set the width of each bar
                bar_width = 0.25 #1/(len(values)+1)

                # Display each category's survival rates
                for i in np.arange(len(frame)):

                    win_bar = axes[r,c].bar(i, frame.loc[i]['Won'],width = bar_width,color = 'g')
                    drew_bar = axes[r,c].bar(i-bar_width, frame.loc[i]['Drew'], width = bar_width, color = 'y')
                    lost_bar = axes[r,c].bar(i-2*bar_width, frame.loc[i]['Lost'], width = bar_width, color = 'b')
                    
                    #axes[r,c].set_xticklabels(values, fontsize= 6)
                    #plt.legend((win_bar[0], drew_bar[0], lost_bar),('Won', 'Drew','Lost'), framealpha = 0.8)
            else:
                
                # Divide the range of data into bins and count survival rates
                min_value = matches[col].min()
                max_value = matches[col].max()
                value_range = max_value - min_value

                # 'Fares' has larger range of values than 'Age' so create more bins
                bins = np.arange(0, matches[col].max() + 10, 10)

                wins = matches[matches['home_team_outcome'] == 'win'][col].reset_index(drop = True)
                losses = matches[matches['home_team_outcome'] == 'lose'][col].reset_index(drop = True)
                draws = matches[matches['home_team_outcome'] == 'draw'][col].reset_index(drop = True)       
                axes[r,c].hist(wins,bins = bins,alpha = 0.6,color = 'g',label = 'Won', normed=True)
                axes[r,c].hist(draws, bins = bins, alpha = 0.6,color = 'y',label = 'Drew',normed=True)
                axes[r,c].hist(losses, bins = bins, alpha = 0.6, color = 'b',label = 'Lost',normed=True)                

    # for label in (axes.get_xticklabels() + ax.get_yticklabels()):
    #     label.set_fontname('Arial')
    #     label.set_fontsize(13)
    #f
    f.tight_layout()
    f.show()
    input()
    print('exiting')


def matches_for_analysis(nseasons, season_select='firstn',filter_team=None,
                compute_form= False, window=3, exclude_firstn=True,
                diff_features= False, home_advantage=None,istrain=False,
                train_test_split=False,league_name="England"):

    season_selectors = {
        'random':get_random_seasons,
        'nth':get_nth_seasons,
        'firstn':get_firstn_seasons,
        'all':get_all_seasons,
        'test':get_test_seasons}

    if filter_team:
        my_team_id = h.get_team_id(filter_team)

    season = season_selectors.get(season_select,get_all_seasons)(nseasons,istrain)
    print("Seasons: {}".format(season))
    # get_random_seasons(nseasons) #['2010/2011','2011/2012','2012/2013','2013/2014']
    #matches = h.preprocess_matches_for_season(season)
    options = {'compute_form':compute_form, 'window':window,'exclude_firstn':exclude_firstn,
                'home_advantage':home_advantage,'league_name':league_name}

    matches = h.preprocess_matches_for_season(season,**options)

    # filter out only the matches with the team of interest
    if filter_team:
        matches = matches[(matches['home_team_api_id'] == my_team_id)  |
                          (matches['away_team_api_id'] == my_team_id)]
    # set the home status of the team of interest
    #matches.loc[matches['home_team_api_id'] == my_team_id,'isteamhome'] = 1
    #matches.loc[matches['home_team_api_id'] != my_team_id,'isteamhome'] = 0

    #print("Shape before cleanup and encode: {}".format(matches.shape))

    # clean up
    matches = h.clean_up_matches(matches,ignore_columns=['season'])
    #print("Matches shape B before encode {}".format(matches.shape))

    matches = h.encode_matches(matches,ignore_columns=['season'])
    #print("Matches shape C after encode {}".format(matches.shape))
    #print("Shape after cleanup and encode: {}".format(matches.shape))

    # create the output columns
    matches['home_team_points'] = 3*(matches['home_team_goal'] > matches['away_team_goal']) + \
                1*(matches['home_team_goal'] == matches['away_team_goal'])

    # define the output classes
    matches['home_team_outcome'] = 'draw'
    matches.loc[matches['home_team_goal'] > matches['away_team_goal'],
                            ['home_team_outcome']] = 'win'
    matches.loc[matches['home_team_goal'] < matches['away_team_goal'],
                            ['home_team_outcome']] = 'lose'

    # get some statistics
    # home team win percentages
    percent_home_win = np.sum(matches['home_team_points'] == 3) /(1. * np.max(matches.shape[0]))
    percent_home_loss = np.sum(matches['home_team_points'] == 0)/(1. * np.max(matches.shape[0]))
    percent_home_draw = np.sum(matches['home_team_points'] == 1)/(1. * np.max(matches.shape[0]))
    win_stats = [percent_home_win, percent_home_loss, percent_home_draw]
    matches_entropy = -np.sum([k*math.log(k,2) for k in win_stats])
    #print("Home team {:.3f} wins, {:.2f} losses, {:.2f} draws".format(
    #        100*percent_home_win,100*percent_home_loss,100*percent_home_draw))
    #print("Matches entropy: {:f}".format(matches_entropy))
    print("")

    # drop Nan rows
    allnas = matches.isnull().any()

    if (sum(allnas == True)):
        matches.dropna(inplace=True)
    #print("Shape of X after dropna: {}".format(matches.shape))
    #print(matches.columns.T)
    #print("Dataframe shape after dropping rows {}".format(matches.shape))

    # # delete all away team information since we are
    # # predicting only the home outcome
    # away_cols = [x for x in matches.columns if 'away_' in x]
    # matches.drop(away_cols,axis=1, inplace=True)

    # subsample data
    #matches = h.subsample_matches(matches)

    # define the output variable
    X_test = None
    y_test = None
    # set up the test data info
    if train_test_split:
        #print("Will split data into train and test")
        test_matches = matches.query('(season in @test_seasons)')
        test_match_index = test_matches.index
        #test_match_index = [x for x in test_match_index if x in matches.index]

        train_matches = matches.query('(season not in @test_seasons)')
        train_match_index = train_matches.index
        matches = train_matches

    y = np.array(matches['home_team_outcome'])
    if train_test_split:
        y_test = np.array(test_matches['home_team_outcome'])

    # then delete columns
    matches_sub = matches.drop(['home_team_points','home_team_goal', 'season',
                        'away_team_goal', 'home_team_outcome','season'], axis=1) #
    if train_test_split:
        test_matches_sub = test_matches.drop(['home_team_points','home_team_goal', 'season',
                        'away_team_goal', 'home_team_outcome','season'], axis=1)
        
    #print("Shape of matches after drop columns: {}".format(matches_sub.shape))
    #print(matches_sub.columns.T)
    print()
    # only  use the diffed features for away & home teams when True
    if diff_features:
        matches_sub= h.matches_home_away_diff(matches_sub)
        if train_test_split:
            test_matches_sub = h.matches_home_away_diff(test_matches_sub)
    # finally transform the data and scale to normalize
    try:
        # for SVC we need to scale both the test and training sets at the same time
        # http://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use
        ss = StandardScaler().fit(matches_sub.append(test_matches_sub))
        X =np.array(ss.transform(matches_sub)) # np.array(StandardScaler().fit_transform(matches_sub))
        if train_test_split:
            ss = StandardScaler().fit(test_matches_sub)
            X_test = np.array(ss.transform(test_matches_sub))  #np.array(StandardScaler().fit_transform(test_matches_sub))
        #print("Shape of X after scaling: {}".format(X.shape))
    except Exception:
        print("excepted")
        matches = None
        matches_sub = None
        X = y = None
        X_test = None
        y_test = None

        raise


    return {'rawdata':matches, 'data':matches_sub, 'X':X,
                'y':y, 'entropy': matches_entropy,
                'X_test': X_test, 'y_test': y_test}

# print the confusion matrix
def print_conf_matrix(y, y_pred, plot=False):
    cnf_matrix = confusion_matrix(y, y_pred) #, labels=output_class)
    print(cnf_matrix)
    if plot:
        plot_conf_matrix(cnf_matrix)
# plot the confusion matrix
def plot_conf_matrix(cnf_matrix):
     np.set_printoptions(precision=2)
     plt.figure()
     h.plot_confusion_matrix(cnf_matrix, classes=output_class,
                      title='Confusion matrix, without normalization')

     plt.figure()
     h.plot_confusion_matrix(cnf_matrix, classes=output_class,
                normalize=True, title='Normalized Confusion matrix')
     plt.show()

# run the first analysis to just pick the best performing algorithms
def analysis_1(i, clfs, matches_data,pipeline_pca=False,debug=False):
    '''
        first analysis to identify best performing classifiers for further tuning
    '''
    X = matches_data['X']
    y = matches_data['y']
    if X is None or y is None:
        return None

    if pipeline_pca:
        if debug :
            print("Shape of X before PCA: {}".format(X.shape))
        # lets try with PCA
        pca = PCA() #28)
        pca = pca.fit(X)
        X = pca.transform(X)
        if debug :
            print("Shape of X after PCA: {}".format(X.shape))

        #print("Percent explain variance")
        #print(100*pca.explained_variance_ratio_)

    # ... get the training score
    #clfs = [{'clf': SVC, 'params':{'kernel':'rbf', 'probability':True}}]

    all_scores = []
    for k in clfs:
        scores = run_kfolds(X,y,k['clf'], **k['params'])
        scores['entropy'] = matches_data['entropy']
        scores['seasons'] = i
        all_scores.append(scores)
        clf = k['clf'](**k['params'])
        #print(clf.__class__.__name__ + ' ' + k['params'].get('kernel',''))
        #if clf.__class__.__name__ in ["SVC","DecisionTreeClassifier"]:
        if debug:
            cnf_matrix = scores['cnf_matrix']
            #print(cnf_matrix)
            #plot_conf_matrix(cnf_matrix)
            #print(classification_report(y, y_pred,digits=5))

    df_scores = pd.DataFrame(all_scores)
    df_scores.set_index('clf', inplace=True)
    #print(df_scores.drop('cnf_matrix',axis=1))
    # save the scores
    #all_runs.append(df_scores.reset_index())
    if debug:
        agg_cols = ['log_loss','score','f1_score']
        print(pd.DataFrame({'Max':df_scores[agg_cols].max(axis= 0),
                            'ArgMax':df_scores[agg_cols].idxmax(axis= 0),
                            'Min':df_scores[agg_cols].min(axis= 0),
                            'ArgMin':df_scores[agg_cols].idxmin(axis= 0)}))
    print("-"*100)
    print()

    return df_scores




# plot the analysis 1 data
def plot_analysis_1(data):

    classifiers = data.unstack()['clf'].unique()
    scores = ['f1_score','log_loss','score']
    clcolor =['red','blue','green','black','yellow','pink']
    fig, ax = plt.subplots(len(scores))
    pltcnt = 0
    for s in scores:
        for c in classifiers:
            myd = data.loc[data['clf'].str.contains(c)] #dfa
            ax[pltcnt].plot(myd['seasons'], myd[s],'-',label=c)
        ax[pltcnt].axis('tight')
        ax[pltcnt].set_xlim(0,len(myd['seasons'])-1)
        ax[pltcnt].legend(loc='lower center',frameon=False,ncol=len(classifiers))
        ax[pltcnt].set_xlabel('# of Seasons in data')
        ax[pltcnt].set_ylabel('Score value')
        pltcnt = pltcnt + 1
    plt.show()

# generate the ROC curves for a given clf and data
def plot_roc_curves(X,y,clf,train_rows,kwargs=None):
    '''
        Multiclass ROC curve plotting
        See http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    '''
    y = label_binarize(y, classes=output_class)
    #print(np.append(y_train,y_test).shape)
    n_classes = y.shape[1]
    #print(y.shape)
    #print(n_classes)

    # Learn to predict each class against the other
    # shuffle and split training and test sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4,random_state=0)
    X_train = X[:train_rows,:]
    X_test = X[train_rows:,:]
    y_train = y[:train_rows,:]
    y_test = y[train_rows:,:]
    #print(y_train.shape)
    #print(y_test.shape)

    classifier = OneVsRestClassifier(clf)
    try:
        y_score = classifier.fit(X_train, y_train).decision_function(X_test)
        #print(y_score.shape)
    except Exception:
        exit()
    
    lw = 2
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_score[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(output_class[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC for {} for EPL dataset'.format(clf.__class__.__name__))
    plt.legend(loc="lower right")
    plt.show()


# Run through the sequence of analyses
if __name__ == '__main__':

    analysis = 2
    league_name = "Germany"

    # run EDA analysis
    if analysis == 0:
        perform_eda_for_matches_1()
        perform_eda_for_matches_2()

    print("Importing data ... ")
    h.import_datasets()
    #setup a few data structures to store results
    all_runs = []
    all_entropies = []
    do_plots = False
    debug = False
    compute_form = False
    sgdc_clf = SGDClassifier(loss='log',alpha=0.001,n_iter=100)
    # for Linear SVC see http://scikit-learn.org/stable/modules/calibration.html
    #https://stackoverflow.com/questions/26478000/converting-linearsvcs-decision-function-to-probabilities-scikit-learn-python
    lscv = LinearSVC(class_weight='balanced')
    ada = AdaBoostClassifier()
    clfs = [{'clf': DummyClassifier,
                'params':{'strategy':'most_frequent', 'random_state':0}},
        {'clf': SVC, 'params':{'kernel':'linear', 'class_weight':'balanced',
                'decision_function_shape':'ovr', 'probability':True}},
        {'clf': SVC, 'params':{'kernel':'rbf', 'class_weight':'balanced', 
            'decision_function_shape':'ovr', 'probability':True}},
        {'clf': SVC, 'params':{'kernel':'precomputed', 'class_weight':'balanced', 
            'decision_function_shape':'ovr', 'probability':True}},
        #{'clf': CalibratedClassifierCV, 'params':{'base_estimator':lscv}},
        {'clf': DecisionTreeClassifier, 'params':{'random_state':0}},
        #{'clf': GNB, 'params':{}},
        {'clf':OneVsRestClassifier, 'params':{'estimator': ada}},
        {'clf': RandomForestClassifier, 'params':{}},
        {'clf': SGDClassifier,
            'params':{'loss':'log','class_weight':'balanced','penalty':'l2','n_iter':1000}}, #,'alpha':0.001,'n_iter':100}},
        {'clf':kNN,'params':{'n_neighbors':5, 'weights':'distance'}},
        {'clf':OneVsRestClassifier, 
            'params':{'estimator': sgdc_clf}}]
        #'params':{'loss':'log','alpha':0.001,'n_iter':100}}}]

    # Analysis 1:
    # use all data and get the basic scores
    #
    if analysis == 1:
        print("Analysis 1:")
        options = {'season_select':'all','compute_form':compute_form,'league_name':league_name,
                'window':0,'exclude_firstn':False, 'diff_features':False,
                    'home_advantage':'both','train_test_split':True}
        output = matches_for_analysis(1,**options)
        df_scores = analysis_1('all', clfs, output, pipeline_pca=False,debug=True)
        all_runs.append(df_scores.reset_index())
        dfa = pd.concat(all_runs,ignore_index= True)
        print("Summary Results for all iterations:")
        print(dfa.drop('cnf_matrix',axis=1))
        print("Confusion Matrices")
        pprint.pprint(dfa['cnf_matrix'])

    
    if analysis == 2:
        print("Analysis 2:")
        all_runs = []
        all_entropies = []
        do_plots = False
        debug = True
        compute_form = True
        home_advantage = 'both' #'goals','points'
        diff_features = False

        options = {'season_select':'all','compute_form':compute_form,'league_name':league_name,
                    'exclude_firstn':True,'diff_features':diff_features, 
                    'home_advantage':home_advantage,'train_test_split':True}

        window_range = 1 
        if compute_form:
            window_range = 20

        print(window_range)
        
        dfa_windows = pd.DataFrame()

        for window in range(window_range):
            all_runs = []
            options['window'] = window
            print("Window: {}".format(window))
            # loop over all data
            for i in range(1): #len(all_seasons)):
                output = matches_for_analysis('all_train',**options)
                df_scores = analysis_1(i, clfs, output, pipeline_pca=False,debug=debug)
                # save the scores
                all_runs.append(df_scores.reset_index())

            dfa = pd.concat(all_runs,ignore_index= True) #, keys=range(len(all_seasons)))
            if debug:
                print()
                print("Summary Results for window {}".format(window))
                pprint.pprint(dfa.drop('cnf_matrix',axis=1))
                #print("Confusion Matrices")
                #pprint.pprint(dfa['cnf_matrix'])
            if do_plots:
                plot_analysis_1(dfa)
            
            dfa['window'] = window
            dfa_windows = dfa_windows.append(dfa)
            print("="*100)

        #pprint.pprint(dfa_windows.drop('cnf_matrix',axis=1))
        markers = ['h','+','p','*','o','s','D','x','1','2']
        clcolor =['r','b','g','k','y','m','c','burlywood','purple','xkcd:royal blue']
        mcount = 0

        for sc in ['score','f1_score','log_loss']:
            mcount = 0
            plt.figure()
            for c in dfa_windows['clf'].unique():
                dfa_subset = dfa_windows.loc[dfa_windows['clf'].str.contains(c)] #dfa
                dfa_subset = dfa_subset[['window',sc]]
                ax = plt.plot(dfa_subset['window'],dfa_subset[[sc]],label=c,
                    linestyle='-', color=clcolor[mcount], marker=markers[mcount])
                mcount = mcount+1

            # use only integer labels for x-axis
            #for axis in [ax.xaxis]:
            #ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            # Shrink current axis's height by 10% on the bottom
            #box = ax.get_position()
            # ax.set_position([box.x0, box.y0 + box.height * 0.1,
            #                 box.width, box.height * 0.9])
            # Put a legend below current axis
            plt.legend(dfa_windows['clf'].unique())
            #,loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=4)
            #plt.legend(dfa_windows['clf'].unique(), ncol=2, loc='upper left')
            plt.title("{} variation".format(sc))
            plt.savefig('{}_variation_with_window.png'.format(sc),format='png')
            #plt.show()
            dfa_windows.drop('cnf_matrix',axis=1).to_csv("{}_var_with_window.csv".format(sc), encoding='utf-8')

        
    if analysis == 3:
        print("Analysis 3: SVC Parameter tuning")
        do_plots = False
        debug = True
        compute_form = True
        home_advantage = 'both'
        window = 14
        options = {'season_select':'all', 'compute_form':compute_form,
                 'exclude_firstn':True, 'diff_features': False, 
                 'home_advantage':home_advantage,'window':window,
                 'train_test_split':True}
        
        output = matches_for_analysis(1,**options)
        X, y = output['X'], output['y']

        # iterate over SVC grid
        C_range =  np.linspace(0.0001,3,40) #np.logspace(0.0001,5,100) # [0.01, 0.25, 0.5, .75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3,10,30,100] #np.linspace(0.001,0.03,15) # 3, 12)
        gamma_range = np.linspace(0.0001,3,40) #np.linspace(0.0001,1,100) #[0, 1e-5, 1e-4,1e-3, .01, .1, .15, .2, .25, .5, .75, 1] # #np.linspace(1,7,15) #12)
        param_grid = dict(gamma=gamma_range, C=C_range)
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        svc_params = {'kernel':'rbf', 'class_weight':'balanced','decision_function_shape':'ovr', 'probability':True}
        grid = GridSearchCV(SVC(**svc_params), param_grid=param_grid, cv=cv,scoring='f1_weighted')
        grid.fit(X, y)

        print("The best parameters are %s with a score of %0.6f"
            % (grid.best_params_, grid.best_score_))
        

        #input() 

        #C_range = grid.best_params_['C']* np.linspace(0.95,1.05,19)#np.linspace(0.6,1.2,10)
        #gamma_range =  grid.best_params_['gamma']* np.linspace(0.95,1.05,19) #np.linspace(0.5,1.,10)
        #0.0001*np.linspace(0.6,1.1,11) #0.0001*np.linspace(0.8,1.2,11)
        all_scores = []
        max_scores = {'f1_score':{'value':0.000, 'C': None, 'gamma':None},
                    'score':{'value':0.000, 'C': None, 'gamma':None},
                    'log_loss':{'value':0.000, 'C': None, 'gamma':None}}

        for C in C_range:
            for gamma in gamma_range:
                #print("C: {}, gamma: {}".format(C, gamma))
                svc_params = {'kernel':'rbf', 'probability':True,
                            'class_weight':'balanced','C':C,'gamma':gamma}
                clf = SVC
                scores = run_kfolds(X,y,clf,**svc_params)
                scores['C'] = C
                scores['gamma'] = gamma
                all_scores.append(scores)

                for k in max_scores.keys():
                    if scores[k] > max_scores[k]['value']:
                        max_scores[k]['value'] = scores[k]
                        max_scores[k]['C'] = C
                        max_scores[k]['gamma'] = gamma

        df_scores = pd.DataFrame(all_scores)  #pd.DataFrame(all_scores)
        df_scores.reset_index()
        #print(df_scores.columns)
        #df_scores.set_index('clf', inplace=True)
        df_scores_nocnf = df_scores.drop('cnf_matrix',axis=1)
        #print(df_scores_nocnf)
        df_scores_nocnf.to_csv("param_tuning_{}.csv".format(window), encoding='utf-8')

        print()
        print("Max Scores :") # df_scores['f1_score'].idxmax()]
        pprint.pprint(max_scores)

        to_plot = ['C','gamma','score']
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.title("F1 Score")
        ax = sbn.heatmap(df_scores.pivot('C','gamma','f1_score'))
        #ax.set_yticklabels(rotation=0)
        ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 8)
        plt.show()


        input()
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.title("Score")
        ax = sbn.heatmap(df_scores.pivot('C','gamma','score'))
        ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 8)
        #plt.yticks(rotation=0) 
        plt.show()

    if analysis == 4:
        # running a test on the test data
        print("Analysis 4:")

        C =   569.04542855790191 #4862.8956398672553 # 569.04542855790191 #6221.488221346056 #1638.0284049793586
        gamma = 0.0001 #0.00001 #0.0001# 6.3636363636363641e-05 #0.0001
        svc_params = {'kernel':'rbf', 'class_weight':'balanced','probability':True,
                        'decision_function_shape':'ovr','C':C,'gamma':gamma}
        pprint.pprint("SVC params {}".format(svc_params))
        compute_form = True
        home_advantage = 'goals'
        window = 14
        options = {'season_select':'all', 'compute_form':compute_form,
                    'exclude_firstn':True, 'diff_features': False, 
                    'home_advantage':home_advantage,'window':window,'istrain':False,
                    'train_test_split':True}

        print("Get training data")        
        output = matches_for_analysis(1,**options)

        X_train, y_train = output['X'], output['y']
        X_test, y_test = output['X_test'], output['y_test']

        # pca
        # ndims = 12
        # pca = PCA(n_components=ndims)
        # pca = pca.fit(X_train)
        # pca_results = h.pca_results(X_train, pca, output['data'].columns)
        # components = pca_results[[c for c in pca_results.columns if c != 'Explained Variance']]
        # components = components.T
        # for d in components.columns:
        #     print("Dimension: {}".format(d))
        #     print(components[abs(components[d])>0.2].index)
        #     print_spacer()

        # print(pca.explained_variance_ratio_)

        clf = SVC(**svc_params)
        # then fit the training data fold
        clf.fit(X_train,y_train)
        # get the scores
        local_score = get_scores(clf, X_test, y_test)
        pprint.pprint(local_score)

        train_size = X_train.shape[0]
        plot_roc_curves(np.append(X_train,X_test,axis=0),
                        np.append(y_train,y_test,axis=0),clf,train_size)

        print("one vs all svc rbf")
        clf = OneVsRestClassifier(SVC(**svc_params))
        clf.fit(X_train, y_train)
        local_score = get_scores(clf, X_test, y_test)
        pprint.pprint(local_score)

        print("one vs all sgd")
        clf = OneVsRestClassifier(SGDClassifier(**{'loss':'log','alpha':0.001,'n_iter':100}))
        clf.fit(X_train, y_train)
        local_score = get_scores(clf, X_test, y_test)
        pprint.pprint(local_score)

# get the confusion matrix and plot for the RBF



exit()


# # get the confusion matrix and plot for the RBF
# #cnf_matrix = confusion_matrix(y_test, y_test_pred) #, labels=output_class)
# #print(np.sum(np.sum(cnf_matrix)))


# #print("Verification of Confusion matrix")
# #for i in output_class:
# #    for j in output_class:
# #        matrix_val = np.dot((y_test==i)*1.,(y_test_pred == j)*1)
# #        print("{} but predicts {}:  {}".format(i,j,matrix_val))

# #print()
# #print(cnf_matrix)
# #print()


# #### Problem seems to be the model is overpredicting one classifier.
# #### Could be a problem due to the unbalanced data set
# #### need to come up with an approach to oversample the smaller categories
# #### and undersample the most frequent class

# print('### Subsampling Data ####')
# draws = matches[matches['home_team_outcome'] == 'draw']
# wins = matches[matches['home_team_outcome'] == 'win']
# losses = matches[matches['home_team_outcome'] == 'lose']
# print("Losses:{}, draws:{}, wins:{}".format(len(losses), len(draws), len(wins)))

# # subsample losses
# are_draws_small = True if len(draws) < len(losses) else False

# if are_draws_small:
#     percentage = len(draws)/float(len(losses))
#     losses_sampled =  losses.sample(frac = percentage, random_state = 2)
#     percentage = len(draws)/float(len(wins))
#     wins_sampled = wins.sample(frac = percentage, random_state = 2)
#     matches_sampled = draws.append(wins_sampled)
#     matches_sampled = matches_sampled.append(losses_sampled)

#     # print stats
#     print("Percentage losses	:", len(losses_sampled)/float(len(matches_sampled)))
#     print("Percentage draws		:", len(draws)/float(len(matches_sampled)))
# else:
#     percentage = len(losses)/float(len(draws))
#     draws_sampled =  draws.sample(frac = percentage, random_state = 2)
#     percentage = len(losses)/float(len(wins))
#     wins_sampled = wins.sample(frac = percentage, random_state = 2)
#     matches_sampled = losses.append(wins_sampled)
#     matches_sampled = matches_sampled.append(draws_sampled)

#     #print stats
#     print("Percentage draws	:", len(draws_sampled)/float(len(matches_sampled)))
#     print("Percentage losses		:", len(losses)/float(len(matches_sampled)))
# #matches_sampled = draws.append(wins_sampled)
# #matches_sampled = matches_sampled.append(losses_sampled)

# print("Percentage wins		:", len(wins_sampled)/float(len(matches_sampled)))
# print("Total matches		:", len(matches_sampled))


# # define the output variable
# y = np.array(matches_sampled['home_team_outcome'])

# print("Unique Y ", matches_sampled['home_team_outcome'].unique())

# # then delete columns
# matches_hm_goals = matches_sampled.drop(['home_team_points','home_team_goal',
#                                     'away_team_goal','home_team_outcome'], axis=1)
# # finally transform the data and scale to normalize
# X = np.array(scaler.fit_transform(matches_hm_goals))
# print("shape of X: {}".format(X.shape))
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=14)

# ## lets try with PCA
# pca = PCA(n_components=28)
# pca = pca.fit(X)
# #X = pca.transform(X)

# #print("Percent explain variance")
# #print(100*pca.explained_variance_ratio_)





# # In[12]:

# from sklearn.metrics import roc_curve, auc
# from sklearn.metrics import accuracy_score
# # Compute ROC curve and ROC area for each class
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(3):
#     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])


