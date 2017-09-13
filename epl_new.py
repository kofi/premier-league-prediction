
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
from sklearn.metrics import make_scorer

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

import matplotlib.ticker as plticker
from matplotlib.ticker import FormatStrFormatter

import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter)
# will list the files in the input directory

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
        scores['clf'] = "{}_{}".format(scores['clf'], kwargs['estimator'].__class__.__name__) #kwargs['estimator'].__class__.__name__
        # 
    if 'base_estimator' in kwargs: # 'OneVsRestClassifier':
        scores['clf'] = "{}_{}".format(scores['clf'],kwargs['base_estimator'].__class__.__name__)
        #kwargs['base_estimator'].__class__.__name__
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
    #print(matches[home_columns].columns.T)
    #print(matches[home_columns].describe())
    sbn.plt.show()

    #print(matches[away_columns].columns.T)
    #print(matches[away_columns].describe())
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
                    input()
                    
                    #axes[r,c].set_xticklabels(values, fontsize= 6)
                    #plt.legend((win_bar[0], drew_bar[0], lost_bar),('Won', 'Drew','Lost'), framealpha = 0.8)
            else:
                
                # Divide the range of data into bins and count survival rates
                min_value = matches[col].min()
                max_value = matches[col].max()
                value_range = max_value - min_value

                # 'Fares' has larger range of values than 'Age' so create more bins
                bins = np.arange(0, matches[col].max() + 10, 10)

                if bins > 1:
                    wins = matches[matches['home_team_outcome'] == 'win'][col].reset_index(drop = True)
                    losses = matches[matches['home_team_outcome'] == 'lose'][col].reset_index(drop = True)
                    draws = matches[matches['home_team_outcome'] == 'draw'][col].reset_index(drop = True)       
                    axes[r,c].hist(wins,bins = bins,alpha = 0.6,color = 'g',label = 'Won', normed=True)
                    axes[r,c].hist(draws, bins = bins, alpha = 0.6,color = 'y',label = 'Drew',normed=True)
                    axes[r,c].hist(losses, bins = bins, alpha = 0.6, color = 'b',label = 'Lost',normed=True)    
                    input()            

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
                train_test_split=False,league_name="England",dropna=True):

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
    options = {'compute_form':compute_form, 'window':window,'exclude_firstn':exclude_firstn,'home_advantage':home_advantage,'league_name':league_name}

    matches = h.preprocess_matches_for_season(season,**options)
    #print(matches.info())
    #print('print(matches.head().T)')
    #print(matches.head())
    #exit(0)

    # filter out only the matches with the team of interest
    if filter_team:
        matches = matches[(matches['home_team_api_id'] == my_team_id)  |
                          (matches['away_team_api_id'] == my_team_id)]

    #print("Shape before cleanup and encode: {}".format(matches.shape))

    # clean up
    matches = h.clean_up_matches(matches,ignore_columns=['season','stage'])
    #print("Matches shape B before encode {}".format(matches.shape))

    matches = h.encode_matches(matches,ignore_columns=['season','stage'])
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
    print("Home team {:.3f}% wins, {:.2f}% losses, {:.2f}% draws".format(
            100*percent_home_win,100*percent_home_loss,100*percent_home_draw))
    print("Matches entropy: {:f}".format(matches_entropy))
    print("")

    # drop Nan rows
    if dropna:
        allnas = matches.isnull().any()
        if (sum(allnas == True)):
            matches.dropna(inplace=True)
    #print("Shape of X after dropna: {}".format(matches.shape))
    #print(matches.columns.T)
    #print("Dataframe shape after dropping rows {}".format(matches.shape))

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
    matches_sub = matches.drop(['home_team_points','home_team_goal', 'season','stage',
                        'away_team_goal', 'home_team_outcome','season'], axis=1) #
    if train_test_split:
        test_matches_sub = test_matches.drop(['home_team_points','home_team_goal', 'season','stage',
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
        if train_test_split:
            ss = StandardScaler().fit(matches_sub.append(test_matches_sub))
        else:
            ss = StandardScaler().fit(matches_sub)
        X =np.array(ss.transform(matches_sub)) # np.array(StandardScaler().fit_transform(matches_sub))
        if train_test_split:
            ss = StandardScaler().fit(test_matches_sub)
            X_test = np.array(ss.transform(test_matches_sub))  #np.array(StandardScaler().fit_transform(test_matches_sub))

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

    all_scores = []
    for k in clfs:
        scores = run_kfolds(X,y,k['clf'], **k['params'])
        scores['entropy'] = matches_data['entropy']
        scores['seasons'] = i
        all_scores.append(scores)
        clf = k['clf'](**k['params'])

        if debug:
            cnf_matrix = scores['cnf_matrix']

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
def plot_roc_curves(X,y,clf,train_rows,estimator='',league_name='EPL',kwargs=None):
    '''
        Multiclass ROC curve plotting
        See http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    '''
    y = label_binarize(y, classes=output_class)
    n_classes = y.shape[1]


    # Learn to predict each class against the other
    X_train = X[:train_rows,:]
    X_test = X[train_rows:,:]
    y_train = y[:train_rows,:]
    y_test = y[train_rows:,:]


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
    plt.title('Multiclass ROC - {} {} for {} dataset'.format(clf.__class__.__name__,estimator, league_name))
    plt.legend(loc="lower right")
    plt.show()

def plot_match_hometeam_outcomes_by(matches,index_column='season',league_name='England'):
    '''
        Plot variation of home wins 
    '''
    if matches is None:
        exit()
    nseasons = len(matches['season'].unique())
    matches = matches[[index_column,'home_team_outcome']]
    matches = pd.get_dummies(matches, prefix=['home_team_outcome'], columns=['home_team_outcome'])
    to_rename = {'home_team_outcome_win':'win',
                    'home_team_outcome_draw':'draw','home_team_outcome_lose':'lose'}
    #print(matches.columns)
    matches.rename(columns=to_rename, inplace=True)
    # mgroups = matches.groupby(['season','home_team_outcome'])[['home_team_outcome']].count()
    mgroups = matches.pivot_table(['win','draw','lose'],index=index_column)
    mgroups.plot()

    #plt.tight_layout()
    plt.title("Variation of home wins/draws/losses by {} ({} league,{} season)".format(index_column,
                            league_name,nseasons))
    plt.xlabel(index_column.title())
    plt.ylabel('Fraction of Games')
    plt.savefig('variation_home_outcomes_by_{}_{}.png'.format(index_column.title(),league_name),
            format='png')
    plt.show()
    input()











# 
# The main routine for selecting which analysis and outputs to generate
# 
def main(argv):
    if len(argv) < 1:
        print("Please provide an integer ranging from 2 - 5")
        exit()

    analysis = argv[0]
    try:
        analysis = int(analysis)
    except:
        exit()
    league_name = "England"

    if analysis < 1 or analysis > 5:
        print("Oops. Cannot access random dreams. Select 1 - 5")
        exit()
    print("Selected analysis: {}".format(analysis))


    # # run EDA analysis
    # if analysis == 0:
    #     perform_eda_for_matches_1()
    #     perform_eda_for_matches_2()
    #     exit()
    print("Importing data ... ")
    h.import_datasets(league_name)
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
        {'clf': DecisionTreeClassifier, 'params':{'random_state':0}},
        {'clf':OneVsRestClassifier, 'params':{'estimator': ada}},
        {'clf': RandomForestClassifier, 'params':{}},
        {'clf': SGDClassifier,
            'params':{'loss':'log','class_weight':'balanced','penalty':'l2','n_iter':1000}}, 
        {'clf':kNN,'params':{'n_neighbors':5, 'weights':'distance'}},
        {'clf':OneVsRestClassifier, 
            'params':{'estimator': sgdc_clf}}]

    # Analysis 1:
    # Generate training and test scores for each classifier
    # Does not use cross-validation and uses only features in existing dataset
    # It loops over all classifiers, fits a model and uses it to compute training and test scores for selected metrics
    if analysis == 1:
        print("Analysis {}:".format(analysis))
        home_advantage = 'both'
        window = 0
        options = {'season_select':'all', 'compute_form':compute_form,
                    'exclude_firstn':False, 'diff_features': False, 
                    'home_advantage':home_advantage,'window':window,'istrain':False,'train_test_split':True}

        print("Get training data")        
        output = matches_for_analysis(1,**options)

        X_train, y_train = output['X'], output['y']
        X_test, y_test = output['X_test'], output['y_test']
        print("Columns:")
        pprint.pprint(output['data'].columns)
        print_spacer()

        for k in range(len(clfs)):
            c = clfs[k]
            clf_class = c['clf']
            kwargs = c['params']
            clf = clf_class(**kwargs)
            clfname = clf.__class__.__name__
            if 'estimator' in kwargs: 
                clfname = clfname + kwargs['estimator'].__class__.__name__
            if 'base_estimator' in kwargs: 
                clfname = clfname + kwargs['base_estimator'].__class__.__name__
            if 'kernel' in kwargs:
                clfname = ("{}_{}".format(clfname,kwargs['kernel']))
            print(clfname)
            # then fit the training data fold
            try:
                clf.fit(X_train,y_train)
                # get the training metrics
                train_score = get_scores(clf, X_train, y_train)
                print("Train score:")
                pprint.pprint(train_score)
                print("Test Confusion Matrix:")
                print(output_class)
                print_conf_matrix(y_train,clf.predict(X_train))
                print()
                # get the test metrics
                test_score = get_scores(clf, X_test, y_test)
                print("Test score:")
                pprint.pprint(test_score)
                print("Test Confusion Matrix:")
                print(output_class)
                print_conf_matrix(y_test,clf.predict(X_test))

            except Exception:
                print("Could not run")
            print()
            print_spacer()

    # Analysis 2:
    # use all data and get the performance scores using K-folds cross-validation using k = 5
    # This uses the preprocessed data with only existing features
    # it loops over all classifiers and reports back the cross-validation scores for a number of metrics
    if analysis == 2:
        # plot the performance of each of the algorithms for default data
        # no windows, no calculation of forms
        print("Analysis {}:".format(analysis))
        options = {'season_select':'all','compute_form':compute_form,
                 'league_name':league_name, 'window':0,'exclude_firstn':False, 'diff_features':False,'home_advantage':'both', 'istrain':True, 'train_test_split':False}
        all_runs = []
        output = matches_for_analysis(1,**options)
        
        df_scores = analysis_1('all', clfs, output, pipeline_pca=False,debug=True)
        all_runs.append(df_scores.reset_index())
        dfa = pd.concat(all_runs,ignore_index= True)
        print("Summary Results for all iterations:")
        print(dfa.drop('cnf_matrix',axis=1))
        print("Confusion Matrices")
        pprint.pprint(dfa['cnf_matrix'])
        #plot_analysis_1(dfa.drop('cnf_matrix',axis=1).drop('seasons',axis=))

    # Explore the impact of the additional features:
    # 1. The team form features
    #      This loops over multiple windows and computes scores for each classifier 
    #      set: compute_form = True,  exclude_firstn = True, home_advantage = None, train_test_split = False, isTrain = True
    # 2. The home team advantage features
    #      This resets the window to 0 and adds a set of home form features
    #       There are two ways to compute the home form: goals or points
    #      set: compute_form = False,  exclude_firstn = False, home_advantage = 'points' or 'goals' , train_test_split = False, isTrain = True  
    #
    if analysis == 3:
        print("Analysis {}:".format(analysis))
        all_runs = []
        all_entropies = []
        do_plots = False
        debug = True
        compute_form = True
        #options are: 'both' #'goals','points', none
        home_advantage = None 
        diff_features = False
        exclude_firstn = True
        train_test_split = False
        istrain = True
    
        options = {'season_select':'all','compute_form':compute_form,'league_name':league_name, 'exclude_firstn':exclude_firstn,'diff_features':diff_features, 'home_advantage':home_advantage,'train_test_split':train_test_split,'istrain':istrain}

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

            dfa = pd.concat(all_runs,ignore_index= True) 
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
                dfa_subset = dfa_windows.loc[dfa_windows['clf'].str.contains(c)]
                dfa_subset = dfa_subset[['window',sc]]
                ax = plt.plot(dfa_subset['window'],dfa_subset[[sc]],label=c,
                    linestyle='-', color=clcolor[mcount], marker=markers[mcount])
                mcount = mcount+1

            plt.legend(dfa_windows['clf'].unique())
            plt.title("{} variation".format(sc))
            plt.savefig('{}_variation_with_window.png'.format(sc),format='png')
            dfa_windows.drop('cnf_matrix',axis=1).to_csv("{}_var_with_window.csv".format(sc), encoding='utf-8')

    # This section does the parameter tuning for a subset of algorithms
    # assumes a fixed window based on analysis 3    
    if analysis == 4:
        print("Analysis {}:  Parameter tuning".format(analysis))
        do_plots = False
        debug = False
        compute_form = True
        home_advantage = None

        print("Classifier |  Window |  Best Params |  Train Score  | Test Score")
        window = 14
        options = {'season_select':'all', 'compute_form':compute_form,
                'exclude_firstn':True, 'diff_features': False, 
                'home_advantage':home_advantage,'window':window,
                'train_test_split':True,'istrain':False}
    
        output = matches_for_analysis(1,**options)
        X_train, y_train = output['X'], output['y']
        X_test, y_test = output['X_test'], output['y_test'] 

        # f1 scorer
        f1_scorer = make_scorer(f1_score, average='weighted')

        # define cross-validation splits
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

        ## iterate over SVC grid
        C_range =  np.linspace(0.001,3,40) #np.logspace(-4, 8, 13) #np.linspace(0.001,3,40) #
        gamma_range = np.linspace(0.001,3,40) # np.logspace(-9, 3, 13)# np.linspace(0.001,3,5) #
        param_grid = dict(gamma=gamma_range, C=C_range)
        svc_params = {'kernel':'rbf', 'class_weight':'balanced', 'random_state':10, 'decision_function_shape':'ovr', 'probability':True}
        grid = GridSearchCV(SVC(**svc_params), param_grid=param_grid, 
                        cv=cv,scoring=f1_scorer)
        grid.fit(X_train, y_train)

        print("SVC  |  %s  |  %s  |  %0.6f  |  %.6f"% (window, grid.best_params_, grid.best_score_, grid.score(X_test,y_test)))

        # # See https://stackoverflow.com/questions/12632992/gridsearch-for-an-estimator-inside-a-onevsrestclassifier
        # #tune the OneVsRest SGDC
        sgdc_clf = SGDClassifier(loss='log',random_state=42) 
        ovr_sgdc = OneVsRestClassifier(estimator=sgdc_clf)
        alpha_range= 10.0**-np.arange(1,7) 
        n_iter_range =  np.arange(100,3100,100)

        param_grid = {
            "estimator__alpha": alpha_range,
            "estimator__n_iter": n_iter_range
        }
        grid = GridSearchCV(ovr_sgdc, param_grid=param_grid, 
                        cv=cv,scoring=f1_scorer)
        grid.fit(X_train, y_train)
        print("1-vs-rest SGDC  |  %s  |  %s  |  %0.6f  |  %.6f"% (window, grid.best_params_, grid.best_score_, grid.score(X_test,y_test)))

        # # # Tune the Adaboot Classifier
        ada = AdaBoostClassifier(random_state=42)
        n_estimators_range = np.arange(1,51)
        #print(n_estimators_range)
        n_learning_rate_range = [0.00001,0.0001,0.001,0.05,
                                0.01,0.02,0.05,0.1,1]
        #print(n_learning_rate_range)
        param_grid = {
            "estimator__n_estimators":n_estimators_range, 
            "estimator__learning_rate":n_learning_rate_range
        }
        ovr_ada = OneVsRestClassifier(estimator=ada)
        grid = GridSearchCV(ovr_ada, param_grid=param_grid, 
                        cv=cv,scoring=f1_scorer)
        grid.fit(X_train, y_train)
        print("1-vs-Rest Adaboost  |  %s  |  %s  |  %0.6f  |  %.6f"% (window, grid.best_params_, grid.best_score_, grid.score(X_test,y_test)))
    
    # This section computes the test results for the Adaboost and SGDC classifiers
    # It also generates the ROC curves and confusion matrices   
    if analysis == 5:
        print("Analysis {}:  Other leagues".format(analysis))
        do_plots = False
        debug = False
        compute_form = True
        home_advantage = None
        league_names = ["England","Spain", "Germany"]

        for league_name in league_names:

            print("Classifier |  Window |  Best Params |  Train Score  | Test Score")
            window = 14
            options = {'season_select':'all', 'compute_form':compute_form,
                    'exclude_firstn':True, 'diff_features': False, 
                    'home_advantage':home_advantage,'window':window,
                    'train_test_split':True,'istrain':False, 'league_name': league_name}
        
            output = matches_for_analysis(1,**options)
            X_train, y_train = output['X'], output['y']
            X_test, y_test = output['X_test'], output['y_test']


            # using the SGDC classifier for different leagues
            sgdc_clf = SGDClassifier(loss='log',n_iter=1800,alpha=0.0001, random_state=42) 
            ovr_sgdc = OneVsRestClassifier(estimator=sgdc_clf)
            ovr_sgdc = ovr_sgdc.fit(X_train,y_train)
            train_score = f1_score(y_train, ovr_sgdc.predict(X_train),average='weighted')
            test_score = f1_score(y_test, ovr_sgdc.predict(X_test),average='weighted')      
            print("SGDC for {} has Training score: {}, Test score: {}".format(league_name, train_score, test_score))
            
            ada = AdaBoostClassifier(random_state=42, n_estimators=10, learning_rate=1)  
            ovr_ada = OneVsRestClassifier(estimator=ada)
            ovr_ada = ovr_ada.fit(X_train,y_train)
            train_score = f1_score(y_train, ovr_ada.predict(X_train),average='weighted')
            test_score = f1_score(y_test, ovr_ada.predict(X_test),average='weighted')   
            print("Adaboost for {} has Training score: {}, Test score: {}".format(league_name, train_score, test_score))

            train_size = X_train.shape[0]
            plot_roc_curves(np.append(X_train,X_test,axis=0),
                np.append(y_train,y_test,axis=0),ovr_ada,train_size, estimator='Adaboost',league_name=league_name)
            print()
            print("Adaboost Confusion Matrix")
            print_conf_matrix(y_test,ovr_ada.predict(X_test))






#
# Run through the sequence of analyses
#
if __name__ == '__main__':

    main(sys.argv[1:])

    