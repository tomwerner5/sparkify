# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 02:03:03 2021

@author: J76747
"""
import requests
import json
import urllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import os
import sys
import json
from tsdst.utils import updateProgBar
from timeit import default_timer as dt


def drop_extras(data):
    '''
    Drop dataframe columns that contain only zeros. This is meant to be
    used on dummy encoded variables, or any variable where a zero value
    represents the non-interesting case..

    Parameters
    ----------
    data : dataframe
        A pandas dataframe.

    Returns
    -------
    subx : dataframe
        A dataframe with the offending columns dropped.

    '''
    cols_to_drop = []
    for col in subx.columns:
        if subx[col].mean() == 0:
            cols_to_drop.append(col)
    
    print('Dropped: ', cols_to_drop)
    subx = subx.drop(cols_to_drop, axis=1)
    return subx


def data_pipeline(data, estimator, parameters, target_var, scaler=None, 
                 test_size=0.25, random_state=42, cv=5, scoring='f1',
                 verbose=10, n_jobs=1):
    '''
    Data pipeline performs scaling and generates an arbitrary model.

    Parameters
    ----------
    data : dataframe
        The data you want to model.
    estimator : sklearn estimator
        An estimator that is similar to sklearn estimator objects.
    parameters : dict
        A dictionary of parameters to gridsearch as part of the pipeline.
    target_var : str
        The target variable in the dataset.
    scaler : sklearn object, None
        An object that fits/transforms data, default is minmax scaler.
        Follows sklearn API.
    test_size : float
        The size of the test/hold-out set
    random_state : int
        Random seed for reproducing data splits
    cv : int
        Number of (Stratified) K-folds in cross-validation
    scoring : str, list, or sklearn scorer object
        Function to use in scoring gridsearch
    verbose : str
        Print progress of gridsearch. Higher number means more output
    n_jobs : int
        Number of cores to use in gridsearch processing
    
    Returns
    -------
    The model.

    '''
    # List of training columns
    train_cols = [col for col in data.columns if col != target_var]
    
    # Split data into hold-out/training set
    X_train, X_test, y_train, y_test = train_test_split(data[train_cols],
                                                        data[target_var],
                                                        test_size=test_size,
                                                        random_state=random_state)
    
    # Initialize pipeline
    pipe = Pipeline(steps=[('scaler', scaler), ('estimator', estimator)])
    
    # Initialize gridsearch
    clf = GridSearchCV(estimator=pipe, param_grid=parameters, cv=cv,
                       refit=True, scoring=scoring, verbose=verbose,
                       n_jobs=n_jobs)
    
    # fit gridsearch
    clf.fit(X_train, y_train.values.reshape(-1, ))
    
    # Print CV results
    cv_results = pd.concat([pd.DataFrame(clf.cv_results_["params"]),
                            pd.DataFrame(clf.cv_results_["mean_test_score"],
                                         columns=["F1 Score"])],
                           axis=1).sort_values('F1 Score', ascending=False)
    
    # Prediction on hold-out set
    Y_pred = clf.predict(X_test.values)
    
    # Print hold-out results
    print('F1: ', f1_score(y_test.values.reshape(-1, ), Y_pred))
    print('Accuracy: ', accuracy_score(y_test.values.reshape(-1, ), Y_pred))
    
    return clf


# The url for retrieving the data
url = 'https://udacity-dsnd.s3.amazonaws.com/sparkify/sparkify_event_data.json'
# I found that downloading the data to my hardrive made the rest of the steps
# faster for me, so this was worth it to me. If you just want to pull straight 
# from the website, just replace the references of 'large_sparkify' to the url,
# etc.
urllib.request.urlretrieve(url, 'large_sparkify.json')

# First, generate a list of all the unique userId's so that the data
# can be divided into smaller, loadable chunks. The goal is to have several
# small json files where a single user's data only appears in one file (i.e.
# a user's data is grouped together in the same file). This will run very quick
ids = []
with open('large_sparkify.json') as f:
    u_key = 'userId'
    with open('new_large_sparkify.json', 'a') as f2:
        for i, line in enumerate(f):
            match = re.search('\"' + u_key + '\".*?, \"', line).group(0)
            ids.append(match)
            
            if i % 100000 == 0: print(i)
            
            f2.write(line)

# generate unique set of userId's
unique_ids = list(set(ids))
number_of_files = 10

# establish the group size per file (number of users in each file)
group_size = int(len(unique_ids)/number_of_files)+1

# create the unique_groups of userId's
id_groups = [unique_ids[(i*group_size):((i+1)*group_size)] for i in range(number_of_files)]

# initialize and open files for each group
group_files = [open('sparkify_group_' + str(i) + '.json', 'a') for i in range(number_of_files)]        

# Divide and write users to their respective files
with open('large_sparkify.json') as f:
    key = 'userId'
    for i, line in enumerate(f):
        if i % 100000 == 0: print(i)
        
        match = re.search('\"' + key + '\".*?, \"', line).group(0)
        
        for i, group in enumerate(group_files):
            if match in id_groups[i]:
                group.write(line)

for group in group_files:
    group.close()

# Load the mini files that were created
files = ['sparkify_group_'+ str(i) + '.json' for i in range(number_of_files)]

# Create functions for future data aggregations
longest_song = lambda x: 0 if pd.isnull(np.max(x)) else np.max(x)
listening_time = lambda x: 0 if pd.isnull(np.sum(x)) else np.sum(x)
number_of_songs = lambda x: 0 if pd.isnull(x.count()) else x.count()
minmax = lambda x: np.max(x)-np.min(x)

# Initialize dictionaries for translating complicated userAgent labels
ua_value = {'userAgent_mozilla macintosh intel mac os applewebkit khtml like gecko chrome safari': 0}
ua_dict = {'userAgent_mozilla macintosh intel mac os applewebkit khtml like gecko chrome safari': 'userAgent_0'}

# Initialize dictionaries for performing aggregations, i.e. assigns functions
# to columns
agg1_dict = {'itemInSession': 'max',
              'length': [longest_song, listening_time],
              'song': number_of_songs,
              'ts': ['min', 'max', minmax],
              'registration': 'min',
              'cancelled': 'max',
              'gender_F': 'max',
              'gender_M': 'max',
              'gender_Unknown': 'max',
              'status_200': 'sum',
              'status_307': 'sum',
              'status_404': 'sum',
              'level_paid': 'sum',
              'level_free': 'sum',
              'method_PUT': 'sum',
              'method_GET': 'sum',
              'page_NextSong': 'sum',
              'page_Thumbs Up': 'sum',
              'page_Home': 'sum',
              'page_Add to Playlist': 'sum',
              'page_Add Friend': 'sum',
              'page_Roll Advert': 'sum',
              'page_Register': 'sum',
              'page_Submit Registration': 'sum',
              'page_Login': 'sum',
              'page_Logout': 'sum',
              'page_Thumbs Down': 'sum',
              'page_Downgrade': 'sum',
              'page_Settings': 'sum',
              'page_Help': 'sum',
              'page_Upgrade': 'sum',
              'page_About': 'sum',
              'page_Save Settings': 'sum',
              'page_Error': 'sum',
              'page_Submit Upgrade': 'sum',
              'page_Submit Downgrade': 'sum',
              }

agg2_dict = {'itemInSessionmax': 'mean', 
              'length<lambda_0>': 'max', 
              'length<lambda_1>': ['mean', 'sum'], 
              'sessionId': 'count',
              'song<lambda>': 'sum',
              'ts<lambda_0>': ['mean', 'sum'], 
              'tsmin': 'min', 
              'tsmax': 'max', 
              'registrationmin': 'min', 
              'cancelledmax': 'max',
              'gender_Fmax': 'mean',
              'gender_Mmax': 'mean',
              'gender_Unknownmax': 'mean',
              'status_200sum': 'mean',
              'status_307sum': 'mean',
              'status_404sum': 'mean',
              'level_paidsum': 'mean',
              'level_freesum': 'mean',
              'method_PUTsum': 'mean',
              'method_GETsum': 'mean',
              'page_NextSongsum': 'mean',
              'page_Thumbs Upsum': 'mean',
              'page_Homesum': 'mean',
              'page_Add to Playlistsum': 'mean',
              'page_Add Friendsum': 'mean',
              'page_Roll Advertsum': 'mean',
              'page_Registersum': 'mean',
              'page_Submit Registrationsum': 'mean',
              'page_Loginsum': 'mean',
              'page_Logoutsum': 'mean',
              'page_Thumbs Downsum': 'mean',
              'page_Downgradesum': 'mean',
              'page_Settingssum': 'mean',
              'page_Helpsum': 'mean',
              'page_Upgradesum': 'mean',
              'page_Aboutsum': 'mean',
              'page_Save Settingssum': 'mean',
              'page_Errorsum': 'mean',
              'page_Submit Upgradesum': 'mean',
              'page_Submit Downgradesum': 'mean',
              }

X_final = None
X_list = []

#initialize start time for progress tracking
t0 = dt()

# For each file generated earlier
for i, file in enumerate(files):
    print('File ', i)
    print('    read in data')
    
    # Load the data in that file
    X = pd.read_json(file,
                     lines=True, 
                     convert_dates=False,
                      dtype={
                              #'ts': np.int16,
                             'userId': object,
                             'sessionId': object,
                             'page': object,
                             'auth': object,
                             'method': object,
                             'status': object,
                             'level': object,
                             #'itemInSession': np.int8,
                             #'location': str,
                             'userAgent': object,
                             #'lastName': str,
                             #'firstName': str,
                             #'registration': np.int16,
                             'gender': object,
                             #'artist': str,
                             'song': object,
                             #'length': np.float16
                        }
                     )

    print('    edit data')
    
    # Drop na values
    X = X.dropna(subset=["userId", "sessionId"])
    X = X[X['userId'] != ""]
    
    # Drop columns that will not be used in the analysis
    X = X.drop(['location', 'lastName', 'auth', 'artist', 'firstName'], axis=1)
    
    # Reduce timestamps to seconds (for comparing to length variable)
    X['ts'] = X['ts']/1000
    X['registration'] = X['registration']/1000

    # Reduce the complexity of userAgent variable through string manipulation.
    # This allows it to be dummy encoded in a more practical way later
    X['userAgent'] = X['userAgent'].str.replace(r'[^a-zA-Z]', ' ', regex=True)
    X['userAgent'] = X['userAgent'].str.replace(r'\s+', ' ', regex=True).str.lower().str.strip()
    X['userAgent'] = X['userAgent'].str.replace(r'\s[a-z]\s', ' ', regex=True)

    # Fill gender/userAgent NA values with a category, if present
    X[['userAgent', 'gender']] = X[['userAgent','gender']].fillna(value='Unknown')
    X.registration.fillna(X.ts, inplace=True)
    
    print('    create dummy data')
    # Generate the dummy encoding
    X = pd.get_dummies(X, columns=['gender', 'level', 'method', 'userAgent', 'page', 'status'])
    X = X.drop(['page_Cancel'], axis=1)
    
    # Define the truth label (user cancellations)
    X = X.rename({'page_Cancellation Confirmation': 'cancelled'}, axis=1)

    # Generate list of userAgent columns from the dummy encoded data
    userAgent_columns = [col for col in X.columns if 'userAgent' in col]
    
    # Update the userAgent dictionaries
    for col in userAgent_columns:
        if col not in list(ua_value.keys()):
            ua_value.update({col: max(ua_value.values()) + 1})
            ua_dict[col] = 'userAgent_'+str(ua_value[col])
    
    # Rename the columns of X with the new userAgent column names, and
    # update the list of names
    X = X.rename(ua_dict, axis=1)
    userAgent_columns = [col for col in X.columns if 'userAgent' in col]
    
    # Update the first aggregation dict with any additional userAgent columns
    agg1_dict.update({col: 'sum' for col in userAgent_columns})
    all_agg_cols = [col for col in X.columns if col != 'userId' and col != 'sessionId']
    sub_agg1 = {k: agg1_dict[k] for k in all_agg_cols}
    
    print('    aggregate data 1')
    # Perform aggregation at the sessionId level. Remove multilevel index.
    session_summary = X.groupby(['userId', 'sessionId'], as_index=False).agg(sub_agg1)
    session_summary.columns = session_summary.columns.map(''.join)
    
    # Update the second aggregation dict with any additional userAgent columns
    agg2_dict.update({col+'sum': 'mean' for col in userAgent_columns})
    all_agg_cols2 = [col for col in session_summary.columns if col != 'userId']
    sub_agg2 = {k: agg2_dict[k] for k in all_agg_cols2}
    
    print('    aggregate data 2')
    # Perform aggregation at the userId level
    user_summary = session_summary.groupby(['userId'], as_index=False).agg(sub_agg2)
    user_summary.columns = user_summary.columns.map(''.join)
    
    # Append data to list for concatenation later
    X_list.append(user_summary.copy(deep=True))
    
    # garbage collect
    session_summary = []
    user_summary = []
    
    updateProgBar(i+1, len(files), t0)

# Generate complete list of columns, and perpare the dataframes for merging.
# For each dataframe, check if the column exists, and if it doesn't, create
# the column and initialize it with 0.
final_columns = []
for df in X_list:
    final_columns = final_columns + list(df.columns)

final_columns = list(set(final_columns))

for df in X_list:
    for col in final_columns:
        if col not in df.columns:
            df[col] = 0
    # Ensure all dataframe have same order of columns
    df = df[final_columns]

# Concatenate all the aggregated dataframes into one.
X_final = pd.concat(X_list, ignore_index=True)

# Dictionary to rename columns in X_final
rename_keys = {
    'userId': 'userId',
    'gender_Fmaxmean': 'avg_gender_F',
    'gender_Mmaxmean': 'avg_gender_M',
    'gender_Unknownmaxmean': 'avg_gender_Unknown',
    'itemInSessionmaxmean': 'avg_num_items_in_session',
    'length<lambda_0>max': 'longest_song',
    'length<lambda_1>mean': 'longest_song_per_session',
    'length<lambda_1>sum': 'total_session_listening_time',
    'sessionIdcount': 'total_number_of_sessions',
    'registrationminmin': 'registration',
    'tsminmin': 'min_session_begin',
    'tsmaxmax': 'max_session_end',
    'ts<lambda_0>sum': 'total_session_length',
    'ts<lambda_0>mean': 'avg_session_length',
    'song<lambda>sum': 'number_of_songs',
    'level_freesummean': 'avg_num_free_interactions',
    'level_paidsummean': 'avg_num_paid_interactions',
    'method_GETsummean': 'avg_num_get_interactions',
    'method_PUTsummean': 'avg_num_put_interactions',
    'page_Aboutsummean': 'avg_num_about_visits',
    'page_Add Friendsummean': 'avg_num_addfriend_clicks',
    'page_Add to Playlistsummean': 'avg_num_addtoplaylist_clicks',
    'page_Downgradesummean': 'avg_num_downgrade_visits',
    'page_Errorsummean': 'avg_num_errors',
    'page_Helpsummean': 'avg_num_help_visits',
    'page_Homesummean': 'avg_num_home_visits',
    'page_Loginsummean': 'avg_num_login_visits',
    'page_Logoutsummean': 'avg_num_logout_visits',
    'page_NextSongsummean': 'avg_num_nextsong_clicks',
    'page_Roll Advertsummean': 'avg_num_roll_advert_visits',
    'page_Save Settingssummean': 'avg_num_savesettings_clicks',
    'page_Settingssummean': 'avg_num_settings_visits',
    'page_Submit Downgradesummean': 'avg_num_downgrade_clicks',
    'page_Submit Upgradesummean': 'avg_num_upgrade_clicks',
    'page_Submit Registrationsummean': 'avg_num_submitreg_clicks',
    'page_Registersummean': 'avg_num_register_visits',
    'page_Thumbs Downsummean': 'avg_num_thumbsdown_clicks',
    'page_Thumbs Upsummean': 'avg_num_thumbsup_clicks',
    'page_Upgradesummean': 'avg_num_upgrade_visits',
    'status_200summean': 'avg_status_200',
    'status_307summean': 'avg_status_307',
    'status_404summean': 'avg_status_404',
    'userAgent_0summean': 'avg_userAgent_0_interactions',
    'userAgent_1summean': 'avg_userAgent_1_interactions',
    'userAgent_2summean': 'avg_userAgent_2_interactions',
    'userAgent_3summean': 'avg_userAgent_3_interactions',
    'userAgent_4summean': 'avg_userAgent_4_interactions',
    'userAgent_5summean': 'avg_userAgent_5_interactions',
    'userAgent_6summean': 'avg_userAgent_6_interactions',
    'userAgent_7summean': 'avg_userAgent_7_interactions',
    'userAgent_8summean': 'avg_userAgent_8_interactions',
    'userAgent_9summean': 'avg_userAgent_9_interactions',
    'userAgent_10summean': 'avg_userAgent_10_interactions',
    'userAgent_11summean': 'avg_userAgent_11_interactions',
    'userAgent_12summean': 'avg_userAgent_12_interactions',
    'userAgent_13summean': 'avg_userAgent_13_interactions',
    'userAgent_14summean': 'avg_userAgent_14_interactions',
    'userAgent_15summean': 'avg_userAgent_15_interactions',
    'userAgent_16summean': 'avg_userAgent_16_interactions',
    'userAgent_17summean': 'avg_userAgent_17_interactions',
    'userAgent_18summean': 'avg_userAgent_18_interactions',
    'cancelledmaxmax': 'cancelled'
}

# rename columns
X_final.rename(rename_keys, axis=1, inplace=True)

# Additional feature engineering
X_final['listening_time_per_session'] = X_final['total_session_listening_time']/X_final['total_number_of_sessions']
X_final['avg_num_songs_per_session'] = X_final['number_of_songs']/X_final['total_number_of_sessions']
X_final['avg_song_length'] = X_final['total_session_listening_time']/X_final['number_of_songs']
X_final['time_since_joined'] = X_final['max_session_end'] - X_final['registration']
X_final['time_to_first_session'] = X_final['min_session_begin'] - X_final['registration']
X_final['avg_time_between_sessions'] = ((X_final['max_session_end'] - X_final['min_session_begin']) - X_final['total_session_length'])/(X_final['total_number_of_sessions']-1)

# fill in any null values after creating new features
X_final = X_final.fillna(0)

# drop bad userId
X_final = X_final[X_final['userId'] != '1261737']

# drop any columns that have all zeros
X_final = drop_extras(data=X_final)

# drop any last columns from the data (these were used in feature engineering 
# and aren't particularily useful anymore)
X_final = X_final.drop(['max_session_end', 'min_session_begin',
                        'total_session_length', 'registration',
                        'total_session_listening_time',
                        'number_of_songs', 'userId'], axis=1)

data = X_final.copy(deep=True)    

# Prepare the estimators and parameter dictionaries
lr_estimator = LogisticRegression(penalty='elasticnet', max_iter=10000,
                                  solver='saga')
lr_parameters = {
        'estimator__C': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        'estimator__l1_ratio': [0, 0.25, 0.4, 0.5, 0.6, 0.75, 1]
    }
gb_estimator = GradientBoostingClassifier(max_depth=5, min_samples_split=2)
gb_parameters = {
        'estimator__max_depth': [2, 5, 10, 20, 100],
        'estimator__min_samples_split': [2, 8, 16, 32]
    }

# Loop through each estimator type
target_var = 'cancelled'
for i, (estimator, params) in enumerate(zip([lr_estimator, gb_estimator],
                                            [lr_parameters, gb_parameters])):
    clf = data_pipeline(data, estimator, params, target_var)
    
    if i == 0:
        # Build coefficient matrix
        coef = pd.DataFrame(list(clf.best_estimator_.steps[1][1].coef_[0]),
                        index=X_train.columns,
                        columns=["Coefficients"])
        coef['Abs. Value Coefficients'] = np.abs(coef['Coefficients'])
        
        # sort coefficients, assign color to classes
        sorted_coef = coef.sort_values(['Abs. Value Coefficients'], ascending=True)
        sorted_coef['color'] = ['red' if x == -1 else 'blue' for x in np.sign(sorted_coef['Coefficients'])]
        
        # Plot the sorted coefficients
        plt.figure(figsize=(15, 15))
        plt.barh(range(len(sorted_coef.index)), sorted_coef['Abs. Value Coefficients'],
                 color=sorted_coef['color'])
        plt.title('Coefficient Rankings')
        plt.yticks(range(len(sorted_coef.index)), sorted_coef.index)
