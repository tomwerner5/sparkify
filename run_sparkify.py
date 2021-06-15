#!/usr/bin/env python
# coding: utf-8

# # Sparkify Project
# 
# ### Step 1: Setup
# 
# #### Import Libraries

# In[1]:


import pyspark
from functools import reduce
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, Window, DataFrame
from pyspark.sql.types import IntegerType, StringType
from pyspark.sql.functions import (sum as Fsum, min as Fmin, max as Fmax,
                                   col, when, isnull, mean,
                                   count, desc, asc, regexp_replace,
                                   udf, trim, lower, avg)
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import (OneHotEncoder,
                                StringIndexer, VectorAssembler,
                                MinMaxScaler)
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from functools import partial
from typing import Iterable


# #### Start a Spark Session

# In[2]:


spark = SparkSession     .builder     .appName("Sparkify")     .getOrCreate()


# ### Step 2: Load and Clean Dataset

# Uncomment the desired path below. The mini dataset is provided in the repo. The larger dataset can be loaded from an AWS S3 bucket.

# In[3]:


get_ipython().run_cell_magic('time', '', 'path = "mini_sparkify_event_data.json"\n#path = "s3n://udacity-dsnd/sparkify/sparkify_event_data.json"\nuser_log = spark.read.json(path)')


# The following cells remove missing values and other formatting issues discovered in the data.

# In[4]:


# Check for missing values. Also checks for blank strings
user_log.select([count(when(col(c) == "", c).when(isnull(c), c)).alias(c) for c in user_log.columns]).toPandas()


# In[5]:


# Check the size of the dataset
user_log.toPandas().shape


# In[6]:


# There are missing values in userid and sessionid, so need to drop.
# Blank userIds are disguised as empty strings
user_log = user_log.dropna(how = "any", subset = ["userId", "sessionId"])
user_log = user_log.filter(user_log.userId != '')


# In[7]:


# Gut check to see what null values are left, and whether the
# previous step worked
user_log.select([count(when(col(c) == "", c).when(isnull(c), c)).alias(c) for c in user_log.columns]).toPandas()


# In[8]:


# Adjust the timestamp format from miliseconds to seconds for
# convenience
user_log = user_log.withColumn('ts', col('ts')/1000)
user_log = user_log.withColumn('registration', col('registration')/1000)


# In[9]:


# Check the distribution of auth against our target variable (cancellations)
user_log.select(['auth', 'page']).groupBy(['auth', 'page']).agg(count('page')).show()


# In[10]:


# These columns are dropped because they are uhnlikely to provide
# much value. For example, firstname/lastname shouldn't have
# much to do with users deciding to cancel, and auth is perfectly
# correlated with our target variable, so we don't want to include it
user_log = user_log.drop("auth", "firstName", "lastName")


# In[11]:


# artist and userAgent have a lot of unique values. The following lines
# try to reduce the number of characters for each.

# Filter rows with regex (find rows that match expression)
user_log = user_log.withColumn('artist', regexp_replace('artist', r'[^0-9a-zA-Z]', ' '))
user_log = user_log.withColumn('userAgent', regexp_replace('userAgent', r'[^a-zA-Z]', ' '))
user_log = user_log.withColumn('userAgent', trim(lower(regexp_replace('userAgent', r'\s+', ' '))))
user_log = user_log.withColumn('userAgent', regexp_replace('userAgent', r'\s[a-z]\s', ' '))


# In[12]:


# Keep a copy of the original column list for convenience
original_col_list = list(user_log.columns)


# For the categorical columns, it is likely to be helpful to have one-hot encoded columns. To do this in spark, you need to create a string index for each column, and then use that index to instantiate the OneHotEncoder class for each indexed column. 

# In[13]:


## One hot encode column variables

# Create string index for each column to be encoded
onehot_cols = ['gender', 'level', 'method', 'userAgent', 'page', 'status']
indexers = [
    StringIndexer(inputCol=column, outputCol=column+"_indexed")
    for column in onehot_cols
]

# Use the index column to prepare the one-hot encoder
encoders = [
    OneHotEncoder(
        inputCol=indexer.getOutputCol(),
        outputCol=indexer.getOutputCol()+"_encoded",
        dropLast=False
    ) 
    for indexer in indexers
]

# Generate a vector of encoded values using VectorAssembler for
# each row
assembler = [VectorAssembler(
    inputCols=[encoder.getOutputCol() for encoder in encoders],
    outputCol="onehot_features"
)]

# Run the string index, encoder, and assembler as a Pipeline,
# save the output back to the spark dataframe
onehot_pipeline = Pipeline(stages=indexers + encoders + assembler)
onehot_pipeline = onehot_pipeline.fit(user_log)
user_log = onehot_pipeline.transform(user_log)

# Generate the list of column names for convenience later (using
# VectorAssembler, the column names are not saved to the dataframe
# and need to be recovered)
#
# userAgent is renamed here because each value is a really long string
# and this makes the column name more convenient
expanded_oh_names = []
for i, column in enumerate(onehot_cols):
    for j, label in enumerate(onehot_pipeline.stages[i].labels):
        if column == 'userAgent':
            expanded_oh_names.append(column + '_' + str(j))
        else:
            expanded_oh_names.append(column + '_' + label)

# Generate list of columns to keep in the dataframe. For most purposes,
# this would be a list of the columns that were not encoded.
# If any of the columns in the list are one-hot encoded, this list
# will keep the original, non-encoded column in the dataframe
no_hot_cols = [c for c in original_col_list if c not in onehot_cols]
#no_hot_cols = ["artist", "gender", "itemInSession", "length",
#    "level", "location", "method", "page", "registration", "sessionId",
#    "song", "ts", "userAgent", "userId"]


# create a new, expanded column list.
all_cols = no_hot_cols + expanded_oh_names

def extract(row, column_list):
    '''
    For each row, extract the one-hot values in the row vector (from
    VectorAssembler), along with any columns listed in column_list.
    
    Returns a tuple of values.
    '''
    col_tup = tuple(row[column] for column in column_list)
    return col_tup + tuple(row.onehot_features.toArray().tolist())

user_log = user_log.rdd.map(partial(extract, column_list=no_hot_cols)).toDF(all_cols)


# ### Step 3: Define Churn and Explore

# Below are some helpful functions I could have used to define churn. This isn't necessary since the variable is one-hot encoded for us in the previous step. Instead, I just rename an equivalent column.

# In[14]:


## function to flag downgrade
#flag_downgrade_event = udf(lambda x: 1 if x == "Cancellation Confirmation" else 0, IntegerType())
## flag event
#user_log = user_log.withColumn("cancelled", flag_downgrade_event("page"))

## Window 
#windowval = Window.partitionBy("userId").orderBy(desc("ts")).rangeBetween(Window.unboundedPreceding, 0)
#user_log_valid = user_log.withColumn("phase", Fsum("cancelled").over(windowval))


# In[15]:


user_log = user_log.withColumnRenamed('page_Cancellation Confirmation', 'cancelled')


# Next, the data is prepped for some exploration and plotting.

# In[16]:


# Convert to pandas for easy plotting/exploring
id_group = user_log.groupBy(["userId", "sessionId"])                     .agg(
                        Fmax('itemInSession').alias('num_items_in_session'),
                        Fmax('length').alias('longest_song_in_session'),
                        Fsum('length').alias('session_listening_time'),
                        count('song').alias('number_of_songs'),
                        Fmin('ts').alias('session_begin'),
                        Fmax('ts').alias('session_end'),
                        (Fmax('ts')-Fmin('ts')).alias('session_length'),
                        Fmax('gender_F').alias('gender_F'),
                        Fsum('page_Error').alias('num_errors'),
                        Fmax('cancelled').alias('cancelled')
                    ) \
                    .groupBy('userId') \
                    .agg(
                        when(count('sessionId') <= 1, 0)
                        .otherwise(((Fmax('session_end')-Fmin('session_begin'))
                                    -Fsum('session_length'))/(count('sessionId')-1))
                        .alias('avg_time_between_sessions'),
                        avg('session_length').alias('avg_session_length'),
                        avg('gender_F').alias('avg_gender_F'),
                        (Fsum('session_listening_time')/count('sessionId')).alias('listening_time_per_session'),
                        count('sessionId').alias('total_number_of_sessions'),
                        avg('num_errors').alias('avg_num_errors'),
                        Fmax('cancelled').alias('cancelled')
                    ).toPandas()


# In[17]:


def group_differences(data, target_val, group_vals):
    '''
    Groups the target variable by a grouping variable, and returns the mean values
    and error for each group. 
    
    data: dataframe
        The dataframe of data
    target_val : str
        The target column of the dataset
    group_val : str or list
        The column to create the groups for comparison
    '''
    from scipy.stats import t
    ci_stand_error = lambda x: (x.std(ddof=1)/(np.sqrt(x.shape[0])))*t.ppf(0.975, df=x.shape[0])
    
    if isinstance(target_val, str):
        target_val = [target_val]
        
    concats = []
    for tv in target_val:
        new_group = data.groupby(group_vals, as_index=True).agg(
                          {tv: ['mean', ci_stand_error]})
        new_group.columns.set_levels([[tv], ['Mean', 'Error']],
                                  level=None, inplace=True)
        concats.append(new_group)

    grouped_df = pd.concat(concats, axis=1)
    return grouped_df


# In[18]:


sess_info = group_differences(id_group, target_val=['avg_time_between_sessions', 'avg_session_length'],
                              group_vals=['cancelled'])/3600

err_vals = sess_info.T.xs('Error', level=1, drop_level=False).T.values
sess_info.T.xs('Mean', level=1, drop_level=False).plot.bar(yerr=err_vals,
                                                           title='Length and Frequency of Streaming Sessions',
                                                           #xlabel='Data Partitions',
                                                           #ylabel='Mean'
                                                          )
plt.xlabel('Data Partitions')
plt.ylabel('Mean Hours')
plt.xticks([0, 1], ['Avg. Time Between Sessions', 'Avg. Session Length'], rotation=0,
          ha='center')


# In[19]:


sess_info = group_differences(id_group, target_val=['avg_session_length'],
                              group_vals=['cancelled'])/3600

err_vals = sess_info.T.xs('Error', level=1, drop_level=False).T.values
sess_info.T.xs('Mean', level=1, drop_level=False).plot.bar(yerr=err_vals,
                                                           title='Length of Streaming Sessions',
                                                           #xlabel='Data Partitions',
                                                           #ylabel='Mean'
                                                          )
#plt.xlabel('Data Partitions')
plt.ylabel('Mean Hours')
plt.xticks([0], ['Avg. Session Length'], rotation=0,
          ha='center')


# In[20]:


#group_differences(id_group, target_val='avg_time_between_sessions', group_vals=['avg_gender_F', 'cancelled'])


# In[21]:


group_differences(id_group, target_val='avg_session_length', group_vals=['cancelled'])/3600


# In[22]:


group_differences(id_group, target_val='listening_time_per_session', group_vals=['cancelled'])/3600


# In[23]:


group_differences(id_group, target_val='total_number_of_sessions', group_vals=['cancelled'])


# In[24]:


sess_info = group_differences(id_group, target_val='total_number_of_sessions',
                              group_vals=['cancelled'])

err_vals = sess_info.T.xs('Error', level=1, drop_level=False).T.values
sess_info.T.xs('Mean', level=1, drop_level=False).plot.bar(yerr=err_vals,
                                                           title='Number of Sessions Among Users',
                                                           #xlabel='Data Partitions',
                                                           #ylabel='Mean'
                                                          )
#plt.xlabel('Data Partitions')
plt.ylabel('Count')
plt.xticks([0], ['Avg. Sessions per User'], rotation=0,
          ha='center')


# In[25]:


X = id_group['avg_num_errors']
Y = id_group['avg_time_between_sessions']/3600

plt.scatter(X, Y)
plt.title('Number of Errors Recieved and Time Between Sessions')
plt.xlabel('Average Number of Errors')
plt.ylabel('Average Time Between Sessions (hours)')


# ### Step 4: Feature Engineering
# 
# The data is consolidated to a one-row-per-user format by using two consecutive groupby functions. The first groupby condenses the session data for each user, so each user has summary statistics for each session they had. Then, each session is grouped together so that each user has summary statistics about their total session history. The data was grouped using Spark SQL functionality. The full query is shown in the `SqlFeatureEngineer` class.

# In[26]:


# generate the arbitrary list of userAgent column names for passing
# to the SqlFeatureEngineer class
agents = [name for name in expanded_oh_names if "userAgent" in name]


# In[27]:


class SqlFeatureEngineer(Transformer):
    """
    Custom Transform to feature engineer using a SQL query
    """

    def __init__(self, table_name: str, agents: Iterable[str]):
        super(SqlFeatureEngineer, self).__init__()
        self.table_name = table_name
        
        sum_UA = "".join("\n        ,SUM(%s) as num_%s_interactions" % (agent, agent) for agent in agents)
        avg_UA = "".join("\n    ,AVG(sess.num_%s_interactions) as avg_%s_interactions" % (agent, agent) for agent in agents)

        session_query_select = """SELECT 
            sess.userId
            ,AVG(sess.num_items_in_session) as avg_num_items_in_session
            ,MAX(sess.longest_song_in_session) as longest_song
            ,AVG(sess.longest_song_in_session) as longest_song_per_session
            ,COUNT(sess.sessionId) as total_number_of_sessions
            ,SUM(sess.session_listening_time)/COUNT(sess.sessionId) as listening_time_per_session
            ,SUM(sess.number_of_songs)/COUNT(sess.sessionId) as avg_number_of_songs_per_session
            ,SUM(sess.session_listening_time)/SUM(sess.number_of_songs) as avg_song_length
            ,AVG(sess.session_length) as avg_session_length
            ,MAX(sess.session_end)-MIN(sess.registration) as time_since_joined
            ,MIN(sess.session_begin)-MIN(sess.registration) as time_to_first_session 
            ,CASE
                WHEN COUNT(sess.sessionId) <= 1 THEN 0
                ELSE ((MAX(sess.session_end)-MIN(sess.session_begin)) - SUM(sess.session_length))/(COUNT(sess.sessionId)-1)
            END as avg_time_between_sessions
            ,AVG(sess.gender_F) as avg_gender_F
            ,AVG(sess.gender_M) as avg_gender_M
            ,AVG(sess.status_200) as avg_status_200
            ,AVG(sess.status_307) as avg_status_307
            ,AVG(sess.status_404) as avg_status_404
            ,AVG(sess.num_paid_interactions) as avg_num_paid_interactions
            ,AVG(sess.num_free_interactions) as avg_num_free_interactions
            ,AVG(sess.num_put_interactions) as avg_num_put_interactions
            ,AVG(sess.num_get_interactions) as avg_num_get_interactions
            ,AVG(sess.num_nextsong_clicks) as avg_num_nextsong_clicks
            ,AVG(sess.num_thumbsup_clicks) as avg_num_thumbsup_clicks
            ,AVG(sess.num_home_visits) as avg_num_home_visits
            ,AVG(sess.num_addtoplaylist_clicks) as avg_num_addtoplaylist_clicks
            ,AVG(sess.num_addfriend_clicks) as avg_num_addfriend_clicks
            ,AVG(sess.num_rolladvert_visits) as avg_num_rolladvert_visits
            ,AVG(sess.num_logout_visits) as avg_num_logout_visits
            ,AVG(sess.num_thumbsdown_clicks) as avg_num_thumbsdown_clicks
            ,AVG(sess.num_downgrade_visits) as avg_num_downgrade_visits
            ,AVG(sess.num_settings_visits) as avg_num_settings_visits
            ,AVG(sess.num_help_visits) as avg_num_help_visits
            ,AVG(sess.num_upgrade_visits) as avg_num_upgrade_visits
            ,AVG(sess.num_about_visits) as avg_num_about_visits
            ,AVG(sess.num_savesettings_clicks) as avg_num_savesettings_clicks
            ,AVG(sess.num_errors) as avg_num_errors
            ,AVG(sess.num_upgrades) as avg_num_upgrades
            ,AVG(sess.num_downgrades) as avg_num_downgrades
            --,AVG(sess.num_cancel_visits) as avg_num_cancel_visits
            ,MAX(sess.cancelled) as cancelled"""

        session_query_from = """
        FROM (
            SELECT 
                userId
                ,sessionId
                ,MAX(itemInSession) as num_items_in_session
                ,MAX(length) as longest_song_in_session
                ,SUM(length) as session_listening_time
                ,COUNT(song) as number_of_songs
                ,min(ts) as session_begin
                ,max(ts) as session_end
                ,max(ts) - min(ts) as session_length
                ,MIN(registration) as registration
                ,MAX(cancelled) as cancelled
                ,MAX(gender_F) as gender_F
                ,MAX(gender_M) as gender_M
                ,SUM(status_200) as status_200
                ,SUM(status_307) as status_307
                ,SUM(status_404) as status_404
                ,SUM(level_paid) as num_paid_interactions
                ,SUM(level_free) as num_free_interactions
                ,SUM(method_PUT) as num_put_interactions
                ,SUM(method_GET) as num_get_interactions
                ,SUM(page_NextSong) as num_nextsong_clicks
                ,SUM(`page_Thumbs Up`) as num_thumbsup_clicks
                ,SUM(page_Home) as num_home_visits
                ,SUM(`page_Add to Playlist`) as num_addtoplaylist_clicks
                ,SUM(`page_Add Friend`) as num_addfriend_clicks
                ,SUM(`page_Roll Advert`) as num_rolladvert_visits
                ,SUM(page_Logout) as num_logout_visits
                ,SUM(`page_Thumbs Down`) as num_thumbsdown_clicks
                ,SUM(page_Downgrade) as num_downgrade_visits
                ,SUM(page_Settings) as num_settings_visits
                ,SUM(page_Help) as num_help_visits
                ,SUM(page_Upgrade) as num_upgrade_visits
                ,SUM(page_About) as num_about_visits
                ,SUM(`page_Save Settings`) as num_savesettings_clicks
                ,SUM(page_Error) as num_errors
                ,SUM(`page_Submit Upgrade`) as num_upgrades
                ,SUM(`page_Submit Downgrade`) as num_downgrades
                ,SUM(page_Cancel) as num_cancel_visits""" 

        session_query = session_query_select + avg_UA + session_query_from + sum_UA + """
            FROM """ + self.table_name + """
            GROUP BY
                userId,
                sessionId
        ) as sess
        GROUP BY
            userId
        """
        
        self.query = session_query

    def _transform(self, df: DataFrame) -> DataFrame:
        df.createOrReplaceTempView(self.table_name)
        summary = spark.sql(self.query)
        return summary


# In[28]:


get_ipython().run_cell_magic('time', '', "table_name = 'user_log_temp'\nuser_summary = SqlFeatureEngineer(table_name, agents).transform(user_log)")


# In[29]:


get_ipython().run_cell_magic('time', '', '# Create correlation plot for numerical variables\ncorr = user_summary.toPandas().corr()\nmask = np.zeros_like(corr)\nmask[np.triu_indices_from(mask)] = True\nsns.set(rc={\'figure.figsize\':(12, 12)})\nwith sns.axes_style("white"):\n    sns.heatmap(corr, annot=False, square=True, mask=mask)\n    plt.title("Correlation Matrix")')


# In[31]:


get_ipython().run_cell_magic('time', '', '# create a list of columns to min-max scale. Make sure to exclude\n# the userId column and the target variable\ncolumns_to_scale = [column for column in user_summary.columns if column != \'userId\' and column != \'cancelled\']\n\n# min-max scaler\nassembler = VectorAssembler(inputCols=columns_to_scale, outputCol="features")\ntransformed = assembler.transform(user_summary)\nscaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")\nscalerModel =  scaler.fit(transformed.select("features"))\nuser_scaled = scalerModel.transform(transformed)')


# ### Step 5: Modeling
# 
# Modeling is performed by first separating the data into a training set and a test set. The training set is then modeled using a cross-validated grid search over a logistic regression model. The userId variable is dropped from the data before modeling. The min-maxed features are used in the model instead of the variables at their original scale. The best model is selected based on the highest F1 score. 
# 
# Once the best-fit model has been selected, the model is trained on the entire training set and a final evaluation is performed. 

# In[32]:


# drop userId
user_scaled = user_scaled.drop('userId')


# In[33]:


# Split into training and test data
(trainingData, testData) = user_scaled.randomSplit([0.8, 0.2], seed=42)


# In[34]:


# Check base rate
trainingData.select(mean(trainingData.cancelled)).show()
testData.select(mean(testData.cancelled)).show()


# In[35]:


get_ipython().run_cell_magic('time', '', '# Grid Search\ngbt = GBTClassifier(labelCol="cancelled", featuresCol="scaledFeatures",\n                    maxIter=100)\nlogreg = LogisticRegression(labelCol="cancelled", featuresCol="scaledFeatures",\n                            maxIter=100, regParam=0.01, elasticNetParam=0.5)\nevaluator = MulticlassClassificationEvaluator(\n    labelCol="cancelled", predictionCol="prediction",\n    metricName="f1")\n\npipeline_lr = Pipeline(stages=[logreg])\npipeline_gbt = Pipeline(stages=[gbt])\n\nparamGrid_lr = ParamGridBuilder() \\\n                .addGrid(logreg.elasticNetParam, [0]) \\\n                .addGrid(logreg.regParam, [0.01, 0.1]) \\\n                .build()\n    \ncrossval_lr = CrossValidator(estimator=logreg,\n                             estimatorParamMaps=paramGrid_lr,\n                             evaluator=evaluator,\n                             #numFolds=4)\n                             numFolds=2)')


# In[36]:


get_ipython().run_cell_magic('time', '', 'cvLRModel = crossval_lr.fit(trainingData)')


# In[37]:


bestLRModel = cvLRModel.bestModel


# In[38]:


get_ipython().run_cell_magic('time', '', 'predictions_lr = bestLRModel.transform(testData)')


# In[39]:


get_ipython().run_cell_magic('time', '', '\nf1_lr = evaluator.evaluate(predictions_lr)\n\nacc_lr = evaluator.setMetricName("accuracy").evaluate(predictions_lr)\n\nprint("LogReg Test Error = %g -- F1: %g" % (1.0 - acc_lr, f1_lr))')

