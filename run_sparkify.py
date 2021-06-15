# # Sparkify Project
# 
# ### Step 1: Setup
# 
# #### Import Libraries

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
spark = SparkSession.builder.appName("Sparkify").getOrCreate()


# ### Step 2: Load and Clean Dataset

# Load the dataset from the AWS S3 bucket.
path = "s3n://udacity-dsnd/sparkify/sparkify_event_data.json"
user_log = spark.read.json(path)

# There are missing values in userid and sessionid, so need to drop.
# Blank userIds are disguised as empty strings
user_log = user_log.dropna(how = "any", subset = ["userId", "sessionId"])
user_log = user_log.filter(user_log.userId != '')

# Adjust the timestamp format from miliseconds to seconds for
# convenience
user_log = user_log.withColumn('ts', col('ts')/1000)
user_log = user_log.withColumn('registration', col('registration')/1000)

# These columns are dropped because they are unlikely to provide
# much value. For example, firstname/lastname shouldn't have
# much to do with users deciding to cancel, and auth is perfectly
# correlated with our target variable, so we don't want to include it
user_log = user_log.drop("auth", "firstName", "lastName")

# artist and userAgent have a lot of unique values. The following lines
# try to reduce the number of characters for each.

# Filter rows with regex (find rows that match expression)
#user_log = user_log.withColumn('artist', regexp_replace('artist', r'[^0-9a-zA-Z]', ' '))
user_log = user_log.withColumn('userAgent', regexp_replace('userAgent', r'[^a-zA-Z]', ' '))
user_log = user_log.withColumn('userAgent', trim(lower(regexp_replace('userAgent', r'\s+', ' '))))
user_log = user_log.withColumn('userAgent', regexp_replace('userAgent', r'\s[a-z]\s', ' '))

# Keep a copy of the original column list for convenience
original_col_list = list(user_log.columns)

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

user_log = user_log.withColumnRenamed('page_Cancellation Confirmation', 'cancelled')

# ### Step 4: Feature Engineering

# generate the arbitrary list of userAgent column names for passing
# to the SqlFeatureEngineer class
agents = [name for name in expanded_oh_names if "userAgent" in name]

table_name = 'user_log_temp'
user_summary = SqlFeatureEngineer(table_name, agents).transform(user_log)

# create a list of columns to min-max scale. Make sure to exclude
# the userId column and the target variable
columns_to_scale = [column for column in user_summary.columns if column != 'userId' and column != 'cancelled']

# min-max scaler
assembler = VectorAssembler(inputCols=columns_to_scale, outputCol="features")
transformed = assembler.transform(user_summary)
scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
scalerModel =  scaler.fit(transformed.select("features"))
user_scaled = scalerModel.transform(transformed)


# ### Step 5: Modeling

# drop userId
user_scaled = user_scaled.drop('userId')

# Split into training and test data
(trainingData, testData) = user_scaled.randomSplit([0.8, 0.2], seed=42)

# Check base rate
trainingData.select(mean(trainingData.cancelled)).show()
testData.select(mean(testData.cancelled)).show()

# Grid Search
logreg = LogisticRegression(labelCol="cancelled", featuresCol="scaledFeatures",
                            maxIter=100, regParam=0.01, elasticNetParam=0.5)
evaluator = MulticlassClassificationEvaluator(
  labelCol="cancelled", predictionCol="prediction",
  metricName="f1")

pipeline_lr = Pipeline(stages=[logreg])

paramGrid_lr = ParamGridBuilder().addGrid(logreg.elasticNetParam, [0]).addGrid(logreg.regParam, [0.01, 0.1]).build()
crossval_lr = CrossValidator(estimator=logreg,
                             estimatorParamMaps=paramGrid_lr,
                             evaluator=evaluator,
                             #numFolds=4)
                             numFolds=2)

cvLRModel = crossval_lr.fit(trainingData)
bestLRModel = cvLRModel.bestModel

predictions_lr = bestLRModel.transform(testData)

f1_lr = evaluator.evaluate(predictions_lr)
acc_lr = evaluator.setMetricName("accuracy").evaluate(predictions_lr)

print("LogReg Test Error = %g -- F1: %g" % (1.0 - acc_lr, f1_lr))

coef = pd.DataFrame(list(bestLRModel.coefficients),
                    index=columns_to_scale,
                    columns=["Coefficients"])
coef['Abs. Value Coefficients'] = np.abs(coef['Coefficients'])
print("\n\n", coef.sort_values(['Abs. Value Coefficients'], ascending=False))

sorted_coef = coef.sort_values(['Abs. Value Coefficients'], ascending=True)
plt.barh(range(len(sorted_coef.index)), sorted_coef['Abs. Value Coefficients'],
         color=sorted_coef['color'])
plt.title('Coefficient Rankings')
plt.yticks(range(len(sorted_coef.index)), sorted_coef.index)
plt.savefig("coef_rank.png")
