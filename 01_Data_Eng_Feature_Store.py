# Databricks notebook source
# MAGIC %md
# MAGIC # End-to-end Artificial Intelligence (AI) Workshop
# MAGIC ## Overview
# MAGIC This demo will show you how to do the full AI lifecycle using the Machine Learning, Data Governance, and Data Monitoring features on Databricks.
# MAGIC
# MAGIC ## Step 1: Define the Business Problem You Are Solving With AI
# MAGIC This demo is going to use a data table of customer information including whether they churned or not. We want to take a customer's information and predict if they are going to churn or not based on that information.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ![Demo Overview](https://raw.githubusercontent.com/databricks-end-to-end-ai-workshop/uc_ai_workshop/main/_resources/Demo_Overview.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Set up Unity Catalog and Define Cloud Storage Path
# MAGIC Start by naming what catalog and database/schema you want this demo to use in your own metadata organization. Additionally name the cloud storage location for tables for this demo. 

# COMMAND ----------

dbutils.widgets.text('catalog_name', 'main', 'Enter Catalog Name')
dbutils.widgets.text('schema_prefix', 'retail_e2e_ml_workshop', 'Enter Schema Prefix')

# COMMAND ----------

import json

catalog = dbutils.widgets.get('catalog_name')
schema_pre = dbutils.widgets.get('schema_prefix')

result = dbutils.notebook.run("./_resources/uc_setup", 60, {"catalog_name": catalog, "schema_prefix": schema_pre})
results = json.loads(result)
catalog = results["catalog_name"]
dbName = results["dbName"]
print(f"Using the catalog <{catalog}> and database <{dbName}> to store the data and model")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Ingest the raw CSV data to Delta Table
# MAGIC Here we are going to get the csv data and then write it into a delta table called "churn_bronze_customers"

# COMMAND ----------

import requests
from io import StringIO
import pandas as pd
import re

# dataset under apache license: https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/LICENSE
csv = requests.get("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv").text
df = pd.read_csv(StringIO(csv), sep=",")

# clean up column names
df.columns = [re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower().replace("__", "_") for name in df.columns]
df.columns = [re.sub(r'[\(\)]', '', name).lower() for name in df.columns]
df.columns = [re.sub(r'[ -]', '_', name).lower() for name in df.columns]
df = df.rename(columns = {'streaming_t_v': 'streaming_tv', 'customer_i_d': 'customer_id'})

spark.createDataFrame(df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{dbName}.churn_bronze_customers")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Exploratory Data Analysis
# MAGIC Using the Data Profile and Visualization built into Databricks you can further analyze and understanding the main characteristics of the data.
# MAGIC
# MAGIC Additionally you will see SQL can be used along with Python in the Databricks notebooks for further data investigation.

# COMMAND ----------

# DBTITLE 1,Read in Bronze Delta table using Spark
# read into spark
telcoDF = spark.table(f"{catalog}.{dbName}.churn_bronze_customers")
display(telcoDF)

# COMMAND ----------

# MAGIC %md
# MAGIC #### EDA in SQL
# MAGIC You can use SQL in the notebook cells as well as Python

# COMMAND ----------

# write SQL using the spark.sql syntax
full_table_name = f"{catalog}.{dbName}.churn_bronze_customers"
spark.sql("DROP TEMPORARY VARIABLE IF EXISTS fullTableName")
spark.sql(f"DECLARE VARIABLE fullTableName STRING DEFAULT '{full_table_name}'")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT churn, ROUND(AVG(tenure)) AS avg_tenure
# MAGIC FROM IDENTIFIER(system.session.fullTableName)
# MAGIC GROUP BY churn
# MAGIC ORDER BY avg_tenure DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ### You can use Pandas on Spark
# MAGIC
# MAGIC Because our Data Scientist team is familiar with Pandas, we'll use `Pandas on spark` to scale `pandas` code. The Pandas instructions will be converted in the spark engine under the hood and distributed at scale.
# MAGIC
# MAGIC *Note: Starting from `spark 3.2`, koalas is builtin and we can get an Pandas Dataframe using `pandas_api`.*

# COMMAND ----------

# convert our raw spark distributed dataframe into a distributed pandas dataframe
raw_df_pdf = telcoDF.pandas_api()

# perform the same aggregation we did in SQL using familiar Pandas syntax
avg_tenure_by_churn = raw_df_pdf.groupby("churn").mean().round().reset_index()[["churn", "monthly_charges"]].sort_values("monthly_charges", ascending = False)

# re-create the same plot using familiar pandas and matplotlib syntax distributed with Spark
avg_tenure_by_churn.plot(kind = "bar", x = "churn", y = "monthly_charges")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5: Feature Engineering and Preprocessing Data
# MAGIC Here we are going to turn the integer values of the tenure (in the unit of months) of a customer to a category string. Conversely, we are turning the categorical feature of type of customer contract to a number. Additionally we are going to adapt the total charges column to be a log form of the column for better scaling of the features. Lastly, we are adjusting the column names to be lower case and with underscores that are standardized across the columns to make it easier to call.

# COMMAND ----------

# bin tenure into decades
def bin_tenure(value):
  if value <= 10:
    return "newbs"
  elif value in range(10,20):
    return "10s"
  elif value in range(20,30):
    return "20s"
  elif value in range(30,40):
    return "30s"
  elif value in range(40,50):
    return "40s"
  elif value in range(50,60):
    return "50s"
  elif value in range(60,10000):
    return "60+"
  else:
    return "other"

# bin monthly charges into 7 buckets
def bin_monthly_charge(value):
  if value <= 20:
    return "cheapest"
  elif value in range(20,40):
    return "cheap"
  elif value in range(40,60):
    return "mid-cheap"
  elif value in range(60,80):
    return "middle"
  elif value in range(80,100):
    return "mid-expensive"
  elif value in range(100,110):
    return "expensive"
  else:
    return "most-expensive"
  
# take the log of features with skewed distributions
def log_transform(value):
  return float(np.log(value + 1)) # for 0 values

# COMMAND ----------

# MAGIC %md
# MAGIC #### Here is an example of how to clean data with the Pandas for Spark API

# COMMAND ----------

from databricks.feature_store import feature_table
import pyspark.pandas as ps
import numpy as np
import re

# create a function for feature engineering using the pandas API 
def compute_churn_features(data):
  
  # convert to a dataframe compatible with the pandas API
  data = data.pandas_api()

  data['tenure'] = data['tenure'].apply(bin_tenure)
  
  # first convert the 'total_charges' column to a float column
  data['total_charges'] = data['total_charges'].astype(float)
  data['total_charges'] = data['total_charges'].apply(log_transform)

  # then convert the monthly charges to a categorical feature
  data['monthly_charges'] = data['monthly_charges'].apply(bin_monthly_charge)

    # contract categorical -> duration in months
  data['contract'] = data['contract'].replace({
    "Month-to-month": 1,
    "One year": 12,
    "Two year": 24
    })  

  # feature selection - don't include label column in feature table
  
  data = data[['customer_id', 'gender', 'partner', 'dependents',
             'phone_service', 'multiple_lines', 'internet_service',
             'online_security', 'online_backup', 'device_protection',
             'tech_support', 'streaming_tv', 'streaming_movies',
             'contract', 'paperless_billing', 'payment_method',
             'total_charges','tenure','monthly_charges']]


  # drop missing values
  data = data.dropna()
  
  return data

# COMMAND ----------

# MAGIC %md
# MAGIC ####Here is the same feature engineering and cleaning shown above in Pyspark
# MAGIC
# MAGIC Above it was shown how to use the Pandas for Spark API to do feature engineering. However, below we will show the exact same steps but with the Pyspark dataframe so you are aware. This dataframe is not used in the rest of the code, however.

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import *
from pyspark.sql.functions import col, log, when

# bin tenure into decades
bin_tenure_udf = udf(bin_tenure, StringType())
# log transform the total charges column
log_transform_udf = udf(log_transform, FloatType())
# bin monthly charges into 7 buckets
bin_monthly_charge_udf = udf(bin_monthly_charge, StringType())

# create a function for feature engineering using the pandas API
churn_features_sdf = telcoDF \
    .withColumn('tenure', bin_tenure_udf(telcoDF['tenure'])) \
    .withColumn('monthly_charges', bin_monthly_charge_udf(telcoDF['monthly_charges'])) \
    .withColumn('total_charges', col('total_charges').cast('float')) \
    .withColumn('total_charges', log('total_charges')) \
    .withColumn('contract', 
                when(telcoDF['contract'] == "Month-to-month", 1)
                .when(telcoDF['contract'] == "One year", 12)
                .when(telcoDF['contract'] == "Two year", 24)
                .otherwise(telcoDF['contract'])) \
    .select('customer_id', 'gender', 'partner', 'dependents', 
            'phone_service', 'multiple_lines', 'internet_service', 
            'online_security', 'online_backup', 'device_protection', 
            'tech_support', 'streaming_tv', 'streaming_movies', 
            'contract', 'paperless_billing', 'payment_method', 
            'tenure', 'monthly_charges', 'total_charges') \
    .na.drop()

display(churn_features_sdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Now apply the feature engineering functions to the Pyspark dataframe

# COMMAND ----------

# perform feature engineering on the churn features using pyspark syntaxes
churn_features_df = compute_churn_features(telcoDF)
display(churn_features_df)

# COMMAND ----------

print("Dataframe shape before dropping duplicates ", churn_features_df.shape)
new_churn_features_df = churn_features_df.drop_duplicates(subset=['customer_id'])
print("Dataframe shape After dropping duplicates ", new_churn_features_df.shape)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Step 6: Write to Feature Store
# MAGIC
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/mlops-end2end-flow-feature-store.png" style="float:right" width="500" />
# MAGIC
# MAGIC Once our features are ready, we'll save them in Databricks Feature Store. Under the hood, features store are backed by a Delta Lake table.
# MAGIC
# MAGIC This will allow discoverability and reusability of our feature accross our organization, increasing team efficiency.
# MAGIC
# MAGIC Feature store will bring traceability and governance in our deployment, knowing which model is dependent of which set of features.
# MAGIC
# MAGIC Make sure you're using the "Machine Learning" menu to have access to your feature store using the UI.

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

try:
  # drop table if exists
  fs.drop_table(f'{catalog}.{dbName}.dbdemos_mlops_churn_features')
except:
  pass
# note: you might need to delete the FS table using the UI
churn_feature_table = fs.create_table(
  name=f'{catalog}.{dbName}.dbdemos_mlops_churn_features',
  primary_keys='customer_id',
  schema=churn_features_df.spark.schema(),
  description='These features are derived from the churn_bronze_customers table in the lakehouse.  We created dummy variables for the categorical columns, cleaned up their names, and added a boolean flag for whether the customer churned or not.  No aggregations were performed.'
)

fs.write_table(df=churn_features_df.to_spark(), name=f'{catalog}.{dbName}.dbdemos_mlops_churn_features', mode='overwrite')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Publishing the Feature Store
# MAGIC To model serve with Databricks Unity Catalog but also serve your feature data with the [Feature Serving capability](https://docs.databricks.com/en/machine-learning/feature-store/feature-function-serving.html). This is currently a gated public preview feature that is an option for some customers if requested. Otherwise, you can publish your features outside of Databricks as you see [here](https://docs.databricks.com/en/machine-learning/feature-store/publish-features.html) - Top right you can change the cloud type to get instructions for your cloud.
