# Databricks notebook source
# initialize widgets
dbutils.widgets.text('catalog_name', 'main', 'Enter Catalog Name')
dbutils.widgets.text('schema_prefix', 'retail_e2e_ml_workshop', 'Enter Schema Prefix')

# COMMAND ----------

import json

# get catalog name and database name
catalog = dbutils.widgets.get('catalog_name')
schema_pre = dbutils.widgets.get('schema_prefix')

result = dbutils.notebook.run("./_resources/uc_setup", 60, {"catalog_name": catalog, "schema_prefix": schema_pre})
results = json.loads(result)
catalog = results["catalog_name"]
dbName = results["dbName"]
print(f"Using the catalog <{catalog}> and database <{dbName}> to store the data and model")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Step 7: Pull features from feature table
# MAGIC You could continue with the dataframe that holds the data but we wanted to show a simple example of pulling data from a feature table.
# MAGIC
# MAGIC #### Retrieve the features from the feature table
# MAGIC You want to retrieve the data from your feature table and make the training data set for ML training.

# COMMAND ----------

from databricks.feature_store import FeatureLookup, FeatureStoreClient

fs = FeatureStoreClient()

# get our list of ID and labels
training_dataset_key = spark.table(f"{catalog}.{dbName}.churn_bronze_customers").select("customer_id", "churn")

model_feature_lookups = [
      FeatureLookup(
          table_name=f'{catalog}.{dbName}.dbdemos_mlops_churn_features',
          lookup_key=["customer_id"]
      )
]
# fs.create_training_set will look up features in model_feature_lookups with matched key from training_labels_df
training_set = fs.create_training_set(
    training_dataset_key, # joining the labels with our FeatureLookupTable
    feature_lookups = model_feature_lookups,
    exclude_columns=["customer_id"], # exclude features we won't use in our model
    label = "churn",
)

training_pd = training_set.load_df().toPandas()
 
display(training_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Note that there is an imbalanced data set here with Churn

# COMMAND ----------

# MAGIC %md
# MAGIC #### Balance our data set
# MAGIC Therefore we down sample our data set so the data can be more balanced before we go into training

# COMMAND ----------

sy_train = spark.createDataFrame(training_pd)

# reset the DataFrames for no churn (`dfn`) and churn (`dfy`)
dfn = sy_train.filter(sy_train.churn == "No")
dfy = sy_train.filter(sy_train.churn == "Yes")

# calculate summary metrics
N = sy_train.count()
y = dfy.count()
p = y/N

# create a more balanced training dataset
train_b = dfn.sample(False, p, seed = 42).union(dfy)

# print out metrics
print("Total count: %s, Churn cases count: %s, Proportion of churn cases: %s" % (N, y, p))
print("Balanced training dataset count: %s" % train_b.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 8: AutoML
# MAGIC #### Accelerating Churn model creation using Databricks Auto-ML
# MAGIC ##### A glass-box solution that empowers data teams without taking away control
# MAGIC
# MAGIC Databricks simplify model creation and MLOps. However, bootstraping new ML projects can still be long and inefficient. 
# MAGIC
# MAGIC Instead of creating the same boilerplate for each new project, Databricks Auto-ML can automatically generate state of the art models for Classifications, regression, and forecast.
# MAGIC
# MAGIC
# MAGIC <img width="1000" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/auto-ml-full.png"/>
# MAGIC
# MAGIC <img style="float: right" width="600" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/churn-auto-ml.png"/>
# MAGIC
# MAGIC Models can be directly deployed, or instead leverage generated notebooks to boostrap projects with best-practices, saving you weeks of efforts.
# MAGIC
# MAGIC ##### Using Databricks Auto ML with our Churn dataset
# MAGIC
# MAGIC Auto ML is available in the "Machine Learning" space. All we have to do is start a new Auto-ML experimentation and select the feature table we just created (`dbdemos_mlops_churn_features`)
# MAGIC
# MAGIC Our prediction target is the `churn` column.
# MAGIC
# MAGIC Click on Start, and Databricks will do the rest.
# MAGIC
# MAGIC While this is done using the UI, you can also leverage the [python API](https://docs.databricks.com/applications/machine-learning/automl.html#automl-python-api-1)

# COMMAND ----------

import databricks.automl

summary = databricks.automl.classify(train_b, target_col="churn", primary_metric="f1", timeout_minutes=5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 9: Log the best model

# COMMAND ----------

import mlflow

# creating sample input to be logged (do not include the live features in the schema as they'll be computed within the model)
df_sample = train_b.limit(10).toPandas()
x_sample = df_sample.drop(columns=["churn"])

# getting the model created by AutoML 
best_model = summary.best_trial.load_model()

# create a new run in the same experiment as our automl run.
with mlflow.start_run(run_name="best_fs_model", experiment_id=summary.experiment.experiment_id) as run:
  # use the feature store client to log our best model
  fs.log_model(
              model=best_model, # object of your model
              artifact_path="model", #name of the Artifact under MlFlow
              flavor=mlflow.sklearn, # flavour of the model (our LightGBM model has a SkLearn Flavour)
              training_set=training_set, # training data you used to train your model with AutoML
              input_example = x_sample # show a sample data set
              )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 10: Register the model with Unity Catalog to connect in the lineage with the Feature Store
# MAGIC Go to the "Catalog" on the left sidebar and traverse the metastore to find the feature table under the catalog and database/schema name we set in the beginning of the notebook.

# COMMAND ----------

# log ml models to unity catalog
mlflow.set_registry_uri("databricks-uc")

uc_model_name = f"{catalog}.{dbName}.e2e_ml_workshop_churn_model"
uc_registered_model = mlflow.register_model(model_uri=f"runs:/{run.info.run_id}/model", name=uc_model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 11: You can move the stages of models
# MAGIC In Databricks Model Registry you move the model with this command to different stages. For Unity Catalog, you move the model with the name of the model or with the alias.
# MAGIC
# MAGIC For example for moving a model to the production stage it could look like this if you have a "prod" catalog.
# MAGIC `mlflow.register_model(model_uri=f"runs:/{run.info.run_id}/model", name=f"prod.{dbName}.e2e_ml_workshop_churn_model")`
# MAGIC
# MAGIC or you  can create an alias and use that which we will show below.

# COMMAND ----------

from mlflow import MlflowClient
client = MlflowClient()

# moving model stages with an alias for the Unity Catalog registered model
version = uc_registered_model.version #grabbing the model version of the model you just registered
client.set_registered_model_alias(uc_model_name, "prod", version) 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 12: Serve the model or use it for batch inference
# MAGIC Here you will see. how to do batch inference as well as real-time inference with model serving.

# COMMAND ----------

# MAGIC %md
# MAGIC <img width="1000" src="https://github.com/databricks-end-to-end-ai-workshop/uc_ai_workshop/blob/main/_resources/Demo_Overview_Inference_Batch.png?raw=true"/>

# COMMAND ----------

model_uri = f"models:/{uc_model_name}@prod"
# batch predictions
batch_input_df = spark.table(f"{catalog}.{dbName}.churn_bronze_customers").select("customer_id")
with_predictions = fs.score_batch(model_uri, batch_input_df, result_type='string')
display(with_predictions.join(spark.table(f"{catalog}.{dbName}.churn_bronze_customers").select("customer_id", "churn"), on="customer_id"))

# COMMAND ----------

# MAGIC %md
# MAGIC <img width="1000" src="https://github.com/databricks-end-to-end-ai-workshop/uc_ai_workshop/blob/main/_resources/Demo_Overview_model_serving.png"/>

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Serving UI
# MAGIC
# MAGIC #Diagram on serving
# MAGIC
# MAGIC Here is an example of doing A/B testing with two models
# MAGIC
# MAGIC <img width="1000" src="https://www.databricks.com/en-website-assets/static/fa43cf6e8066a2b989f10fbf86120ba5/24143.png"/>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 13 Lakehouse Monitoring
# MAGIC
# MAGIC Databricks Lakehouse Monitoring allows teams to monitor their entire data pipelines — from data and features to ML models — without additional tools and complexity. Powered by Unity Catalog, it lets users uniquely ensure that their data and AI assets are high quality, accurate and reliable through deep insight into the lineage of their data and AI assets. The single, unified approach to monitoring enabled by lakehouse architecture makes it simple to diagnose errors, perform root cause analysis and find solutions.
# MAGIC
# MAGIC **Fully managed** so no time wasted managing infrastructure, calculating metrics, or building dashboards from scratch
# MAGIC
# MAGIC **Frictionless** with easy setup and out-of-the-box metrics and generated dashboards
# MAGIC
# MAGIC **Unified solution** for data and models for holistic understanding
# MAGIC
# MAGIC <img width="1000" src="https://www.databricks.com/en-website-assets/static/920ea66119b097a9d7555c1f3530bb09/22607.png"/>
