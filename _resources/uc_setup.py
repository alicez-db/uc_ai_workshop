# Databricks notebook source
#Try to use the UC catalog "dbdemos" when possible. IF not will fallback to hive_metastore which means you should stop doing this workshop since it is meant for a Unity Catalog table & model
catalog = dbutils.widgets.get('catalog_name')
dbName = dbutils.widgets.get('schema_prefix')

current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
dbName = dbName + "_" + current_user.split('@')[0].replace(".", "_")
cloud_storage_path = f"/Users/{current_user}/demos/{dbName}"


print(f"using cloud storage path {cloud_storage_path} to store database tables")
print(f"using catalog.database {catalog}.{dbName}")

# COMMAND ----------

if catalog == "hive_metastore":
  print("This workshop is showing the AI features with Unity Catalog. You need to choose a Unity Catalog catalog name instead of the legacy hive_metastore.")

# COMMAND ----------

def use_and_create_db(catalog, dbName, cloud_storage_path = None):
  print(f"USE CATALOG `{catalog}`")
  spark.sql(f"USE CATALOG `{catalog}`")
  if cloud_storage_path is None or catalog not in ['hive_metastore', 'spark_catalog']:
    spark.sql(f"""create database if not exists `{dbName}` """)
  else:
    spark.sql(f"""create database if not exists `{dbName}` LOCATION '{cloud_storage_path}/tables' """)

def use_and_create_db(catalog, dbName, cloud_storage_path = None):
  print(f"USE CATALOG `{catalog}`")
  spark.sql(f"USE CATALOG `{catalog}`")
  if cloud_storage_path is None or catalog not in ['hive_metastore', 'spark_catalog']:
    spark.sql(f"""create database if not exists `{dbName}` """)
  else:
    spark.sql(f"""create database if not exists `{dbName}` LOCATION '{cloud_storage_path}/tables' """)

# COMMAND ----------

if catalog == "spark_catalog":
  catalog = "hive_metastore"
  
#If the catalog is defined, we force it to the given value and throw exception if not.
if len(catalog) > 0:
  current_catalog = spark.sql("select current_catalog()").collect()[0]['current_catalog()']
  if current_catalog != catalog:
    catalogs = [r['catalog'] for r in spark.sql("SHOW CATALOGS").collect()]
    if catalog not in catalogs and catalog not in ['hive_metastore', 'spark_catalog']:
      spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
  use_and_create_db(catalog, dbName)
else:
  #otherwise we'll try to setup the catalog to DBDEMOS and create the database here. If we can't we'll fallback to legacy hive_metastore
  print("Try to setup UC catalog")
  try:
    catalogs = [r['catalog'] for r in spark.sql("SHOW CATALOGS").collect()]
    if len(catalogs) == 1 and catalogs[0] in ['hive_metastore', 'spark_catalog']:
      print(f"UC doesn't appear to be enabled, will fallback to hive_metastore (spark_catalog)")
      catalog = "hive_metastore"
    else:
      if "dbdemos" not in catalogs:
        spark.sql("CREATE CATALOG IF NOT EXISTS dbdemos")
      catalog = "dbdemos"
    use_and_create_db(catalog, dbName)
  except Exception as e:
    print(f"error with catalog {e}, do you have permission or UC enabled? will fallback to hive_metastore")
    catalog = "hive_metastore"
    use_and_create_db(catalog, dbName)

# COMMAND ----------

import json

result = {"catalog_name": catalog, "dbName": dbName}
dbutils.notebook.exit(json.dumps(result))
