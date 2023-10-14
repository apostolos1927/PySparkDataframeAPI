# Databricks notebook source
# MAGIC %md
# MAGIC ####Create dataframe from RDD

# COMMAND ----------

dept = [("Finance",10),("Marketing",20),("Sales",30),("IT",40)]
rdd = sc.parallelize(dept)
columns = ["DEPARTMENT","VALUE"]
df_rdd = rdd.toDF(columns)
df_rdd.show(truncate=False)

# COMMAND ----------

type(df_rdd)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Read csv data from Filestore

# COMMAND ----------

df1 = (spark.read.format("csv")
      .options(header='True', inferSchema='True', delimiter=',') 
      .load("/FileStore/testdata.csv"))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Different ways to display dataframe

# COMMAND ----------

df1.show()

# COMMAND ----------

df1.head(5)

# COMMAND ----------

df1.first()

# COMMAND ----------

df1.display()

# COMMAND ----------

df1.take(2)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Define data types manually by providing schema

# COMMAND ----------

from pyspark.sql.types import StructType,IntegerType,StringType
schema = StructType() \
      .add("ID",IntegerType(),True) \
      .add("AGE",IntegerType(),True) \
      .add("CITY",StringType(),True) 

# COMMAND ----------

df_with_schema = spark.read.format("csv") \
      .option("header", True) \
      .option("delimiter", ",") \
      .schema(schema) \
      .load("/FileStore/testdata.csv")
display(df_with_schema)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Create temporary View and query the data

# COMMAND ----------

# MAGIC %md
# MAGIC ######createOrReplaceTempView registers a DataFrame as a table that you can query using SQL (lasts as long as the SparkSession). This method does not allow you to achieve any performance improvement like cache/persist

# COMMAND ----------

df_with_schema.createOrReplaceTempView("TestView")

# COMMAND ----------

df_query = spark.sql("select * from TestView")
df_query.show()

# COMMAND ----------

df_with_schema.where('AGE>40').orderBy('ID').show()

# COMMAND ----------

spark.sql('select * from TestView where AGE>40 order by ID').show()

# COMMAND ----------

df_temp = spark.sql('select * from TestView where AGE>40 order by ID')

# COMMAND ----------

df_temp.write.saveAsTable("Demo")

# COMMAND ----------

spark.sql('select * from Demo').display()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Convert Spark Dataframe into Pandas dataframe that runs only in the driver node. Pandas has many useful libraries which cannot run in Spark distributed architecture. 

# COMMAND ----------

pandas_df = df_with_schema.toPandas()
pandas_df

# COMMAND ----------

type(pandas_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Use Cache or Persist for better performance

# COMMAND ----------

# MAGIC %md
# MAGIC ######Cache and persist saves the Dataframe in memory, making it faster for access in the subsequent actions. No need to re-execute the same transformations as it has already been cached and it is available.
# MAGIC

# COMMAND ----------

df_with_schema.cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Rename column in PySpark Dataframe

# COMMAND ----------

df1 = df1.withColumnRenamed("CITY","TOWN")#.printSchema()
df1.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Create a new column with a constant value using the lit pyspark sql function

# COMMAND ----------

import pyspark.sql.functions as F
df1 = df1.select('*',F.lit("1").alias("lit_value"))
df1.show()

# COMMAND ----------

df1 = df1.withColumn("AGE",F.col("AGE").cast("STRING"))
df1.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Create new column based on another column

# COMMAND ----------

df1.withColumn("NewAgeColumn",F.col("AGE") -1).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Create a new constant column

# COMMAND ----------

df1.withColumn("COUNTRY",F.lit("USA")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Join tables

# COMMAND ----------

df2 = (spark.read.format("csv")
      .options(header='True', inferSchema='True', delimiter=',') 
      .load("/FileStore/testdata2.csv"))
df2.show()

# COMMAND ----------

df1.join(df2, df1.ID == df2.ID, 'INNER') \
    .select(df1.ID,df1.AGE, df2.SALARY, F.round("SALARY", 0)) \
    .withColumnRenamed('round(SALARY, 0)','RoundedSalary') \
    .sort(F.desc("SALARY")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Broadcast Join for improved performance in the joins

# COMMAND ----------

# MAGIC %md
# MAGIC ######PySpark splits the data into different nodes for parallel processing, when you have two DataFrames, the data from both are distributed across multiple nodes in the cluster so, when you perform traditional join, PySpark is required to shuffle the data.
# MAGIC
# MAGIC ######With broadcast join, PySpark broadcast the smaller DataFrame to all worker nodes that keep this DataFrame in memory while the larger DataFrame is distributed across all executors so that PySpark can perform a join without shuffling
# MAGIC

# COMMAND ----------

df1.join(F.broadcast(df2), df1.ID == df2.ID, 'left').select('*').show(truncate=False)

# COMMAND ----------

df1.join(F.broadcast(df2), df1.ID == df2.ID, 'left') \
    .select('*') \
    .explain(extended=True) #logical and physical plan

# COMMAND ----------

# MAGIC %md
# MAGIC ####Aggregation in Dataframe

# COMMAND ----------

df2.groupBy("salary") \
    .agg(F.count("*").alias("count")) \
    .show(truncate=False)

# COMMAND ----------

df2.groupBy("occupation") \
    .agg(F.max("salary").alias("max_salary"),F.avg("salary").alias("avg_salary")) \
    .where(F.col("avg_salary") >= 45) \
    .show(truncate=False)

# COMMAND ----------

df2.groupBy("occupation") \
    .agg({'salary': 'avg', '*': 'count'}) \
    .withColumnRenamed('avg(salary)','average_salary') \
    .filter('average_salary>50') \
    .show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Merging dataframes with Union

# COMMAND ----------

df3 = (spark.read.format("csv")
      .options(header='True', inferSchema='True', delimiter=',') 
      .load("/FileStore/testdata2.csv"))
df3.show()

# COMMAND ----------

unionDF = df2.union(df3).distinct()
unionDF.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Define your own function with UDF
# MAGIC

# COMMAND ----------

def upperCase(txt):
    return txt.upper()

# COMMAND ----------

upperCaseUDF = F.udf(lambda x:upperCase(x),StringType()) 

# COMMAND ----------

df2.withColumn("OCCUPATION", upperCaseUDF(F.col("OCCUPATION"))).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Use RDD operations like map by converting a dataframe to rdd and back to dataframe

# COMMAND ----------

schema = StructType() \
      .add("ID",IntegerType(),True) \
      .add("OCCUPATION",StringType(),True) \
      .add("SALARY",IntegerType(),True) 

# COMMAND ----------

df2.rdd.collect()

# COMMAND ----------

rdd2=df2.rdd.map(lambda x: (x[0]*2,x[1],x[2]+50))
rdd2.collect()

# COMMAND ----------

df_updated=spark.createDataFrame(rdd2,schema)
df_updated.show()
