from pyspark.sql import SparkSession
from pyspark.sql.functions import count, sum, avg, min, max

spark = SparkSession.builder \
    .appName("Employee Demo") \
    .master("local[2]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# 1. Create a DataFrame from JSON
df = spark.read.json("employees.json")
df.show()

# 2. Select columns
df.select("name", "department", "salary").show()

# 3. Filter records
df.filter("department = 'IT' AND salary >= 65000").show()

# 4. Sort by salary, highest first
df.orderBy("salary", ascending=False).show()

# 5. Group by department and aggregate
df.groupBy("department").agg(
    count("*").alias("employees"),
    sum("salary").alias("total_salary"),
    avg("salary").alias("average_salary"),
    min("salary").alias("minimum_salary"),
    max("salary").alias("maximum_salary")
).show()

# 6. Overall totals
df.agg(
    count("*").alias("employees"),
    sum("salary").alias("total_salary"),
    avg("salary").alias("average_salary")
).show()

spark.stop()