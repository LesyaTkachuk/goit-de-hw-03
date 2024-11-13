from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, round


# create spark session
spark = SparkSession.builder.appName("HW-03-data-analysis").getOrCreate()

# ----------------------------------------------------------------------
# Task 1 - load datasets
products_df = spark.read.csv("./data/products.csv", header=True)
purchases_df = spark.read.csv("./data/purchases.csv", header=True)
users_df = spark.read.csv("./data/users.csv", header=True)

products_df.show(5)
purchases_df.show(5)
users_df.show(5)

print(
    f"Number of rows: purchases - {purchases_df.count()}, products - {products_df.count()}, users - {users_df.count()} "
)

# ----------------------------------------------------------------------
# Task 2 - drop all rows that contains null values
products_df = products_df.dropna()
purchases_df = purchases_df.dropna()
users_df = users_df.dropna()

print(
    f"Number of rows after dropna(): purchases - {purchases_df.count()}, products - {products_df.count()}, users - {users_df.count()} "
)

# ----------------------------------------------------------------------
# Task 3 - calculate the total purchase amount for each product category
purchases_df.join(
    products_df, purchases_df.product_id == products_df.product_id, "inner"
).drop(products_df.product_id).select(
    "product_id", "quantity", "category", "price"
).withColumn(
    "total", round(col("quantity") * col("price"), 2)
).groupby(
    "category"
).agg(
    sum("total").alias("total_sum")
).withColumn(
    "total_sum", round("total_sum", 2)
).show()

# ----------------------------------------------------------------------
# Task 4 - calculate the total purchase amount for each product category for users aged 18 to 25
cat_total_sum = (
    purchases_df.join(
        products_df, purchases_df.product_id == products_df.product_id, "inner"
    )
    .join(users_df, purchases_df.user_id == users_df.user_id, "inner")
    .drop(products_df.product_id, users_df.user_id)
    .select("product_id", "category", "price", "quantity", "age")
    .filter(col("age").between(18, 25))
    .withColumn("total", round(col("quantity") * col("price"), 2))
    .groupby("category")
    .agg(sum("total").alias("total_sum"))
    .withColumn("total_sum", round("total_sum", 2))
)

cat_total_sum.show()

# ----------------------------------------------------------------------
# Task 5 - calculate the proportion of each category's total purchase amount relative to the overall purchase amount for all categories for users aged 18 to 25
general_sum = cat_total_sum.agg(sum("total_sum")).collect()[0][0]

cat_total_sum_perc = cat_total_sum.withColumn(
    "percentage", round((col("total_sum") / general_sum) * 100, 2)
)

cat_total_sum_perc.show()

# ----------------------------------------------------------------------
# Task 6 - get top-3 categories with the highest purchase percentages for users aged 18 to 25
cat_total_sum_perc.orderBy(col("percentage").desc()).limit(3).show()

spark.stop()
