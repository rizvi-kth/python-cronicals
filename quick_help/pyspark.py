
pyspark
import os
os.environ["PYSPARK_PYTHON"]
os.environ["PYSPARK_DRIVER_PYTHON"]

# -- All Important Imports --
# >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>>
from pyspark.sql.types import  LongType, StringType, StructField, StructType, BooleanType, ArrayType, IntegerType, DoubleType, DateType, TimestampType
from pyspark.sql.functions import lower, lit, udf, col, broadcast, datediff, to_date, desc, asc, regexp_replace, size, length, monotonically_increasing_id, from_json, split, count
from pyspark.ml.feature import Tokenizer, RegexTokenizer, MinHashLSH, CountVectorizer, NGram

# -- Show Info --
# >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>>
spark.sparkContext.defaultParallelism
df.rdd.getNumPartitions()
df.printSchema()


# -- Repartition --
# >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>>
p_df = df.repartition(250)


# -- Sort --
# >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>>
df.sort(col("SCORE").asc()).show()


# -- Filters ('where' and 'filter' are exactly same in spark) --
# >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>>
df.filter(col("CONTENT_ID")==395271)
df.filter(col("URL").contains("dagensopinion.se/artikel/"))
df.filter("SIZE_LCS_2 > 3")
df.filter(col("SOME_FEATURE").isNull())


# -- DataType convert - IntegerType - DoubleType -- DateTimeType --
# >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>>
new_df = df.withColumn("JCS_1", df["JCS"].cast(DoubleType()))
new_df = df.withColumn("Article_Published_At_dt", to_timestamp("Article_Published_At"))
# -- DataType convert -  JSON to StructType --
s = StructType([StructField("id", IntegerType()), StructField("sex", StringType())])
df_2 = df.withColumn("ARTICLE", from_json(df.ARTICLE_JSON_DIC, s))


# -- Joins --
# >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>>
df = df1.join(df2, on=['id'], how='inner')
df = df1.join(df2, on=['id'], how='outer')
# LEFT JOIN is a type of join between 2 tables. It allows to list all results of the left table (left = left) even if there is no match in the second table. This join is particularly interesting for retrieving information from df1 while retrieving associated data, even if there is no match with df2. Namely, if there is no match the columns of df2 will all be null.
df = df1.join(df2, on=['id'], how='left')
# All rows in the left dataset that match in the right dataset are returned in the final result. However, unlike the left outer join, the result does not contain merged data from the two datasets. It contains only the columns brought by the left dataset.
df  = df1.join(df2, on=['id'], how='left_semi')
# Selects all rows from df1 that are not present in df2
df  = df1.join(df2, on=['id'], how='left_anti')
df = df1.crossJoin(df2)
# This is the same as the left join operation performed on right side dataframe, i.e df2 in this example.
df = df1.join(df2, on=['id'], how='right')


# -- Save  --
# >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>>
sc_sample_df.select(['ARTICLE_ID', 'ARTICLE_TEXT']).coalesce(1).write.format("csv").save("s3a://mnd-spark-test/sg/semantic-score-hits_3/", header=True)


# -- Schema defination for read --
# >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>>
schema = StructType([
    StructField("ARTICLE_ID", IntegerType(), True),
    StructField("ARTICLE_TEXT", StringType(), True),
])
result_df = spark.read.csv(path=files, encoding='UTF-8', header=False, schema=schema)


# -- PySpark SQL (Create Table) --
# >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>>
result_df.createOrReplaceTempView("matches")
# (Run SQL)
spark.sql("select count(*) from matches").show()
# (Select top 70 rows of each group)
result_df=spark.sql("select * from (select *, row_number() OVER (PARTITION BY PR_ID ORDER BY SCALED_SCORE_DB DESC) as rn  FROM matches) tmp where rn < 70")


# -- Group By --
# >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>>
pr_df_1.groupBy(["LCS"]).agg(countDistinct("PR_Id").alias("PR_Count"), countDistinct("Article_Id").alias("Article_Count")).sort(col('PR_Count').desc()).show()


# -- Mapping discreate value in a column with dictionary --
# >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>>
from itertools import chain
from pyspark.sql.functions import create_map, lit
simple_dict = {'pressreleases':1001, 'news':1002, 'blog_posts':1003}
mapping_expr = create_map([lit(x) for x in chain(*simple_dict.items())])
df_2 = df_1.withColumn('content_type_int', mapping_expr[df_1['content_type']])


# -- Cache DF (two types of caching) --
# >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>>
cached_result = result.select(col('Article_Title'), col('Article_Source'), col('Article_Id'), col('Article_Url'), col('LCS')).persist(_path_)
cached_result = result.select(col('Article_Title'), col('Article_Source'), col('Article_Id'), col('Article_Url'), col('LCS')).cache()
# Use cached DF
cached_result.show()
cached_result.filter ...
# Destroy the cash to avoid memory overflow
cachedDF.unpersist()

# -- PySpark UDFs --
# >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>> >>>
# String UDFs
@udf("string")
def udf_fixEncoding(sv_text):
    clean_sv_text = re.compile('\\\\N\{(.*?)\}').sub('', sv_text)
    return clean_sv_text
cleaned_df = df.withColumn('BODY_2', udf_fixEncoding(col('BODY_1')))

# StructType UDFs
# - 1. Define the Structure
resultSchema = StructType([
    StructField("COSINESCORE", DoubleType(), False),
    StructField("DTTEXT", StringType(), False)
])
# - 2. Define UDF function
def udf_match_DT(sample_text):
    max_score = 3.145
    max_text = sample_text
    return max_score, max_text
# - 3. Define the UDF
dt_matcher_udf = udf(udf_match_DT, resultSchema)
# - 4. Use the UDF
lcs_df_2 = lcs_df.withColumn('DT', dt_matcher_udf(lcs_df['LCS']))
lcs_df_2.select(col('DT.*')).show()
