import findspark
from pyspark import SparkContext
from pyspark.sql import SparkSession, Window, Row
from pyspark.sql.functions import *
from pyspark.sql.types import *
import matplotlib.pyplot as plt


# https://spark.apache.org/docs/latest/sql-getting-started.html
spark = SparkSession.builder.appName("firstSpark").getOrCreate()


def load_dataframe(filename):
    df = spark.read.format("csv").options(header="true").load(filename)
    return df


# creating a dataframe
df_matches = load_dataframe("./Data/Matches.csv")
df_matches.limit(5).show()


# lets rename some of the columns
old_cols = df_matches.columns[-3:]
new_cols = ["HomeTeamGoals", "AwayTeamGoals", "FinalResult"]
old_new_cols = [*zip(old_cols, new_cols)]
for old_col, new_col in old_new_cols:
    df_matches = df_matches.withColumnRenamed(old_col, new_col)

df_matches.limit(5).toPandas()


df_matches = (
    df_matches.withColumn(
        "HomeTeamWin", when(col("FinalResult") == "H", 1).otherwise(0)
    )
    .withColumn("AwayTeamWin", when(col("FinalResult") == "A", 1).otherwise(0))
    .withColumn("GameTie", when(col("FinalResult") == "D", 1).otherwise(0))
)


# bundesliga is a D1 division and we are interested in season <= 2010 and >= 2000
bundesliga = df_matches.filter(
    (col("Season") >= 2000) & (col("Season") <= 2010) & (col("Div") == "D1")
)

# home team features
home = (
    bundesliga.groupby("Season", "HomeTeam")
    .agg(
        sum("HomeTeamWin").alias("TotalHomeWin"),
        sum("AwayTeamWin").alias("TotalHomeLoss"),
        sum("GameTie").alias("TotalHomeTie"),
        sum("HomeTeamGoals").alias("HomeScoredGoals"),
        sum("AwayTeamGoals").alias("HomeAgainstGoals"),
    )
    .withColumnRenamed("HomeTeam", "Team")
)

# away game features
away = (
    bundesliga.groupby("Season", "AwayTeam")
    .agg(
        sum("AwayTeamWin").alias("TotalAwayWin"),
        sum("HomeTeamWin").alias("TotalAwayLoss"),
        sum("GameTie").alias("TotalAwayTie"),
        sum("AwayTeamGoals").alias("AwayScoredGoals"),
        sum("HomeTeamGoals").alias("AwayAgainstGoals"),
    )
    .withColumnRenamed("AwayTeam", "Team")
)

# season features
window = ["Season"]
window = Window.partitionBy(window).orderBy(
    col("WinPct").desc(), col("GoalDifferentials").desc()
)
table = (
    home.join(away, ["Team", "Season"], "inner")
    .withColumn("GoalsScored", col("HomeScoredGoals") + col("AwayScoredGoals"))
    .withColumn("GoalsAgainst", col("HomeAgainstGoals") + col("AwayAgainstGoals"))
    .withColumn("GoalDifferentials", col("GoalsScored") - col("GoalsAgainst"))
    .withColumn("Win", col("TotalHomeWin") + col("TotalAwayWin"))
    .withColumn("Loss", col("TotalHomeLoss") + col("TotalAwayLoss"))
    .withColumn("Tie", col("TotalHomeTie") + col("TotalAwayTie"))
    .withColumn(
        "WinPct", round((100 * col("Win") / (col("Win") + col("Loss") + col("Tie"))), 2)
    )
    .drop("HomeScoredGoals", "AwayScoredGoals", "HomeAgainstGoals", "AwayAgainstGoals")
    .drop(
        "TotalHomeWin",
        "TotalAwayWin",
        "TotalHomeLoss",
        "TotalAwayLoss",
        "TotalHomeTie",
        "TotalAwayTie",
    )
    .withColumn("TeamPosition", rank().over(window))
)

table_df = table.filter(col("TeamPosition") == 1).orderBy(asc("Season")).toPandas()
table_df
