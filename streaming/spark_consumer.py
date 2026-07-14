import os
import logging

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.avro.functions import from_avro
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BROKER         = os.getenv("REDPANDA_BROKERS", "localhost:29092")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "./data/checkpoints")
OUTPUT_DIR     = "./data/windowed"

_SCHEMAS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "schemas")
NEWS_SCHEMA_STR  = open(os.path.join(_SCHEMAS_DIR, "news_event_v1.avsc")).read()
PRICE_SCHEMA_STR = open(os.path.join(_SCHEMAS_DIR, "price_tick_v1.avsc")).read()


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("FinPlatformStreaming")
        .master(os.getenv("SPARK_MASTER", "local[2]"))
        .config("spark.sql.shuffle.partitions", "4")
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,"
            "org.apache.spark:spark-avro_2.12:3.5.1",
        )
        .getOrCreate()
    )


def read_topic(spark: SparkSession, topic: str):
    return (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", BROKER)
        .option("subscribe", topic)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )


def parse_news(df):
    """Strip the 5-byte Confluent magic prefix then deserialize Avro."""
    stripped = df.select(
        F.expr("substring(value, 6)").alias("avro_payload")
    )
    return (
        stripped
        .select(from_avro("avro_payload", NEWS_SCHEMA_STR).alias("d"))
        .select("d.*")
        .withColumn("event_time", (F.col("ts") / 1000).cast("timestamp"))
        .withWatermark("event_time", "10 minutes")
    )


def parse_prices(df):
    """Strip the 5-byte Confluent magic prefix then deserialize Avro."""
    stripped = df.select(
        F.expr("substring(value, 6)").alias("avro_payload")
    )
    return (
        stripped
        .select(from_avro("avro_payload", PRICE_SCHEMA_STR).alias("d"))
        .select("d.*")
        .withColumn("event_time", (F.col("ts") / 1000).cast("timestamp"))
        .withWatermark("event_time", "10 minutes")
    )


def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    news_raw   = read_topic(spark, "news-raw")
    price_raw  = read_topic(spark, "price-ticks")

    news   = parse_news(news_raw)
    prices = parse_prices(price_raw)

    news_windowed = (
        news.groupBy(
            F.window("event_time", "5 minutes"),
            F.col("ticker"),
        )
        .agg(
            F.count("title").alias("headline_count"),
            F.collect_list("title").alias("titles"),
        )
        .select(
            F.col("window.start").alias("window_start"),
            F.col("window.end").alias("window_end"),
            F.col("ticker"),
            F.col("headline_count"),
            F.col("titles"),
        )
    )

    price_windowed = (
        prices.groupBy(
            F.window("event_time", "5 minutes"),
            F.col("ticker"),
        )
        .agg(
            F.avg("close").alias("avg_close"),
            F.avg("volume").alias("avg_volume"),
            F.last("close").alias("last_close"),
            F.first("close").alias("first_close"),
        )
        .withColumn(
            "pct_change",
            (F.col("last_close") - F.col("first_close")) / F.col("first_close") * 100,
        )
        .select(
            F.col("window.start").alias("window_start"),
            F.col("window.end").alias("window_end"),
            F.col("ticker"),
            F.col("avg_close"),
            F.col("avg_volume"),
            F.col("pct_change"),
        )
    )

    news_query = (
        news_windowed.writeStream
        .outputMode("append")
        .format("parquet")
        .option("path", f"{OUTPUT_DIR}/news")
        .option("checkpointLocation", f"{CHECKPOINT_DIR}/news")
        .trigger(processingTime="30 seconds")
        .start()
    )

    price_query = (
        price_windowed.writeStream
        .outputMode("append")
        .format("parquet")
        .option("path", f"{OUTPUT_DIR}/prices")
        .option("checkpointLocation", f"{CHECKPOINT_DIR}/prices")
        .trigger(processingTime="30 seconds")
        .start()
    )

    log.info("Streaming queries started. Waiting for termination.")
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
