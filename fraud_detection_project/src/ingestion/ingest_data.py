"""
Data Ingestion Module

Handles:
- Schema enforcement
- Data quality checks
- Partitioning
- Deduplication
- Delta Lake writes
"""

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import *
from pyspark.sql.window import Window
from delta import configure_spark_with_delta_pip
from delta.tables import DeltaTable
from loguru import logger
from pathlib import Path
import yaml


def get_transaction_schema() -> StructType:
    """Define strict schema for transaction data."""
    return StructType([
        StructField("transaction_id", StringType(), False),
        StructField("account_id", StringType(), False),
        StructField("customer_id", StringType(), True),
        StructField("amount", DoubleType(), False),
        StructField("currency", StringType(), True),
        StructField("merchant_id", StringType(), True),
        StructField("merchant_category", StringType(), True),
        StructField("device_id", StringType(), True),
        StructField("ip_address", StringType(), True),
        StructField("geo_location", StringType(), True),
        StructField("event_time", TimestampType(), False),
        StructField("event_date", DateType(), False),
        StructField("transaction_type", StringType(), True),
        StructField("channel", StringType(), True),
        StructField("is_fraud", IntegerType(), True)
    ])


def get_kyc_schema() -> StructType:
    """Define schema for KYC data."""
    return StructType([
        StructField("customer_id", StringType(), False),
        StructField("account_id", StringType(), False),
        StructField("first_name", StringType(), True),
        StructField("last_name", StringType(), True),
        StructField("email", StringType(), True),
        StructField("phone_hash", StringType(), True),
        StructField("address", StringType(), True),
        StructField("city", StringType(), True),
        StructField("state", StringType(), True),
        StructField("zip_code", StringType(), True),
        StructField("country", StringType(), True),
        StructField("ssn_hash", StringType(), True),
        StructField("date_of_birth", DateType(), True),
        StructField("kyc_date", DateType(), True),
        StructField("risk_score", DoubleType(), True),
        StructField("employment_status", StringType(), True),
        StructField("annual_income", DoubleType(), True)
    ])


def normalize_phone_number(df, column_name="phone"):
    """Normalize phone numbers to standard format."""
    from pyspark.sql.functions import regexp_replace, col
    
    # Remove all non-digit characters
    df = df.withColumn(
        column_name,
        regexp_replace(col(column_name), r"[^\d]", "")
    )
    
    # Ensure 10 digits (US format)
    df = df.withColumn(
        column_name,
        F.when(F.length(col(column_name)) == 10, col(column_name))
         .when(F.length(col(column_name)) == 11, F.substring(col(column_name), 2, 10))
         .otherwise(None)
    )
    
    return df


def enforce_schema(spark, raw_path: str, schema: StructType, merge_schema: bool = True) -> DataFrame:
    """Read raw data with schema enforcement."""
    logger.info(f"Reading data from {raw_path} with schema enforcement")
    
    try:
        df = (spark.read
              .schema(schema)
              .option("mergeSchema", merge_schema)
              .option("multiline", "true")
              .json(raw_path))
        logger.info(f"Successfully loaded {df.count()} records")
        return df
    except Exception as e:
        logger.error(f"Schema enforcement failed: {str(e)}")
        raise


def check_data_quality(df, table_name: str) -> dict:
    """Perform comprehensive data quality checks."""
    logger.info(f"Running data quality checks on {table_name}")
    
    quality_report = {
        "total_records": df.count(),
        "null_counts": {},
        "duplicate_counts": {},
        "invalid_values": {}
    }
    
    # Null analysis
    for col_name in df.columns:
        null_count = df.filter(F.col(col_name).isNull()).count()
        null_pct = (null_count / quality_report["total_records"]) * 100
        quality_report["null_counts"][col_name] = {
            "count": null_count,
            "percentage": round(null_pct, 2)
        }
        
        if null_pct > 50:
            logger.warning(f"Column {col_name} has {null_pct:.2f}% null values")
    
    # Duplicate analysis
    key_columns = [c for c in df.columns if "id" in c.lower() or "key" in c.lower()]
    if key_columns:
        total = df.count()
        distinct = df.select(key_columns).distinct().count()
        duplicates = total - distinct
        quality_report["duplicate_counts"] = {
            "total": total,
            "distinct": distinct,
            "duplicates": duplicates,
            "duplicate_rate": round((duplicates / total) * 100, 2) if total > 0 else 0
        }
    
    # Amount validation (negative amounts for certain transaction types)
    if "amount" in df.columns:
        negative_amounts = df.filter(F.col("amount") < 0).count()
        quality_report["invalid_values"]["negative_amounts"] = negative_amounts
    
    logger.info(f"Data quality check complete for {table_name}")
    return quality_report


def partition_and_write(df, output_path: str, partition_by: list, format: str = "delta"):
    """Write data with partitioning for efficient querying."""
    logger.info(f"Writing data to {output_path}, partitioned by {partition_by}")
    
    # Repartition before writing
    df = df.repartition(len(partition_by), *partition_by)
    
    if format == "delta":
        (df.write
         .format("delta")
         .mode("overwrite")
         .partitionBy(*partition_by)
         .option("mergeSchema", "true")
         .save(output_path))
    elif format == "parquet":
        (df.write
         .mode("overwrite")
         .partitionBy(*partition_by)
         .parquet(output_path))
    
    logger.info(f"Successfully wrote data to {output_path}")


def deduplicate(df, key_columns: list, order_by: list[str] | None = None) -> DataFrame:
    """Remove duplicate records based on key columns.

    Order duplicates by the provided columns when possible, otherwise keep the
    first record within each partition.
    """
    logger.info(f"Deduplicating on columns: {key_columns}")

    if order_by is None:
        order_by = ["event_time"] if "event_time" in df.columns else []

    order_exprs = []
    for col_name in order_by:
        if col_name in df.columns:
            order_exprs.append(F.desc(col_name))
        else:
            logger.warning(f"Order-by column '{col_name}' not found in DataFrame; skipping")

    if not order_exprs:
        order_exprs.append(F.desc(F.lit(1)))

    window_spec = Window.partitionBy(*key_columns).orderBy(*order_exprs)

    df_dedup = (df.withColumn("row_num", F.row_number().over(window_spec))
                .filter(F.col("row_num") == 1)
                .drop("row_num"))

    original_count = df.count()
    dedup_count = df_dedup.count()
    removed = original_count - dedup_count

    logger.info(f"Removed {removed} duplicates ({(removed/original_count)*100:.2f}%)")

    return df_dedup


def run_ingestion(config: dict) -> dict:
    """Main ingestion pipeline."""
    from pyspark.sql import SparkSession
    
    logger.info("Starting data ingestion pipeline")
    
    # Initialize Spark session
    spark_config = config.get('spark', {})
    data_config = config.get('data', {})
    ingestion_config = config.get('ingestion', {})
    
    builder = (SparkSession.builder
               .appName(spark_config.get('app_name', 'FraudDetection'))
               .master(spark_config.get('master', 'local[*]'))
               .config("spark.executor.memory", spark_config.get('executor_memory', '4g'))
               .config("spark.driver.memory", spark_config.get('driver_memory', '2g'))
               .config("spark.sql.shuffle.partitions", spark_config.get('shuffle_partitions', 200))
               .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
               .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"))
    
    builder = configure_spark_with_delta_pip(builder)
    spark = builder.getOrCreate()
    
    results = {
        "transactions": {},
        "kyc": {},
        "quality_reports": {}
    }
    
    try:
        # Create directories
        Path(data_config['bronze_path']).mkdir(parents=True, exist_ok=True)
        Path(data_config['silver_path']).mkdir(parents=True, exist_ok=True)
        
        # ========== TRANSACTIONS ==========
        logger.info("\n" + "="*60)
        logger.info("PROCESSING TRANSACTIONS")
        logger.info("="*60)
        
        # Load raw transactions
        transaction_schema = get_transaction_schema()
        transactions_raw = enforce_schema(
            spark,
            data_config['raw_path'] + 'transactions/',
            transaction_schema,
            ingestion_config.get('merge_schema', True)
        )
        
        # Normalize phone numbers if present
        if 'phone' in transactions_raw.columns:
            transactions_raw = normalize_phone_number(transactions_raw, 'phone')
        
        # Data quality check
        tx_quality = check_data_quality(transactions_raw, "transactions")
        results['quality_reports']['transactions'] = tx_quality
        
        # Deduplicate
        transactions_clean = deduplicate(
            transactions_raw,
            ['transaction_id']
        )
        
        # Add event_date partition column if not present
        if 'event_date' not in transactions_clean.columns:
            transactions_clean = transactions_clean.withColumn(
                'event_date',
                F.to_date(F.col('event_time'))
            )
        
        # Write to bronze layer
        partition_and_write(
            transactions_clean,
            data_config['bronze_path'] + 'transactions',
            ['event_date'],
            'delta'
        )
        
        results['transactions'] = {
            "input_records": tx_quality['total_records'],
            "output_records": transactions_clean.count(),
            "duplicates_removed": tx_quality['total_records'] - transactions_clean.count()
        }
        
        # ========== KYC DATA ==========
        logger.info("\n" + "="*60)
        logger.info("PROCESSING KYC DATA")
        logger.info("="*60)
        
        kyc_schema = get_kyc_schema()
        kyc_raw = enforce_schema(
            spark,
            data_config['raw_path'] + 'kyc/',
            kyc_schema,
            ingestion_config.get('merge_schema', True)
        )
        
        # Normalize phone numbers
        if 'phone_hash' in kyc_raw.columns:
            kyc_raw = normalize_phone_number(kyc_raw, 'phone_hash')
        
        # Data quality check
        kyc_quality = check_data_quality(kyc_raw, "kyc")
        results['quality_reports']['kyc'] = kyc_quality
        
        # Deduplicate
        kyc_clean = deduplicate(kyc_raw, ['customer_id', 'account_id'], order_by=['kyc_date'])
        
        # Write to silver layer
        partition_and_write(
            kyc_clean,
            data_config['silver_path'] + 'kyc',
            ['country'],
            'delta'
        )
        
        results['kyc'] = {
            "input_records": kyc_quality['total_records'],
            "output_records": kyc_clean.count()
        }
        
        # ========== SUMMARY ==========
        logger.info("\n" + "="*60)
        logger.info("INGESTION SUMMARY")
        logger.info("="*60)
        logger.info(f"Transactions: {results['transactions']}")
        logger.info(f"KYC: {results['kyc']}")
        
        # Save quality report
        import json
        report_path = Path(data_config['output_path']) / 'quality_report.json'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSON-serializable format
        serializable_report = {}
        for key, value in results['quality_reports'].items():
            serializable_report[key] = {
                k: v for k, v in value.items()
            }
        
        with open(report_path, 'w') as f:
            json.dump(serializable_report, f, indent=2, default=str)
        
        logger.info(f"Quality report saved to {report_path}")
        
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}", exc_info=True)
        raise
    finally:
        spark.stop()
    
    return results


if __name__ == "__main__":
    # Example usage
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    results = run_ingestion(config)
    print(f"Ingestion complete: {results}")
