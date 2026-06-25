"""
Exploratory Data Analysis Module

Handles:
- Missingness analysis
- Label distribution
- Time-series fraud patterns
- Amount distribution analysis
- Device reuse heatmaps
- Graph structure analysis
"""

import pyspark.sql.functions as F
from pyspark.sql.window import Window
from loguru import logger
from pathlib import Path
import json


def analyze_missingness(df, table_name: str) -> dict:
    """Analyze missing values in DataFrame."""
    logger.info(f"Analyzing missingness for {table_name}")
    
    total = df.count()
    missingness_report = {}
    
    for col_name in df.columns:
        null_count = df.filter(F.col(col_name).isNull()).count()
        null_pct = (null_count / total) * 100 if total > 0 else 0
        
        missingness_report[col_name] = {
            "null_count": null_count,
            "null_percentage": round(null_pct, 2),
            "severity": "high" if null_pct > 50 else "medium" if null_pct > 20 else "low"
        }
    
    return missingness_report


def analyze_label_distribution(df) -> dict:
    """Analyze fraud label distribution."""
    logger.info("Analyzing label distribution")
    
    if 'is_fraud' not in df.columns:
        return {"error": "is_fraud column not found"}
    
    total = df.count()
    fraud_count = df.filter(F.col('is_fraud') == 1).count()
    legit_count = total - fraud_count
    
    fraud_rate = (fraud_count / total) * 100 if total > 0 else 0
    
    report = {
        "total_records": total,
        "fraud_count": fraud_count,
        "legitimate_count": legit_count,
        "fraud_rate_percentage": round(fraud_rate, 4),
        "class_imbalance_ratio": round(legit_count / fraud_count, 2) if fraud_count > 0 else float('inf'),
        "severity": "extreme" if fraud_rate < 0.1 else "high" if fraud_rate < 1 else "moderate"
    }
    
    logger.info(f"Fraud rate: {fraud_rate:.4f}% ({fraud_count}/{total})")
    
    return report


def analyze_time_series_patterns(df) -> dict:
    """Analyze fraud patterns over time."""
    logger.info("Analyzing time-series fraud patterns")
    
    if 'event_date' not in df.columns or 'is_fraud' not in df.columns:
        return {"error": "Required columns not found"}
    
    # Daily fraud rate
    daily_stats = (df.groupBy('event_date')
                   .agg(
                       F.count('*').alias('total_volume'),
                       F.sum('is_fraud').alias('fraud_count'),
                       F.avg('is_fraud').alias('fraud_rate'),
                       F.sum('amount').alias('total_amount'),
                       F.sum(F.when(F.col('is_fraud') == 1, F.col('amount')).otherwise(0)).alias('fraud_amount')
                   )
                   .orderBy('event_date')
                   .toPandas())
    
    # Weekly patterns
    df_with_dow = df.withColumn('day_of_week', F.dayofweek('event_date'))
    weekly_stats = (df_with_dow.groupBy('day_of_week')
                    .agg(
                        F.avg('is_fraud').alias('avg_fraud_rate'),
                        F.count('*').alias('volume')
                    )
                    .orderBy('day_of_week')
                    .toPandas())
    
    # Hourly patterns (if event_time available)
    hourly_stats = None
    if 'event_time' in df.columns:
        df_with_hour = df.withColumn('hour', F.hour('event_time'))
        hourly_stats = (df_with_hour.groupBy('hour')
                        .agg(
                            F.avg('is_fraud').alias('avg_fraud_rate'),
                            F.count('*').alias('volume')
                        )
                        .orderBy('hour')
                        .toPandas())
    
    report = {
        "daily_statistics": daily_stats.to_dict('records') if daily_stats is not None else [],
        "weekly_statistics": weekly_stats.to_dict('records') if weekly_stats is not None else [],
        "hourly_statistics": hourly_stats.to_dict('records') if hourly_stats is not None else []
    }
    
    return report


def analyze_amount_distribution(df) -> dict:
    """Analyze transaction amount distributions."""
    logger.info("Analyzing amount distribution")
    
    if 'amount' not in df.columns:
        return {"error": "amount column not found"}
    
    # Overall statistics
    overall_stats = df.select(
        F.min('amount').alias('min_amount'),
        F.max('amount').alias('max_amount'),
        F.avg('amount').alias('avg_amount'),
        F.expr('percentile_approx(amount, 0.5)').alias('median_amount'),
        F.expr('percentile_approx(amount, 0.95)').alias('p95_amount'),
        F.expr('percentile_approx(amount, 0.99)').alias('p99_amount'),
        F.stddev('amount').alias('std_amount')
    ).first().asDict()
    
    # Fraud vs legitimate comparison
    if 'is_fraud' in df.columns:
        comparison_stats = (df.groupBy('is_fraud')
                            .agg(
                                F.count('*').alias('count'),
                                F.avg('amount').alias('avg_amount'),
                                F.expr('percentile_approx(amount, 0.5)').alias('median_amount'),
                                F.sum('amount').alias('total_amount')
                            )
                            .withColumn('label', 
                                       F.when(F.col('is_fraud') == 1, 'fraud').otherwise('legitimate'))
                            .toPandas()
                            .to_dict('records'))
    else:
        comparison_stats = []
    
    # Amount buckets
    bucketed = (df.withColumn('amount_bucket',
                              F.when(F.col('amount') < 100, '<100')
                               .when(F.col('amount') < 500, '100-500')
                               .when(F.col('amount') < 1000, '500-1000')
                               .when(F.col('amount') < 5000, '1000-5000')
                               .otherwise('5000+'))
                .groupBy('amount_bucket')
                .agg(
                    F.count('*').alias('total_count'),
                    F.avg('is_fraud').alias('fraud_rate')
                )
                .orderBy('amount_bucket')
                .toPandas()
                .to_dict('records'))
    
    report = {
        "overall_statistics": {k: float(v) if v else None for k, v in overall_stats.items()},
        "fraud_vs_legitimate": comparison_stats,
        "amount_buckets": bucketed
    }
    
    return report


def analyze_device_reuse(df) -> dict:
    """Analyze device reuse patterns (indicator of fraud rings)."""
    logger.info("Analyzing device reuse patterns")
    
    if 'device_id' not in df.columns:
        return {"error": "device_id column not found"}
    
    # Count accounts per device
    device_account_count = (df.groupBy('device_id')
                            .agg(
                                F.countDistinct('account_id').alias('account_count'),
                                F.count('*').alias('transaction_count'),
                                F.avg('is_fraud').alias('fraud_rate')
                            )
                            .filter(F.col('account_count') > 1)
                            .orderBy(F.desc('account_count')))
    
    # Statistics
    total_devices = df.select('device_id').distinct().count()
    multi_account_devices = device_account_count.count()
    
    # High-risk devices (used by 10+ accounts)
    high_risk_devices = device_account_count.filter(F.col('account_count') >= 10).count()
    
    # Top 10 most reused devices
    top_devices = device_account_count.limit(10).toPandas().to_dict('records')
    
    report = {
        "total_unique_devices": total_devices,
        "devices_with_multiple_accounts": multi_account_devices,
        "high_risk_devices_10plus": high_risk_devices,
        "percentage_multi_account": round((multi_account_devices / total_devices) * 100, 2) if total_devices > 0 else 0,
        "top_reused_devices": top_devices
    }
    
    logger.info(f"Found {high_risk_devices} high-risk devices used by 10+ accounts")
    
    return report


def analyze_graph_structure(df) -> dict:
    """Analyze graph structure for fraud ring detection."""
    logger.info("Analyzing graph structure")
    
    try:
        from graphframes import GraphFrame
        
        # Build bipartite graph: accounts <-> devices
        if 'device_id' not in df.columns or 'account_id' not in df.columns:
            return {"error": "Required columns for graph not found"}
        
        # Create edges
        edges = (df.select('account_id', 'device_id')
                 .filter(F.col('device_id').isNotNull())
                 .distinct()
                 .select(
                     F.col('account_id').alias('src'),
                     F.col('device_id').alias('dst')
                 ))
        
        # Create vertices
        account_vertices = df.select('account_id').distinct().withColumnRenamed('account_id', 'id')
        device_vertices = df.select('device_id').distinct().withColumnRenamed('device_id', 'id')
        vertices = account_vertices.union(device_vertices).distinct()
        
        # Create graph
        g = GraphFrame(vertices, edges)
        
        # Connected components (fraud rings)
        sc_checkpoint_dir = Path('/tmp/graphframes_checkpoints')
        sc_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        from pyspark import SparkContext
        sc = SparkContext.getOrCreate()
        sc.setCheckpointDir(str(sc_checkpoint_dir))
        
        components = g.connectedComponents()
        
        # Component statistics
        component_stats = (components.groupBy('component')
                          .agg(F.count('*').alias('size'))
                          .orderBy(F.desc('size')))
        
        total_components = components.select('component').distinct().count()
        largest_component = component_stats.first()['size'] if component_stats.count() > 0 else 0
        avg_component_size = component_stats.agg(F.avg('size')).first()[0]
        
        # Large components (>10 nodes) - potential fraud rings
        large_components = component_stats.filter(F.col('size') > 10).count()
        
        report = {
            "total_components": total_components,
            "largest_component_size": largest_component,
            "average_component_size": round(avg_component_size, 2) if avg_component_size else 0,
            "large_components_10plus": large_components,
            "potential_fraud_rings": large_components
        }
        
        logger.info(f"Found {large_components} potential fraud rings (components with >10 nodes)")
        
        return report
        
    except ImportError:
        logger.warning("GraphFrames not installed, skipping graph analysis")
        return {"error": "GraphFrames not available"}


def run_eda(config: dict) -> dict:
    """Main EDA pipeline."""
    from pyspark.sql import SparkSession
    
    logger.info("Starting Exploratory Data Analysis")
    
    spark_config = config.get('spark', {})
    data_config = config.get('data', {})
    
    spark = (SparkSession.builder
             .appName(spark_config.get('app_name', 'FraudDetection_EDA'))
             .master(spark_config.get('master', 'local[*]'))
             .config("spark.executor.memory", spark_config.get('executor_memory', '4g'))
             .config("spark.driver.memory", spark_config.get('driver_memory', '2g'))
             .getOrCreate())
    
    results = {}
    
    try:
        # Load bronze layer data
        transactions_path = data_config['bronze_path'] + 'transactions'
        logger.info(f"Loading transactions from {transactions_path}")
        
        df = spark.read.format("delta").load(transactions_path)
        
        # Sample for faster analysis if dataset is large
        total_count = df.count()
        if total_count > 100000:
            logger.info(f"Dataset large ({total_count} records), sampling 100k records")
            df_sample = df.sample(fraction=100000/total_count, seed=42)
        else:
            df_sample = df
        
        logger.info(f"Analyzing {df_sample.count()} records")
        
        # Run all analyses
        logger.info("\n" + "="*60)
        logger.info("MISSINGNESS ANALYSIS")
        logger.info("="*60)
        results['missingness'] = analyze_missingness(df_sample, "transactions")
        
        logger.info("\n" + "="*60)
        logger.info("LABEL DISTRIBUTION")
        logger.info("="*60)
        results['label_distribution'] = analyze_label_distribution(df_sample)
        
        logger.info("\n" + "="*60)
        logger.info("TIME-SERIES PATTERNS")
        logger.info("="*60)
        results['time_series'] = analyze_time_series_patterns(df_sample)
        
        logger.info("\n" + "="*60)
        logger.info("AMOUNT DISTRIBUTION")
        logger.info("="*60)
        results['amount_distribution'] = analyze_amount_distribution(df_sample)
        
        logger.info("\n" + "="*60)
        logger.info("DEVICE REUSE ANALYSIS")
        logger.info("="*60)
        results['device_reuse'] = analyze_device_reuse(df_sample)
        
        logger.info("\n" + "="*60)
        logger.info("GRAPH STRUCTURE ANALYSIS")
        logger.info("="*60)
        results['graph_structure'] = analyze_graph_structure(df_sample)
        
        # Save EDA report
        output_path = Path(data_config['output_path'])
        output_path.mkdir(parents=True, exist_ok=True)
        
        eda_report_path = output_path / 'eda_report.json'
        
        # Convert to JSON-serializable format
        def serialize(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif hasattr(obj, 'isoformat'):
                return obj.isoformat()
            return str(obj)
        
        with open(eda_report_path, 'w') as f:
            json.dump(results, f, indent=2, default=serialize)
        
        logger.info(f"\nEDA report saved to {eda_report_path}")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("EDA SUMMARY")
        logger.info("="*60)
        if 'label_distribution' in results:
            ld = results['label_distribution']
            logger.info(f"Fraud Rate: {ld.get('fraud_rate_percentage', 'N/A')}%")
            logger.info(f"Class Imbalance: {ld.get('class_imbalance_ratio', 'N/A')}:1")
        
        if 'device_reuse' in results:
            dr = results['device_reuse']
            logger.info(f"High-risk devices: {dr.get('high_risk_devices_10plus', 'N/A')}")
        
        if 'graph_structure' in results:
            gs = results['graph_structure']
            logger.info(f"Potential fraud rings: {gs.get('potential_fraud_rings', 'N/A')}")
        
    except Exception as e:
        logger.error(f"EDA failed: {str(e)}", exc_info=True)
        raise
    finally:
        spark.stop()
    
    return results


if __name__ == "__main__":
    import yaml
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    results = run_eda(config)
    print(f"EDA complete: {list(results.keys())}")
