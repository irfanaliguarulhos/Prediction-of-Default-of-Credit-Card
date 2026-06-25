"""
Banking Fraud Detection - Main Pipeline Entry Point

This script orchestrates the end-to-end fraud detection pipeline:
1. Data Ingestion & Quality
2. Exploratory Data Analysis
3. Feature Engineering & Embeddings
4. Model Training & Tuning
5. Model Evaluation
6. Model Export & Serving
7. Monitoring Setup

Usage:
    python src/main.py --config config.yaml
    python src/main.py --config config.yaml --stage ingestion
    python src/main.py --config config.yaml --stage all
"""

import argparse
import yaml
import sys
from pathlib import Path
from loguru import logger
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from ingestion.ingest_data import run_ingestion
from eda.explore_data import run_eda
from features.engineer_features import run_feature_engineering
from models.train_model import run_model_training
from evaluation.evaluate_model import run_evaluation
from serving.export_model import export_model_for_serving
from monitoring.setup_monitoring import setup_monitoring_dashboard


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: dict):
    """Configure logging based on config."""
    log_config = config.get('logging', {})
    log_file = log_config.get('file', 'output/logs/fraud_detection.log')
    
    # Create log directory
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure loguru
    logger.remove()
    logger.add(
        sys.stdout,
        format=log_config.get('format', "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"),
        level=log_config.get('level', 'INFO')
    )
    logger.add(
        log_file,
        format=log_config.get('format', "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"),
        level=log_config.get('level', 'INFO'),
        rotation=log_config.get('rotation', '100 MB'),
        retention=log_config.get('retention', '30 days')
    )
    
    return logger


def run_pipeline(config: dict, stage: str = 'all'):
    """Run the fraud detection pipeline."""
    logger.info("=" * 80)
    logger.info("BANKING FRAUD DETECTION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().isoformat()}")
    logger.info(f"Running stage: {stage}")
    
    results = {}
    
    try:
        # Stage 1: Data Ingestion
        if stage in ['all', 'ingestion']:
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 1: DATA INGESTION & QUALITY")
            logger.info("=" * 80)
            results['ingestion'] = run_ingestion(config)
            logger.info(f"Ingestion complete: {results['ingestion']}")
        
        # Stage 2: Exploratory Data Analysis
        if stage in ['all', 'eda']:
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 2: EXPLORATORY DATA ANALYSIS")
            logger.info("=" * 80)
            results['eda'] = run_eda(config)
            logger.info(f"EDA complete: {results['eda']}")
        
        # Stage 3: Feature Engineering
        if stage in ['all', 'features']:
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 3: FEATURE ENGINEERING & EMBEDDINGS")
            logger.info("=" * 80)
            results['features'] = run_feature_engineering(config)
            logger.info(f"Feature engineering complete: {results['features']}")
        
        # Stage 4: Model Training
        if stage in ['all', 'training']:
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 4: MODEL TRAINING & TUNING")
            logger.info("=" * 80)
            results['training'] = run_model_training(config)
            logger.info(f"Model training complete: {results['training']}")
        
        # Stage 5: Model Evaluation
        if stage in ['all', 'evaluation']:
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 5: MODEL EVALUATION")
            logger.info("=" * 80)
            results['evaluation'] = run_evaluation(config)
            logger.info(f"Evaluation complete: {results['evaluation']}")
        
        # Stage 6: Model Export
        if stage in ['all', 'export']:
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 6: MODEL EXPORT FOR SERVING")
            logger.info("=" * 80)
            results['export'] = export_model_for_serving(config)
            logger.info(f"Model export complete: {results['export']}")
        
        # Stage 7: Monitoring Setup
        if stage in ['all', 'monitoring']:
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 7: MONITORING DASHBOARD SETUP")
            logger.info("=" * 80)
            results['monitoring'] = setup_monitoring_dashboard(config)
            logger.info(f"Monitoring setup complete: {results['monitoring']}")
        
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 80)
        for stage_name, result in results.items():
            logger.info(f"{stage_name}: {result}")
        
        logger.info(f"\nEnd time: {datetime.now().isoformat()}")
        logger.info("Pipeline completed successfully! ✅")
        
        return results
    
    except Exception as e:
        logger.error(f"Pipeline failed at stage {stage}: {str(e)}", exc_info=True)
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Banking Fraud Detection Pipeline')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--stage', type=str, default='all',
                        choices=['all', 'ingestion', 'eda', 'features', 'training', 
                                 'evaluation', 'export', 'monitoring'],
                        help='Pipeline stage to run')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(config)
    
    # Run pipeline
    results = run_pipeline(config, args.stage)
    
    return results


if __name__ == '__main__':
    main()
