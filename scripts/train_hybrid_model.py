# train_hybrid_model.py

import os
import sys
import logging

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import the ModelTrainer class
from src.models.train_model import ModelTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'hybrid_training.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('hybrid_training')

if __name__ == "__main__":
    logger.info("Starting hybrid model training")
    
    # Create model trainer
    trainer = ModelTrainer()
    
    # Train using hybrid pipeline
    submission = trainer.run_hybrid_pipeline(
        transform_target=True,
        explore_transformations=True
    )
    
    logger.info("Hybrid model training complete")