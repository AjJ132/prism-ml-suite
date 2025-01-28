from pipelines.data_engineering import DataEngineeringPipeline
from pipelines.data_fetching import DataFetchingPipeline
from pipelines.machine_learning_pipeline import MachineLearningPipeline
from pipelines.save_results import SaveResultsPipeline
from util.logging.logger import setup_logging
import os


def main():
    #initalize mode, 0 = normal, 1 = destructive
    mode = 0
    #initalize logger
    logger = setup_logging()
    logger.info("Starting prism-ml-suite")
    
    logger.info("Begining folder check")
    
    folders = [
        'data',
        'data/ml_ready',
        'data/models',
        'data/output',
        'data/raw',
        'data/raw/by_position',
        'data/raw/stats',
        'data/engineered'
    ]
    
    valid_positions = [
        'QB',
        'RB',
        'WR'
    ]
    
    model_test_season = 2024
    upcoming_season = 2025
    
    # ensure directories exist, if mode = destructive, delete all files in all folders
    for folder in folders:
        if mode == 1:
            logger.info("Deleting all files in " + folder)
            # delete all files in folder
            for file in os.listdir(folder):
                os.remove(os.path.join(folder, file))
        else:
            logger.info("Checking if " + folder + " exists")
            # check if folder exists
            # if not, create folder
            if not os.path.exists(folder):
                logger.info("Creating " + folder)
                os.makedirs(folder)
                
    logger.info("File System Initialized")
    logger.info("Prism-ml-suite is ready to use")
    
    data_fetching_pipeline = DataFetchingPipeline(logger, valid_positions, mode)
    
    data_fetching_pipeline.run()
    
    print("\n")
    
    data_engineering_pipeline = DataEngineeringPipeline(logger=logger, mode=mode, upcoming_season=upcoming_season)
    
    data_engineering_pipeline.run()
    
    # exit()
    
    print("\n")
    
    machine_learning_pipeline = MachineLearningPipeline(logger=logger, mode=mode, upcoming_season=upcoming_season, test_season=model_test_season)
    
    machine_learning_pipeline.run()
    
    save_results_pipeline = SaveResultsPipeline(logger=logger, mode=mode, predictions_season=upcoming_season)
    
    save_results_pipeline.run()
            
    
    
    
if __name__ == "__main__":
    main()