from pipelines.data_engineering import DataEngineeringPipeline
from pipelines.data_fetching import DataFetchingPipeline
from pipelines.machine_learning_pipeline import MachineLearningPipeline
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
    
    # data_fetching_pipeline.run()
    
    print("\n")
    
    data_engineering_pipeline = DataEngineeringPipeline(logger=logger, mode=mode)
    
    # data_engineering_pipeline.run()
    
    # exit()
    
    print("\n")
    
    machine_learning_pipeline = MachineLearningPipeline(logger=logger, mode=mode)
    
    machine_learning_pipeline.run()
            
    
    
    
if __name__ == "__main__":
    main()