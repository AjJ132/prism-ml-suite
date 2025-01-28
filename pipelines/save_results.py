from logging import Logger
import pandas as pd

from database.db_manager import get_db
from database.models.qb_passing_yards_ml_predictions import QbPassYdsMlPredictions
from contextlib import contextmanager

from database.models.qb_rush_yards_ml_predictions import QbRushYdsMlPredictions
from database.models.rb_receiving_yards_ml_predictions import RbReceivingYdsMlPredictions
from database.models.rb_rush_yards_ml_predictions import RbRushYdsMlPredictions
from database.models.wr_receiving_yards_ml_predictions import WrReceivingYdsMlPredictions

@contextmanager
def transaction_scope(db):
    """Provide a transactional scope around a series of operations."""
    try:
        yield
        db.commit()
    except Exception as e:
        db.rollback()
        raise

class SaveResultsPipeline:
    def __init__(self, logger: Logger, mode: int, predictions_season: int, output_path: str = "data/output"):
        self.logger = logger
        self.mode = mode
        self.predictions_season = predictions_season
        self.output_path = output_path
        
        
        
    def save_qb_passing_results(self, filename: str):
        try:
            df = pd.read_csv(filename)
            
            #rename columns
            #example: random_forest_prediction -> random_forest_predicted_pass_yds
            df.rename(columns={
                'random_forest_prediction': 'random_forest_predicted_pass_yds',
                'lightgbm_prediction': 'lightgbm_predicted_pass_yds',
                'gradient_boosting_prediction': 'gradient_boosting_predicted_pass_yds',
                'lasso_prediction': 'lasso_predicted_pass_yds',
                'linear_prediction': 'linear_regression_predicted_pass_yds',
                'neural_network_prediction': 'neural_network_predicted_pass_yds',
                'ridge_prediction': 'ridge_predicted_pass_yds',
                'ensemble_prediction': 'ensemble_predicted_pass_yds'
            }, inplace=True)
            
            #add season column
            df['prediction_season'] = self.predictions_season
            
            with get_db(logger=self.logger) as db:
                with transaction_scope(db):
                    #save to database
                    QbPassYdsMlPredictions.upsert_from_dataframe(db, df)
                    
            self.logger.info(f"Succesfully saved QB passing results from {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save QB passing results from {filename}: {e}")
            raise

    def save_qb_rushing_results(self, filename: str):
        try:
            df = pd.read_csv(filename)
            
            #rename columns
            #example: random_forest_prediction -> random_forest_predicted_rush_yds
            df.rename(columns={
                'random_forest_prediction': 'random_forest_predicted_rush_yds',
                'lightgbm_prediction': 'lightgbm_predicted_rush_yds',
                'gradient_boosting_prediction': 'gradient_boosting_predicted_rush_yds',
                'lasso_prediction': 'lasso_predicted_rush_yds',
                'linear_prediction': 'linear_regression_predicted_rush_yds',
                'neural_network_prediction': 'neural_network_predicted_rush_yds',
                'ridge_prediction': 'ridge_predicted_rush_yds',
                'ensemble_prediction': 'ensemble_predicted_rush_yds'
            }, inplace=True)
            
            #add season column
            df['prediction_season'] = self.predictions_season
            
            with get_db(logger=self.logger) as db:
                with transaction_scope(db):
                    #save to database
                    QbRushYdsMlPredictions.upsert_from_dataframe(db, df)
                    
            self.logger.info(f"Succesfully saved QB rushing results from {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save QB rushing results from {filename}: {e}")
            raise
        
    def save_rb_rushing_results(self, filename: str):
        try:
            df = pd.read_csv(filename)
            
            #rename columns
            #example: random_forest_prediction -> random_forest_predicted_rush_yds
            df.rename(columns={
                'random_forest_prediction': 'random_forest_predicted_rush_yds',
                'lightgbm_prediction': 'lightgbm_predicted_rush_yds',
                'gradient_boosting_prediction': 'gradient_boosting_predicted_rush_yds',
                'lasso_prediction': 'lasso_predicted_rush_yds',
                'linear_prediction': 'linear_regression_predicted_rush_yds',
                'neural_network_prediction': 'neural_network_predicted_rush_yds',
                'ridge_prediction': 'ridge_predicted_rush_yds',
                'ensemble_prediction': 'ensemble_predicted_rush_yds'
            }, inplace=True)
            
            #add season column
            df['prediction_season'] = self.predictions_season
            
            with get_db(logger=self.logger) as db:
                with transaction_scope(db):
                    #save to database
                    RbRushYdsMlPredictions.upsert_from_dataframe(db, df)
                    
            self.logger.info(f"Succesfully saved RB rushing results from {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save RB rushing results from {filename}: {e}")
            raise
        
    def save_rb_receiving_results(self, filename: str):
        try:
            df = pd.read_csv(filename)
            
            #rename columns
            #example: random_forest_prediction -> random_forest_predicted_receiving_yds
            df.rename(columns={
                'random_forest_prediction': 'random_forest_predicted_receiving_yds',
                'lightgbm_prediction': 'lightgbm_predicted_receiving_yds',
                'gradient_boosting_prediction': 'gradient_boosting_predicted_receiving_yds',
                'lasso_prediction': 'lasso_predicted_receiving_yds',
                'linear_prediction': 'linear_regression_predicted_receiving_yds',
                'neural_network_prediction': 'neural_network_predicted_receiving_yds',
                'ridge_prediction': 'ridge_predicted_receiving_yds',
                'ensemble_prediction': 'ensemble_predicted_receiving_yds'
            }, inplace=True)
            
            #add season column
            df['prediction_season'] = self.predictions_season
            
            with get_db(logger=self.logger) as db:
                with transaction_scope(db):
                    #save to database
                    RbReceivingYdsMlPredictions.upsert_from_dataframe(db, df)
                    
            self.logger.info(f"Succesfully saved RB receiving results from {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save RB receiving results from {filename}: {e}")
            raise
    
    def save_wr_receiving_results(self, filename: str):
        try:
            df = pd.read_csv(filename)
            
            #rename columns
            #example: random_forest_prediction -> random_forest_predicted_receiving_yds
            df.rename(columns={
                'random_forest_prediction': 'random_forest_predicted_receiving_yds',
                'lightgbm_prediction': 'lightgbm_predicted_receiving_yds',
                'gradient_boosting_prediction': 'gradient_boosting_predicted_receiving_yds',
                'lasso_prediction': 'lasso_predicted_receiving_yds',
                'linear_prediction': 'linear_regression_predicted_receiving_yds',
                'neural_network_prediction': 'neural_network_predicted_receiving_yds',
                'ridge_prediction': 'ridge_predicted_receiving_yds',
                'ensemble_prediction': 'ensemble_predicted_receiving_yds'
            }, inplace=True)
            
            #add season column
            df['prediction_season'] = self.predictions_season
            
            with get_db(logger=self.logger) as db:
                with transaction_scope(db):
                    #save to database
                    WrReceivingYdsMlPredictions.upsert_from_dataframe(db, df)
                    
            self.logger.info(f"Succesfully saved WR receiving results from {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save WR receiving results from {filename}: {e}")
        
    def run(self):
        
        config = {
            'QB': {
                'passing': '/QB/passing/QB_passing_predictions.csv',
                'rushing': '/QB/rushing/QB_rushing_predictions.csv'
            },
            'RB': {
                'rushing': '/RB/rushing/RB_rushing_predictions.csv',
                'receiving': '/RB/receiving/RB_receiving_predictions.csv'
            },
            'WR': {
                'receiving': '/WR/receiving/WR_receiving_predictions.csv'
            }
        }
        
        for position, stats in config.items():
            for stat, filename in stats.items():
                if position == 'QB':
                    if stat == 'passing':
                        self.save_qb_passing_results(self.output_path + filename)
                    elif stat == 'rushing':
                        self.save_qb_rushing_results(self.output_path + filename)
                    else:
                        raise ValueError("Invalid stat type for QB")
                    
                elif position == 'RB':
                    if stat == 'rushing':
                        self.save_rb_rushing_results(self.output_path + filename)
                    elif stat == 'receiving':
                        self.save_rb_receiving_results(self.output_path + filename)
                    else:
                        raise ValueError("Invalid stat type for RB")
                    
                elif position == 'WR':
                    if stat == 'receiving':
                        self.save_wr_receiving_results(self.output_path + filename)
                    else:
                        raise ValueError("Invalid stat type for WR")