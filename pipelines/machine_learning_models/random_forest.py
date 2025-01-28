import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, TimeSeriesSplit
from scipy.stats import randint
from logging import Logger

class RandomForestTrainer:
    def __init__(self, logger: Logger, results_path: str, n_jobs: int = 16):
        """
        Initialize RandomForestTrainer
        
        Args:
            logger (Logger): Logger instance for tracking training process
            results_path (str): Path to save training results and parameters
            n_jobs (int): Number of CPU cores to use for parallel processing
        """
        self.logger = logger
        self.results_path = results_path
        self.n_jobs = n_jobs
        
    # def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    #     """
    #     Train Random Forest model with two-stage parameter optimization
        
    #     Args:
    #         x_train (pd.DataFrame): Training features
    #         y_train (pd.Series): Training target values
            
    #     Returns:
    #         RandomForestRegressor: Trained model with optimized parameters
    #     """
    #     self.logger.info("Training Random Forest Model with two-stage search")
        
    #     # Stage 1: Broad random search
    #     random_param_dist = {
    #         'n_estimators': randint(100, 1000),
    #         'max_depth': [10, 20, 30, 40, 50, None],
    #         'min_samples_split': randint(2, 10),
    #         'min_samples_leaf': randint(1, 4),
    #         'max_features': ['sqrt', 'log2', None],
    #         'bootstrap': [True, False]
    #     }
        
    #     # Initialize base model
    #     base_model = RandomForestRegressor(random_state=42, n_jobs=self.n_jobs)
        
    #     # Initialize RandomizedSearchCV
    #     tscv = TimeSeriesSplit(n_splits=5)
    #     random_search = RandomizedSearchCV(
    #         estimator=base_model,
    #         param_distributions=random_param_dist,
    #         n_iter=100,
    #         cv=tscv,
    #         scoring='neg_mean_squared_error',
    #         verbose=2,
    #         n_jobs=self.n_jobs
    #     )
        
    #     # Fit RandomizedSearchCV
    #     self.logger.info("Starting RandomizedSearchCV...")
    #     random_search.fit(x_train, y_train)
        
    #     # Get best parameters from random search
    #     best_random = random_search.best_params_
        
    #     # Stage 2: Focused grid search around best parameters
    #     grid_param = {}
    
    #     # Handle max_depth separately
    #     if best_random['max_depth'] is None:
    #         grid_param['max_depth'] = [None]
    #     else:
    #         grid_param['max_depth'] = [
    #             max(1, best_random['max_depth'] - 10),
    #             best_random['max_depth'],
    #             best_random['max_depth'] + 10
    #         ]
        
    #     # Add other parameters
    #     grid_param.update({
    #         'n_estimators': [best_random['n_estimators'] - 100, best_random['n_estimators'], best_random['n_estimators'] + 100],
    #         'min_samples_split': [max(2, best_random['min_samples_split'] - 2),
    #                             best_random['min_samples_split'],
    #                             best_random['min_samples_split'] + 2],
    #         'min_samples_leaf': [max(1, best_random['min_samples_leaf'] - 1),
    #                         best_random['min_samples_leaf'],
    #                         best_random['min_samples_leaf'] + 1],
    #         'max_features': [best_random['max_features']],
    #         'bootstrap': [best_random['bootstrap']]
    #     })

    #     # Initialize GridSearchCV with focused parameter grid
    #     grid_search = GridSearchCV(
    #         estimator=base_model,
    #         param_grid=grid_param,
    #         cv=5,
    #         scoring='neg_mean_squared_error',
    #         verbose=2,
    #         n_jobs=self.n_jobs
    #     )
            
    #     # Fit GridSearchCV
    #     self.logger.info("Starting focused GridSearchCV...")
    #     grid_search.fit(x_train, y_train)
        
    #     # Save results
    #     self._save_search_results(random_search, grid_search)
        
    #     return grid_search.best_estimator_
    
    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
        """
        Train Random Forest model with predefined parameters
        
        Args:
            x_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target values
            
        Returns:
            RandomForestRegressor: Trained model with specified parameters
        """
        self.logger.info("Training Random Forest Model with predefined parameters")
        
        # Initialize model with specified parameters
        model = RandomForestRegressor(
            bootstrap=True,
            max_depth=10,
            max_features='sqrt',
            min_samples_leaf=3,
            min_samples_split=9,
            n_estimators=936,
            n_jobs=self.n_jobs,
            random_state=42
        )
        
        # Fit the model
        self.logger.info("Training model...")
        model.fit(x_train, y_train)
        
        # Save parameters
        self._save_parameters(model)
        
        return model
    
    def _save_parameters(self, model: RandomForestRegressor):
        """
        Save model parameters to file
        
        Args:
            model (RandomForestRegressor): Trained model
        """
        results_dir = os.path.join(self.results_path, 'parameter_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save parameters
        with open(os.path.join(results_dir, 'model_parameters.json'), 'w') as f:
            json.dump({
                'parameters': model.get_params(),
            }, f, indent=4)
    
    # def _save_search_results(self, random_search: RandomizedSearchCV, grid_search: GridSearchCV):
    #     """
    #     Save parameter search results to files
        
    #     Args:
    #         random_search (RandomizedSearchCV): Completed random search
    #         grid_search (GridSearchCV): Completed grid search
    #     """
    #     results_dir = os.path.join(self.results_path, 'parameter_search_results')
    #     os.makedirs(results_dir, exist_ok=True)
        
    #     # Save random search results
    #     random_cv_results = pd.DataFrame(random_search.cv_results_)
    #     random_cv_results.to_csv(os.path.join(results_dir, 'random_search_results.csv'), index=False)
        
    #     # Save grid search results
    #     grid_cv_results = pd.DataFrame(grid_search.cv_results_)
    #     grid_cv_results.to_csv(os.path.join(results_dir, 'grid_search_results.csv'), index=False)
        
    #     # Save final parameters
    #     with open(os.path.join(results_dir, 'best_parameters.json'), 'w') as f:
    #         json.dump({
    #             'random_search_best': random_search.best_params_,
    #             'final_best': grid_search.best_params_
    #         }, f, indent=4)