import os
import json
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, TimeSeriesSplit
from scipy.stats import randint, uniform
from logging import Logger

class LightGBMTrainer:
    def __init__(self, logger: Logger, results_path: str):
        """
        Initialize LightGBMTrainer
        
        Args:
            logger (Logger): Logger instance for tracking training process
            results_path (str): Path to save training results and parameters
        """
        self.logger = logger
        self.results_path = results_path
        
    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> LGBMRegressor:
        """
        Train LightGBM model with predefined parameters
        
        Args:
            x_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target values
            
        Returns:
            LGBMRegressor: Trained model with specified parameters
        """
        self.logger.info("Training LightGBM Model with predefined parameters")
        
        # Initialize model with specified parameters
        model = LGBMRegressor(
            objective='regression',
            metric='rmse',
            n_jobs=-1,
            random_state=42,
            verbose=-1,
            # Predefined parameters (update these based on your optimal configuration)
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=50,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1
        )
        
        # Fit the model
        self.logger.info("Training model...")
        model.fit(x_train, y_train)
        
        # Save parameters and feature importance
        self._save_parameters(model)
        self._save_feature_importance(model, x_train.columns)
        
        return model
    
    def _save_parameters(self, model: LGBMRegressor):
        """
        Save model parameters to file
        
        Args:
            model (LGBMRegressor): Trained model
        """
        results_dir = os.path.join(self.results_path, 'parameter_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save parameters
        with open(os.path.join(results_dir, 'model_parameters.json'), 'w') as f:
            json.dump({
                'parameters': model.get_params(),
            }, f, indent=4)
            
    def _save_feature_importance(self, model: LGBMRegressor, feature_names: list):
        """
        Save feature importance plot and scores
        
        Args:
            model (LGBMRegressor): Trained LightGBM model
            feature_names (list): List of feature names
        """
        results_dir = os.path.join(self.results_path, 'feature_importance')
        os.makedirs(results_dir, exist_ok=True)
        
        # Get feature importance scores
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save feature importance scores
        importance_df.to_csv(
            os.path.join(results_dir, 'feature_importance.csv'), 
            index=False
        )
        
        # Create and save feature importance plot
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(importance_df)), importance_df['importance'])
        plt.xticks(range(len(importance_df)), importance_df['feature'], rotation=45, ha='right')
        plt.title('Feature Importance (LightGBM)')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'feature_importance.png'))
        plt.close()
        
    # def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> LGBMRegressor:
    #     """
    #     Train LightGBM model with two-stage parameter optimization
        
    #     Args:
    #         x_train (pd.DataFrame): Training features
    #         y_train (pd.Series): Training target values
            
    #     Returns:
    #         LGBMRegressor: Trained model with optimized parameters
    #     """
    #     self.logger.info("Training LightGBM Model with two-stage search")
        
    #     # Stage 1: Broad random search
    #     random_param_dist = {
    #         'n_estimators': randint(100, 1000),
    #         'learning_rate': uniform(0.01, 0.3),
    #         'max_depth': randint(3, 12),
    #         'num_leaves': randint(20, 100),
    #         'min_child_samples': randint(10, 50),
    #         'subsample': uniform(0.6, 0.4),
    #         'colsample_bytree': uniform(0.6, 0.4),
    #         'reg_alpha': uniform(0, 2),
    #         'reg_lambda': uniform(0, 2)
    #     }
        
    #     # Initialize base model
    #     base_model = LGBMRegressor(
    #         objective='regression',
    #         metric='rmse',
    #         n_jobs=-1,
    #         random_state=42,
    #         verbose=-1
    #     )
        
    #     # Initialize RandomizedSearchCV with time series split
    #     tscv = TimeSeriesSplit(n_splits=5)
    #     random_search = RandomizedSearchCV(
    #         estimator=base_model,
    #         param_distributions=random_param_dist,
    #         n_iter=100,
    #         cv=tscv,
    #         scoring='neg_mean_squared_error',
    #         verbose=2,
    #         n_jobs=-1
    #     )
        
    #     # Fit RandomizedSearchCV
    #     self.logger.info("Starting RandomizedSearchCV...")
    #     random_search.fit(x_train, y_train)
        
    #     # Get best parameters from random search
    #     best_random = random_search.best_params_
        
    #     # Stage 2: Focused grid search around best parameters
    #     grid_param = {
    #         'n_estimators': [max(100, best_random['n_estimators'] - 100),
    #                        best_random['n_estimators'],
    #                        best_random['n_estimators'] + 100],
    #         'learning_rate': [max(0.01, best_random['learning_rate'] - 0.01),
    #                         best_random['learning_rate'],
    #                         min(0.3, best_random['learning_rate'] + 0.01)],
    #         'max_depth': [max(3, best_random['max_depth'] - 1),
    #                      best_random['max_depth'],
    #                      best_random['max_depth'] + 1],
    #         'num_leaves': [max(20, best_random['num_leaves'] - 10),
    #                      best_random['num_leaves'],
    #                      min(100, best_random['num_leaves'] + 10)],
    #         'min_child_samples': [best_random['min_child_samples']],
    #         'subsample': [best_random['subsample']],
    #         'colsample_bytree': [best_random['colsample_bytree']],
    #         'reg_alpha': [best_random['reg_alpha']],
    #         'reg_lambda': [best_random['reg_lambda']]
    #     }
        
    #     # Initialize GridSearchCV with focused parameter grid
    #     grid_search = GridSearchCV(
    #         estimator=base_model,
    #         param_grid=grid_param,
    #         cv=tscv,
    #         scoring='neg_mean_squared_error',
    #         verbose=2,
    #         n_jobs=-1
    #     )
        
    #     # Fit GridSearchCV
    #     self.logger.info("Starting focused GridSearchCV...")
    #     grid_search.fit(x_train, y_train)
        
    #     # Save results
    #     self._save_search_results(random_search, grid_search)
        
    #     # Save feature importance plot
    #     self._save_feature_importance(grid_search.best_estimator_, x_train.columns)
        
    #     return grid_search.best_estimator_
    
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
    #     random_cv_results.to_csv(
    #         os.path.join(results_dir, 'random_search_results.csv'), 
    #         index=False
    #     )
        
    #     # Save grid search results
    #     grid_cv_results = pd.DataFrame(grid_search.cv_results_)
    #     grid_cv_results.to_csv(
    #         os.path.join(results_dir, 'grid_search_results.csv'), 
    #         index=False
    #     )
        
    #     # Save final parameters
    #     with open(os.path.join(results_dir, 'best_parameters.json'), 'w') as f:
    #         json.dump({
    #             'random_search_best': random_search.best_params_,
    #             'final_best': grid_search.best_params_
    #         }, f, indent=4)
            
    # def _save_feature_importance(self, model: LGBMRegressor, feature_names: list):
        """
        Save feature importance plot and scores
        
        Args:
            model (LGBMRegressor): Trained LightGBM model
            feature_names (list): List of feature names
        """
        results_dir = os.path.join(self.results_path, 'feature_importance')
        os.makedirs(results_dir, exist_ok=True)
        
        # Get feature importance scores
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save feature importance scores
        importance_df.to_csv(
            os.path.join(results_dir, 'feature_importance.csv'), 
            index=False
        )
        
        # Create and save feature importance plot
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(importance_df)), importance_df['importance'])
        plt.xticks(range(len(importance_df)), importance_df['feature'], rotation=45, ha='right')
        plt.title('Feature Importance (LightGBM)')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'feature_importance.png'))
        plt.close()