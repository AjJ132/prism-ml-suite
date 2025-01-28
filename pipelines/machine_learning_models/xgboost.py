import os
import json
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, TimeSeriesSplit
from scipy.stats import randint, uniform
from logging import Logger
import matplotlib.pyplot as plt



class XGBoostTrainer:
    def __init__(self, logger: Logger, results_path: str):
        """
        Initialize XGBoostTrainer
        
        Args:
            logger (Logger): Logger instance for tracking training process
            results_path (str): Path to save training results and parameters
        """
        self.logger = logger
        self.results_path = results_path
        
    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
        """
        Train XGBoost model with two-stage parameter optimization
        
        Args:
            x_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target values
            
        Returns:
            XGBRegressor: Trained model with optimized parameters
        """
        self.logger.info("Training XGBoost Model with two-stage search")
        
        # Stage 1: Broad random search
        random_param_dist = {
            'n_estimators': randint(100, 1000),
            'learning_rate': uniform(0.01, 0.3),
            'max_depth': randint(3, 12),
            'min_child_weight': randint(1, 7),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'gamma': uniform(0, 5),
            'reg_alpha': uniform(0, 2),
            'reg_lambda': uniform(0, 2)
        }
        
        # Initialize base model
        base_model = XGBRegressor(
            objective='reg:squarederror',
            n_jobs=-1,
            random_state=42,
            verbosity=0
        )
        
        # Initialize RandomizedSearchCV with time series split
        tscv = TimeSeriesSplit(n_splits=5)
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=random_param_dist,
            n_iter=100,
            cv=tscv,
            scoring='neg_mean_squared_error',
            verbose=2,
            n_jobs=-1
        )
        
        # Fit RandomizedSearchCV
        self.logger.info("Starting RandomizedSearchCV...")
        random_search.fit(x_train, y_train)
        
        # Get best parameters from random search
        best_random = random_search.best_params_
        
        # Stage 2: Focused grid search around best parameters
        grid_param = {
            'n_estimators': [max(100, best_random['n_estimators'] - 100),
                           best_random['n_estimators'],
                           best_random['n_estimators'] + 100],
            'learning_rate': [max(0.01, best_random['learning_rate'] - 0.01),
                            best_random['learning_rate'],
                            min(0.3, best_random['learning_rate'] + 0.01)],
            'max_depth': [max(3, best_random['max_depth'] - 1),
                         best_random['max_depth'],
                         best_random['max_depth'] + 1],
            'min_child_weight': [max(1, best_random['min_child_weight'] - 1),
                               best_random['min_child_weight'],
                               best_random['min_child_weight'] + 1],
            'subsample': [best_random['subsample']],
            'colsample_bytree': [best_random['colsample_bytree']],
            'gamma': [best_random['gamma']],
            'reg_alpha': [best_random['reg_alpha']],
            'reg_lambda': [best_random['reg_lambda']]
        }
        
        # Initialize GridSearchCV with focused parameter grid
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=grid_param,
            cv=tscv,
            scoring='neg_mean_squared_error',
            verbose=2,
            n_jobs=-1
        )
        
        # Fit GridSearchCV
        self.logger.info("Starting focused GridSearchCV...")
        grid_search.fit(x_train, y_train)
        
        # Save results
        self._save_search_results(random_search, grid_search)
        
        # Save feature importance plot and metrics
        self._save_feature_importance(grid_search.best_estimator_, x_train.columns)
        
        return grid_search.best_estimator_
    
    
    
    def _save_search_results(self, random_search: RandomizedSearchCV, grid_search: GridSearchCV):
        """
        Save parameter search results to files
        
        Args:
            random_search (RandomizedSearchCV): Completed random search
            grid_search (GridSearchCV): Completed grid search
        """
        results_dir = os.path.join(self.results_path, 'parameter_search_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save random search results
        random_cv_results = pd.DataFrame(random_search.cv_results_)
        random_cv_results.to_csv(
            os.path.join(results_dir, 'random_search_results.csv'), 
            index=False
        )
        
        # Save grid search results
        grid_cv_results = pd.DataFrame(grid_search.cv_results_)
        grid_cv_results.to_csv(
            os.path.join(results_dir, 'grid_search_results.csv'), 
            index=False
        )
        
        # Save final parameters
        with open(os.path.join(results_dir, 'best_parameters.json'), 'w') as f:
            json.dump({
                'random_search_best': random_search.best_params_,
                'final_best': grid_search.best_params_
            }, f, indent=4)
            
    def _save_feature_importance(self, model: XGBRegressor, feature_names: list):
        """
        Save feature importance plot and scores
        
        Args:
            model (XGBRegressor): Trained XGBoost model
            feature_names (list): List of feature names
        """
        results_dir = os.path.join(self.results_path, 'feature_importance')
        os.makedirs(results_dir, exist_ok=True)
        
        # Get feature importance scores (both gain and weight)
        importance_gain = model.get_booster().get_score(importance_type='gain')
        importance_weight = model.get_booster().get_score(importance_type='weight')
        
        # Convert to DataFrames
        importance_gain_df = pd.DataFrame({
            'feature': list(importance_gain.keys()),
            'importance_gain': list(importance_gain.values())
        }).sort_values('importance_gain', ascending=False)
        
        importance_weight_df = pd.DataFrame({
            'feature': list(importance_weight.keys()),
            'importance_weight': list(importance_weight.values())
        }).sort_values('importance_weight', ascending=False)
        
        # Save feature importance scores
        importance_gain_df.to_csv(
            os.path.join(results_dir, 'feature_importance_gain.csv'), 
            index=False
        )
        importance_weight_df.to_csv(
            os.path.join(results_dir, 'feature_importance_weight.csv'), 
            index=False
        )
        
        # Create and save feature importance plots
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.bar(range(len(importance_gain_df)), importance_gain_df['importance_gain'])
        plt.xticks(range(len(importance_gain_df)), importance_gain_df['feature'], 
                  rotation=45, ha='right')
        plt.title('Feature Importance (Gain)')
        
        plt.subplot(1, 2, 2)
        plt.bar(range(len(importance_weight_df)), importance_weight_df['importance_weight'])
        plt.xticks(range(len(importance_weight_df)), importance_weight_df['feature'], 
                  rotation=45, ha='right')
        plt.title('Feature Importance (Weight)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'feature_importance.png'))
        plt.close()
        
        # Save additional model metrics
        # self._save_model_metrics(model, results_dir)
        
        
    def save_model(self, model: XGBRegressor, position_dir: str):
        """
        Save the trained XGBoost model
        
        Args:
            model (XGBRegressor): Trained XGBoost model
            position_dir (str): Directory to save the model
        """
        self.logger.info("Saving XGBoost model")
        
        # Create model directory if it doesn't exist
        os.makedirs(position_dir, exist_ok=True)
        
        # Save model in XGBoost binary format
        model_path = os.path.join(position_dir, 'model.json')
        model.save_model(model_path)
        
        # Save model parameters
        params_path = os.path.join(position_dir, 'model_params.json')
        with open(params_path, 'w') as f:
            json.dump(model.get_params(), f, indent=4)
            
        self.logger.info(f"Model saved to {model_path}")