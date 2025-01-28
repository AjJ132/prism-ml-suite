import os
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, TimeSeriesSplit
from scipy.stats import uniform
from logging import Logger
import matplotlib.pyplot as plt


class LinearRegressionTrainer:
    def __init__(self, logger: Logger, results_path: str, model_type: str = "linear"):
        """
        Initialize LinearRegressionTrainer
        
        Args:
            logger (Logger): Logger instance for tracking training process
            results_path (str): Path to save training results and parameters
            model_type (str): Type of linear model ('linear', 'ridge', 'lasso', 'elasticnet')
        """
        self.logger = logger
        self.results_path = results_path
        self.model_type = model_type.lower()
        
        valid_models = ["linear", "ridge", "lasso", "elasticnet"]
        if self.model_type not in valid_models:
            raise ValueError(f"model_type must be one of {valid_models}")

    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> object:
        """
        Train linear regression model with optional regularization and parameter optimization
        
        Args:
            x_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target values
            
        Returns:
            Trained linear model with optimized parameters if applicable
        """
        self.logger.info(f"Training {self.model_type} regression model")
        
        if self.model_type == "linear":
            # Standard linear regression doesn't require parameter tuning
            model = LinearRegression(n_jobs=-1)
            model.fit(x_train, y_train)
            self._save_model_info(model)
            return model
            
        # For regularized models, perform parameter optimization
        param_dist = self._get_param_distribution()
        base_model = self._get_base_model()
        
        # Initialize RandomizedSearchCV with time series split
        tscv = TimeSeriesSplit(n_splits=5)
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
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
        
        # Perform focused grid search around best parameters
        grid_param = self._get_grid_params(best_random)
        
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
        
        return grid_search.best_estimator_

    def _get_param_distribution(self) -> dict:
        """Get parameter distribution for random search based on model type"""
        if self.model_type == "ridge":
            return {
                'alpha': uniform(0, 100)
            }
        elif self.model_type == "lasso":
            return {
                'alpha': uniform(0, 100)
            }
        else:  # elasticnet
            return {
                'alpha': uniform(0, 100),
                'l1_ratio': uniform(0, 1)
            }

    def _get_base_model(self) -> object:
        """Get base model instance based on model type"""
        if self.model_type == "ridge":
            return Ridge(random_state=42)
        elif self.model_type == "lasso":
            return Lasso(random_state=42)
        else:  # elasticnet
            return ElasticNet(random_state=42)

    def _get_grid_params(self, best_random: dict) -> dict:
        """Get focused grid search parameters based on random search results"""
        if self.model_type in ["ridge", "lasso"]:
            return {
                'alpha': [
                    max(0, best_random['alpha'] - 10),
                    best_random['alpha'],
                    best_random['alpha'] + 10
                ]
            }
        else:  # elasticnet
            return {
                'alpha': [
                    max(0, best_random['alpha'] - 10),
                    best_random['alpha'],
                    best_random['alpha'] + 10
                ],
                'l1_ratio': [
                    max(0, best_random['l1_ratio'] - 0.1),
                    best_random['l1_ratio'],
                    min(1, best_random['l1_ratio'] + 0.1)
                ]
            }

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

    def _save_model_info(self, model: LinearRegression):
        """
        Save model coefficients and intercept
        
        Args:
            model (LinearRegression): Trained linear regression model
        """
        results_dir = os.path.join(self.results_path, 'model_info')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save coefficients
        coef_df = pd.DataFrame({
            'feature': range(len(model.coef_)),
            'coefficient': model.coef_
        })
        coef_df.to_csv(os.path.join(results_dir, 'coefficients.csv'), index=False)
        
        # Save intercept
        with open(os.path.join(results_dir, 'model_info.json'), 'w') as f:
            json.dump({
                'intercept': float(model.intercept_),
                'model_type': self.model_type
            }, f, indent=4)

    def save_model(self, model: object, position_dir: str):
        """
        Save the trained linear model
        
        Args:
            model: Trained linear model
            position_dir (str): Directory to save the model
        """
        self.logger.info(f"Saving {self.model_type} model")
        
        # Create model directory if it doesn't exist
        os.makedirs(position_dir, exist_ok=True)
        
        # Save model parameters and coefficients
        params_dict = {
            'model_type': self.model_type,
            'coefficients': model.coef_.tolist(),
            'intercept': float(model.intercept_)
        }
        
        if hasattr(model, 'alpha'):
            params_dict['alpha'] = model.alpha
        if hasattr(model, 'l1_ratio'):
            params_dict['l1_ratio'] = model.l1_ratio
            
        params_path = os.path.join(position_dir, 'model_params.json')
        with open(params_path, 'w') as f:
            json.dump(params_dict, f, indent=4)
            
        self.logger.info(f"Model parameters saved to {params_path}")