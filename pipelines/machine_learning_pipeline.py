import datetime
import json
import os
import glob
import re
from logging import Logger
import pickle


from sklearn.metrics import r2_score

from database.db_manager import get_db
from database.models.vw_player_passing_stats import VwPlayerPassingStats
from database.models.vw_player_receiving_stats import VwPlayerReceivingStats
from database.models.vw_player_rushing_stats import VwPlayerRushingStats
from database.models.vw_players import VwPlayers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np


class MachineLearningPipeline:
    def __init__(self, logger: Logger, mode: int, engineered_data_path: str = "data/engineered", machine_learning_path: str = "data/ml_ready"):
        self.logger = logger
        self.mode = mode
        self.engineered_data_path = engineered_data_path
        self.machine_learning_base_path = machine_learning_path  # Renamed to indicate this is the base path
        self.machine_learning_path = None  # Will be set when get_new_run_path is called
        self.players = None
        self.test_season = 2024
        
    def get_new_run_path(self):
        """
        Creates a new versioned run directory and returns its path.
        Format: model_run_XXX where XXX is an incrementing number
        """
        # Get all existing run directories
        run_pattern = os.path.join(self.machine_learning_base_path, "model_run_*")
        existing_runs = glob.glob(run_pattern)
        
        # Find the highest run number
        max_run_num = 0
        for run in existing_runs:
            match = re.search(r'model_run_(\d{3})', run)
            if match:
                run_num = int(match.group(1))
                max_run_num = max(max_run_num, run_num)
        
        # Create new run number and directory
        new_run_num = max_run_num + 1
        new_run_dir = os.path.join(self.machine_learning_base_path, f"model_run_{new_run_num:03d}")
        
        # Create the directory and necessary subdirectories
        os.makedirs(new_run_dir, exist_ok=True)
        
        # Log the creation of new run directory
        self.logger.info(f"Created new model run directory: {new_run_dir}")
        
        return new_run_dir
    
    def compare_runs(self, run_numbers=None):
        """
        Compare metrics across different runs.
        If run_numbers is None, compares all runs.
        """
        run_pattern = os.path.join(self.machine_learning_base_path, "model_run_*")
        available_runs = glob.glob(run_pattern)
        
        if run_numbers:
            runs_to_compare = [os.path.join(self.machine_learning_base_path, f"model_run_{num:03d}") 
                             for num in run_numbers]
        else:
            runs_to_compare = available_runs
        
        comparison_results = {}
        
        for run_path in runs_to_compare:
            run_name = os.path.basename(run_path)
            comparison_results[run_name] = {}
            
            # Get all position directories in the run
            position_dirs = glob.glob(os.path.join(run_path, "*"))
            for position_dir in position_dirs:
                if not os.path.isdir(position_dir):
                    continue
                    
                position = os.path.basename(position_dir)
                metadata_path = os.path.join(position_dir, 'run_metadata.json')
                
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        comparison_results[run_name][position] = metadata
        
        return comparison_results
        
    def train_random_forest(self, x_train, y_train):
        """
        Train Random Forest model with extensive hyperparameter tuning
        """
        self.logger.info("Training Random Forest Model with GridSearchCV")
        
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import GridSearchCV
        
        # Define an extensive parameter grid
        param_grid = {
            'n_estimators': [100, 200, 500, 1000],
            'max_depth': [10, 20, 30, 40, 50, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
            'max_samples': [0.7, 0.8, 0.9, None],
            'min_impurity_decrease': [0.0, 0.01, 0.1]
        }
        
        # Initialize base model
        base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=5,  # 5-fold cross-validation
            scoring='neg_mean_squared_error',  # Using MSE as our metric
            verbose=2,  # Detailed output
            n_jobs=-1  # Use all available cores
        )
        
        # Fit GridSearchCV
        self.logger.info("Starting GridSearchCV fit - this may take a while...")
        grid_search.fit(x_train, y_train)
        
        # Log best parameters and score
        self.logger.info(f"Best parameters found: {grid_search.best_params_}")
        self.logger.info(f"Best cross-validation score: {-grid_search.best_score_}")  # Negative because of scoring metric
        
        # Save grid search results
        results_dir = os.path.join(self.machine_learning_path, 'grid_search_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save detailed CV results
        cv_results = pd.DataFrame(grid_search.cv_results_)
        cv_results.to_csv(os.path.join(results_dir, 'grid_search_results.csv'), index=False)
        
        # Save best parameters
        with open(os.path.join(results_dir, 'best_parameters.json'), 'w') as f:
            json.dump(grid_search.best_params_, f, indent=4)
        
        return grid_search.best_estimator_
    
    def prepare_data(self, data, position):
        """
        Prepare data for machine learning by splitting features and target, excluding first-year players
        """
        self.logger.info(f"Preparing {position} data for machine learning")
        
        print("Original data shape:", data.shape)
        
        # Identify first-year players (all previous stats are 0)
        # previous_stat_columns = [col for col in data.columns if 'previous_' in col or 'ewma' in col]
        # new_player_mask = (data[previous_stat_columns] == 0).all(axis=1)
        
        # # Remove first-year players
        # data = data[~new_player_mask]
        # print("Data shape after removing first-year players:", data.shape)
        # print("Unique seasons in data:", sorted(data['season'].unique()))
        
        # Store player IDs before dropping
        player_ids = data['master_player_id']
        
        # Drop player ID and season columns
        features_data = data.drop(['master_player_id'], axis=1)
        
        target_map = {
            'QB': 'passing_yards',
            'RB': 'rushing_yards',
            'WR': 'receiving_yards'
        }
        target_col = target_map[position]
        
        y = features_data[target_col]
        X = features_data.drop([target_col, 'season'], axis=1)
        
        train_mask = features_data['season'] < self.test_season
        test_mask = features_data['season'] == self.test_season
        
        print("Number of training samples:", train_mask.sum())
        print("Number of test samples:", test_mask.sum())
        print("Training seasons:", sorted(features_data[train_mask]['season'].unique()))
        print("Test seasons:", sorted(features_data[test_mask]['season'].unique()))
        
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        player_ids_test = player_ids[test_mask]
        
        print("X_train shape:", X_train.shape)
        print("X_test shape:", X_test.shape)
        
        # Save to csv
        X_train.to_csv(os.path.join(self.machine_learning_path, 'X_train.csv'), index=False)
        X_test.to_csv(os.path.join(self.machine_learning_path, 'X_test.csv'), index=False)
        y_train.to_csv(os.path.join(self.machine_learning_path, 'y_train.csv'), index=False)
        y_test.to_csv(os.path.join(self.machine_learning_path, 'y_test.csv'), index=False)
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test, player_ids_test


    def evaluate_model(self, model, X_test, y_test, position, player_ids):
        """
        Comprehensive evaluation function with multiple metrics and diagnostic plots
        """
        self.logger.info(f"Evaluating {position} model")
        results_dir = os.path.join(self.machine_learning_path, position)
        os.makedirs(results_dir, exist_ok=True)
        
        # Get predictions for both training and test sets
        y_pred = model.predict(X_test)
        
        # Calculate various metrics
        r2_test = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Print comprehensive metrics
        print(f"\nModel Performance Metrics for {position}:")
        print(f"R² Score (Test): {r2_test:.4f}")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        
        # Save metrics to file
        metrics_dict = {
            'r2_test': r2_test,
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }
        
        with open(os.path.join(results_dir, 'metrics.txt'), 'w') as f:
            for metric, value in metrics_dict.items():
                f.write(f"{metric}: {value}\n")
        
        # 1. Scatter plot with perfect prediction line
        plt.figure(figsize=(10, 10))
        plt.scatter(y_test, y_pred, alpha=0.5)
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        line = np.linspace(min_val, max_val, 100)
        plt.plot(line, line, 'r--', label='Perfect Prediction')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{position} Model: Predicted vs Actual Values\nR² = {r2_test:.4f}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axis('equal')
        plt.savefig(os.path.join(results_dir, 'prediction_scatter.png'))
        plt.close()
        
        # 2. Feature Importance Plot
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(feature_importance)), feature_importance['importance'])
        plt.xticks(range(len(feature_importance)), feature_importance['feature'], rotation=45, ha='right')
        plt.title(f'{position} Model: Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'feature_importance.png'))
        plt.close()
        
        # Save feature importance to a text file
        with open(os.path.join(results_dir, 'feature_importance.txt'), 'w') as f:
            for index, row in feature_importance.iterrows():
                f.write(f"{row['feature']}: {row['importance']}\n")
        
        # 3. Residual Plot
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'{position} Model: Residual Plot')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(results_dir, 'residuals.png'))
        plt.close()
        
        # Save predictions vs actual with residuals for analysis
        results_df = pd.DataFrame({
            'master_player_id': player_ids,
            'actual': y_test,
            'predicted': y_pred,
            'residual': residuals,
            'abs_residual': np.abs(residuals)
        })
        
        # Add top 10 largest residuals analysis
        top_residuals = results_df.nlargest(10, 'abs_residual')
        print("\nTop 10 Largest Prediction Errors:")
        print(top_residuals[['master_player_id', 'actual', 'predicted', 'residual']])
        
        results_df.to_csv(os.path.join(results_dir, 'predictions_detailed.csv'), index=False)
        
        # Save model parameters
        with open(os.path.join(results_dir, 'model_params.txt'), 'w') as f:
            f.write("Model Parameters:\n")
            for param, value in model.get_params().items():
                f.write(f"{param}: {value}\n")
        
        return results_df, metrics_dict

    def run(self):
        self.logger.info("Running Machine Learning Pipeline")
        
        # Set up the versioned run directory
        self.machine_learning_path = self.get_new_run_path()
        
        # Load data
        qb_passing = pd.read_csv(os.path.join(self.engineered_data_path, "qb_passing_stats.csv"))
        rb_rushing = pd.read_csv(os.path.join(self.engineered_data_path, "rb_rushing_stats.csv"))
        wr_receiving = pd.read_csv(os.path.join(self.engineered_data_path, "wr_receiving_stats.csv"))
        
        # Process each position
        positions = {
            'QB': qb_passing,
            # 'RB': rb_rushing,
            # 'WR': wr_receiving
        }
        
        results = {}
        for position, data in positions.items():
            self.logger.info(f"Processing {position} data")
            
            # Create position-specific directory within the run directory
            position_dir = os.path.join(self.machine_learning_path, position)
            os.makedirs(position_dir, exist_ok=True)
            
            # Prepare data
            X_train, X_test, y_train, y_test, player_ids_test = self.prepare_data(data, position)
            
            # Train model
            model = self.train_random_forest(X_train, y_train)
            
            # Evaluate model
            report, metrics = self.evaluate_model(model, X_test, y_test, position, player_ids_test)
            
            # Store results
            results[position] = {
                'model': model,
                'report': report,
                'metrics': metrics
            }
            
            # Save model
            model_path = os.path.join(position_dir, 'model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Save run metadata
            metadata = {
                'run_timestamp': datetime.datetime.now().isoformat(),
                'test_season': self.test_season,
                'model_parameters': model.get_params(),
                'metrics': metrics
            }
            
            metadata_path = os.path.join(position_dir, 'run_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
        
        # Handle QB predictions
        players_to_analyze = [
            'f589f248-558c-48b2-8825-e396b6a19fa3',  # Carson Beck
            '160d493c-12d7-4e86-a685-eb83bed4f3bd',  # Davis Bryson
            '514fe10d-d116-42e4-878e-281adaf702e3'   # Jalen Milroe
        ]
        
        qb_predictions = pd.read_csv(os.path.join(self.machine_learning_path, 'QB', 'predictions_detailed.csv'))
        qb_predictions = qb_predictions[qb_predictions['master_player_id'].isin(players_to_analyze)]
        
        # Save predictions in the versioned run directory
        predictions_path = os.path.join(self.machine_learning_path, 'player_predictions.csv')
        qb_predictions.to_csv(predictions_path, index=False)
        
        return results