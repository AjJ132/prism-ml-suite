import datetime
import json
import os
import glob
import re
from logging import Logger
import pickle
from sklearn.metrics import r2_score
import torch
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from pipelines.helpers.model_results_saver import ModelResultsSaver
from pipelines.machine_learning_models.linear_regression import LinearRegressionTrainer
from pipelines.machine_learning_models.light_gbm import LightGBMTrainer
from pipelines.machine_learning_models.neural_network import NeuralNetworkTrainer
from pipelines.machine_learning_models.random_forest import RandomForestTrainer
from pipelines.helpers.model_visualizer import ModelVisualizer

class MachineLearningPipeline:
    def __init__(self, logger: Logger, mode: int, upcoming_season: int, test_season: int,
                 engineered_data_path: str = "data/engineered", 
                 machine_learning_path: str = "data/ml_ready"):
        self.logger = logger
        self.mode = mode
        self.upcoming_season = upcoming_season
        self.test_season = test_season
        self.engineered_data_path = engineered_data_path
        self.machine_learning_base_path = machine_learning_path
        self.run_path = None  # Will store the newly created run folder
        self.models = [
            "random_forest", 
            "lightgbm", 
            "neural_network", 
            "linear"
        ]
        self.positions = {}
        self.target_features = {
            'QB': {
                'passing': 'passing_yards',
                'rushing': 'rush_yards'
            },
            'RB': {
                'rushing': 'rush_yards',
                'receiving': 'receiving_yards'
            },
            'WR': {
                'receiving': 'receiving_yards'
            }
        }

    def get_new_run_path(self):
        """Create a new top-level run directory once per script execution."""
        run_pattern = os.path.join(self.machine_learning_base_path, "model_run_*")
        existing_runs = glob.glob(run_pattern)

        max_run_num = 0
        for run in existing_runs:
            match = re.search(r'model_run_(\d{3})', run)
            if match:
                run_num = int(match.group(1))
                max_run_num = max(max_run_num, run_num)

        new_run_num = max_run_num + 1
        base_run_dir = os.path.join(self.machine_learning_base_path, f"model_run_{new_run_num:03d}")
        os.makedirs(base_run_dir, exist_ok=True)

        self.logger.info(f"Created new run directory: {base_run_dir}")
        return base_run_dir

    def train_model(self, model_type: str, x_train, y_train, position, stat_type):
        """Train model based on selected model type."""
        if model_type == "random_forest":
            trainer = RandomForestTrainer(self.logger, self.run_path)
            return trainer.train(x_train, y_train)
        elif model_type == "lightgbm":
            trainer = LightGBMTrainer(self.logger, self.run_path)
            return trainer.train(x_train, y_train)
        elif model_type == "linear":
            trainer = LinearRegressionTrainer(self.logger, self.run_path)
            return trainer.train(x_train, y_train)
        elif model_type == "neural_network":
            trainer = NeuralNetworkTrainer(self.logger, self.run_path)
            return trainer.train(x_train, y_train, position=position, stat_type=stat_type)
        else:
            raise ValueError(f"Invalid model type: {model_type}")

    def prepare_data(self, data, position, stat_type, is_upcoming=False):
        """Prepare data for either test season validation or upcoming season predictions."""
        self.logger.info(f"Preparing {position} {stat_type} data for {'upcoming season' if is_upcoming else 'test season'}")

        player_ids = data['master_player_id']
        features_data = data.drop(['master_player_id'], axis=1)
        target_col = self.target_features[position][stat_type]
        
        y = features_data[target_col]
        X = features_data.drop([target_col, 'season'], axis=1)

        if is_upcoming:
            # For upcoming season, use all historical data for training
            X_train = X[features_data['season'] <= self.test_season]
            y_train = y[features_data['season'] <= self.test_season]
            X_test = X[features_data['season'] == self.upcoming_season]
            y_test = y[features_data['season'] == self.upcoming_season]
            player_ids_test = player_ids[features_data['season'] == self.upcoming_season]
        else:
            # For test season validation
            train_mask = features_data['season'] < self.test_season
            test_mask = features_data['season'] == self.test_season
            X_train = X[train_mask]
            X_test = X[test_mask]
            y_train = y[train_mask]
            y_test = y[test_mask]
            player_ids_test = player_ids[test_mask]

        return X_train, X_test, y_train, y_test, player_ids_test, target_col

    def evaluate_and_save(self, model, model_type, X_test, y_test, position, stat_type, player_ids, target_col, is_upcoming=False):
        """Evaluate model and save results to appropriate directory."""
        self.logger.info(f"Evaluating {position} {stat_type} model for {'upcoming season' if is_upcoming else 'test season'}")
        
        base_path = "data/output" if is_upcoming else self.run_path
        results_dir = os.path.join(base_path, model_type, position, stat_type)
        os.makedirs(results_dir, exist_ok=True)

        # Get predictions
        if model_type == "neural_network":
            trainer = NeuralNetworkTrainer(self.logger, self.run_path)
            y_pred = trainer.predict(model, X_test, position=position, stat_type=stat_type)
        else:
            y_pred = model.predict(X_test)

        # Calculate metrics
        metrics_dict = {
            'r2_test': r2_score(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'target_feature': target_col
        }

        # Save results
        results_saver = ModelResultsSaver(self.logger)
        results_saver.save_metrics(metrics_dict, results_dir)
        results_df = results_saver.save_detailed_results(y_test, y_pred, player_ids, f"{position}_{stat_type}", results_dir)

        # Create visualizations only for test season validation
        if not is_upcoming:
            visualizer = ModelVisualizer()
            visualizer.create_scatter_plot(y_test, y_pred, metrics_dict['r2_test'], f"{position}_{stat_type}", results_dir)
            visualizer.create_residual_plot(y_test, y_pred, f"{position}_{stat_type}", model_type, results_dir)
            
            if model_type in ["random_forest", "lightgbm"]:
                visualizer.create_feature_importance_plot(model, X_test, f"{position}_{stat_type}", model_type, results_dir)

        return results_df, metrics_dict

    def organize_and_save_predictions(self, all_results, is_upcoming=False):
        """
        Organizes predictions from all models and creates ensemble predictions with confidence intervals.
        
        Args:
            all_results: Dictionary containing results from all models
            is_upcoming: Boolean indicating if these are predictions for the upcoming season
        """
        # Set base path based on whether this is for upcoming season or test season
        output_base_path = "data/output" if is_upcoming else self.run_path
        os.makedirs(output_base_path, exist_ok=True)
        
        # Process each position and stat type separately
        for position in self.positions.keys():
            for stat_type in self.target_features[position].keys():
                position_predictions = {}
                player_ids = None
                target_col = None
                
                # Collect predictions from each model
                for model_type in self.models:
                    predictions_path = os.path.join(
                        "data/output" if is_upcoming else self.run_path,
                        model_type, 
                        position, 
                        stat_type, 
                        'predictions_detailed.csv'
                    )
                    if os.path.exists(predictions_path):
                        pred_df = pd.read_csv(predictions_path)
                        
                        if player_ids is None:
                            player_ids = pred_df['master_player_id']
                            target_col = self.target_features[position][stat_type]
                        
                        position_predictions[f'{model_type}_prediction'] = pred_df['predicted']
                
                if position_predictions:
                    # Create combined DataFrame
                    combined_df = pd.DataFrame({
                        'master_player_id': player_ids,
                        'target_feature': target_col
                    })
                    
                    # Add individual model predictions
                    for model_name, preds in position_predictions.items():
                        combined_df[model_name] = preds
                    
                    # Calculate ensemble prediction
                    prediction_columns = [col for col in combined_df.columns if col.endswith('_prediction')]
                    if prediction_columns:
                        # Mean prediction
                        combined_df['ensemble_prediction'] = combined_df[prediction_columns].mean(axis=1)
                        
                        # Calculate prediction intervals with scaling based on prediction magnitude
                        predictions_array = combined_df[prediction_columns].values
                        
                        # Scale confidence intervals based on prediction magnitude
                        prediction_magnitude = combined_df['ensemble_prediction']
                        magnitude_factor = np.clip(prediction_magnitude / prediction_magnitude.mean(), 0.5, 2.0)
                        
                        # Standard deviation across model predictions
                        prediction_std = np.std(predictions_array, axis=1)
                        
                        # Calculate historical error rate based on individual model performance
                        model_errors = []
                        for col in prediction_columns:
                            if 'actual_value' in combined_df.columns:
                                mae = mean_absolute_error(combined_df['actual_value'], combined_df[col])
                                model_errors.append(mae)
                        
                        # Average historical error
                        avg_historical_error = np.mean(model_errors) if model_errors else prediction_std.mean()
                        
                        # Use the larger of model variance or historical error for confidence bounds
                        # We use 1.96 for 95% confidence interval
                        base_interval = np.maximum(prediction_std, avg_historical_error) * 1.96
                        confidence_interval = base_interval * magnitude_factor
                        
                        # Add prediction intervals to DataFrame
                        combined_df['prediction_lower_bound'] = combined_df['ensemble_prediction'] - confidence_interval
                        combined_df['prediction_upper_bound'] = combined_df['ensemble_prediction'] + confidence_interval
                        combined_df['confidence_interval'] = confidence_interval
                        
                        # Ensure lower bounds aren't negative for yards
                        if 'yards' in target_col.lower():
                            combined_df['prediction_lower_bound'] = combined_df['prediction_lower_bound'].clip(lower=0)
                        
                        # Calculate uncertainty percentage
                        combined_df['uncertainty_percentage'] = (confidence_interval / combined_df['ensemble_prediction'] * 100).round(1)
                    
                    # Add actual values if available (only for test season)
                    if not is_upcoming and 'actual_value' in pred_df.columns:
                        combined_df['actual_value'] = pred_df['actual_value']
                    
                    # Save to appropriate folder
                    position_path = os.path.join(output_base_path, position, stat_type)
                    os.makedirs(position_path, exist_ok=True)
                    output_file = os.path.join(position_path, f'{position}_{stat_type}_predictions.csv')
                    combined_df.to_csv(output_file, index=False)
                    
                    self.logger.info(f"Saved combined predictions for {position} {stat_type} to {output_file}")
                    
                    # Create summary statistics only if we have actual values (test season)
                    if not is_upcoming and 'actual_value' in combined_df.columns:
                        summary_stats = pd.DataFrame({
                            'model': prediction_columns + ['ensemble_prediction'],
                            'target_feature': target_col,
                            'mae': [mean_absolute_error(combined_df['actual_value'], 
                                                    combined_df[col]) for col in prediction_columns + ['ensemble_prediction']],
                            'rmse': [np.sqrt(mean_squared_error(combined_df['actual_value'], 
                                                            combined_df[col])) for col in prediction_columns + ['ensemble_prediction']],
                            'r2': [r2_score(combined_df['actual_value'], 
                                        combined_df[col]) for col in prediction_columns + ['ensemble_prediction']]
                        })
                        
                        # Add average uncertainty metrics to summary
                        summary_stats.loc[len(summary_stats)] = [
                            'prediction_intervals',
                            target_col,
                            combined_df['confidence_interval'].mean(),
                            combined_df['confidence_interval'].std(),
                            combined_df['uncertainty_percentage'].mean()
                        ]
                        
                        # Save summary statistics
                        stats_file = os.path.join(position_path, f'{position}_{stat_type}_model_performance_summary.csv')
                        summary_stats.to_csv(stats_file, index=False)
                        self.logger.info(f"Saved performance summary for {position} {stat_type} to {stats_file}")

    def run(self):
        """Run the pipeline for both test season validation and upcoming season predictions."""
        self.logger.info("Running Machine Learning Pipeline")
        self.run_path = self.get_new_run_path()

        # Load all data files
        data_files = {
            'QB': {
                'passing': pd.read_csv(os.path.join(self.engineered_data_path, "qb_passing_stats.csv")),
                'rushing': pd.read_csv(os.path.join(self.engineered_data_path, "qb_rushing_stats.csv"))
            },
            'RB': {
                'rushing': pd.read_csv(os.path.join(self.engineered_data_path, "rb_rushing_stats.csv")),
                'receiving': pd.read_csv(os.path.join(self.engineered_data_path, "rb_receiving_stats.csv"))
            },
            'WR': {
                'receiving': pd.read_csv(os.path.join(self.engineered_data_path, "wr_receiving_stats.csv"))
            }
        }
        
        self.positions = data_files
        all_results = {'test_season': {}, 'upcoming_season': {}}

        # Process each model type
        for model_type in self.models:
            self.logger.info(f"\nProcessing {model_type.upper()} model")
            model_dir = os.path.join(self.run_path, model_type)
            os.makedirs(model_dir, exist_ok=True)

            results_saver = ModelResultsSaver(self.logger)
            test_results = {}
            upcoming_results = {}

            # Process each position and stat type
            for position, stat_types in data_files.items():
                test_results[position] = {}
                upcoming_results[position] = {}
                
                for stat_type, data in stat_types.items():
                    # First, train and evaluate on test season
                    X_train, X_test, y_train, y_test, player_ids_test, target_col = self.prepare_data(
                        data, position, stat_type, is_upcoming=False
                    )
                    model = self.train_model(model_type, X_train, y_train, position, stat_type)
                    report, metrics = self.evaluate_and_save(
                        model, model_type, X_test, y_test, position, stat_type, 
                        player_ids_test, target_col, is_upcoming=False
                    )
                    test_results[position][stat_type] = {
                        'model': model,
                        'report': report,
                        'metrics': metrics
                    }

                    # Then, retrain on all data and predict upcoming season
                    X_train, X_test, y_train, y_test, player_ids_test, target_col = self.prepare_data(
                        data, position, stat_type, is_upcoming=True
                    )
                    model = self.train_model(model_type, X_train, y_train, position, stat_type)
                    report, metrics = self.evaluate_and_save(
                        model, model_type, X_test, y_test, position, stat_type, 
                        player_ids_test, target_col, is_upcoming=True
                    )
                    upcoming_results[position][stat_type] = {
                        'model': model,
                        'report': report,
                        'metrics': metrics
                    }

                    # Save model artifacts in test season directory
                    results_saver.save_model(model, model_type, model_dir)
                    results_saver.save_run_metadata(model, model_type, self.test_season, metrics, model_dir)

            all_results['test_season'][model_type] = test_results
            all_results['upcoming_season'][model_type] = upcoming_results

        # Organize predictions for both test and upcoming seasons
        self.organize_and_save_predictions(all_results['test_season'], is_upcoming=False)
        self.organize_and_save_predictions(all_results['upcoming_season'], is_upcoming=True)
        
        #DEBUG get targetted people
        qb_info = {
            '160d493c-12d7-4e86-a685-eb83bed4f3bd': ('Bryson Davis', 'Kennesaw State'),
            '514fe10d-d116-42e4-878e-281adaf702e3': ('Jalen Milroe', 'Alabama'),
            'f208bc81-d708-48b2-8325-cd8071b3c32d': ('Thomas Castellanos', 'Boston College'),
            'f589f248-558c-48b2-8825-e396b6a19fa3': ('Carson Beck', 'Georgia'),
            'f58f8dbe-91e9-42c4-a13f-108aae1fda13': ('Jaxson Dart', 'Ole Miss'),
        }

        # Create a list to store all predictions
        qb_predictions = []

        # Get predictions for each quarterback
        for player_id, (name, school) in qb_info.items():
            for position, stat_types in data_files.items():
                for stat_type, data in stat_types.items():
                    X_train, X_test, y_train, y_test, player_ids_test, target_col = self.prepare_data(
                        data, position, stat_type, is_upcoming=True
                    )
                    
                    # Get predictions from all models
                    predictions = {}
                    for model_type in self.models:
                        model = all_results['upcoming_season'][model_type][position][stat_type]['model']
                        if model_type == 'neural_network':
                            trainer = NeuralNetworkTrainer(self.logger, self.run_path)
                            pred = trainer.predict(model, X_test, position=position, stat_type=stat_type)
                        else:
                            pred = model.predict(X_test)
                        
                        player_ids = player_ids_test.to_list()
                        if player_id in player_ids:
                            player_index = player_ids.index(player_id)
                            predictions[f'{model_type}_prediction'] = pred[player_index]
                    
                    if predictions:
                        # Calculate ensemble prediction
                        ensemble_pred = sum(predictions.values()) / len(predictions)
                        
                        qb_predictions.append({
                            'player_id': player_id,
                            'name': name,
                            'school': school,
                            'stat_type': stat_type,
                            'ensemble_prediction': round(ensemble_pred, 1),
                            **{k: round(v, 1) for k, v in predictions.items()}
                        })

        # Convert to DataFrame and save
        predictions_df = pd.DataFrame(qb_predictions)
        #ordery by stat type
        predictions_df = predictions_df.sort_values(by=['stat_type'])
        
        output_file = os.path.join('data/output', 'targeted_qb_predictions.csv')
        predictions_df.to_csv(output_file, index=False)
        self.logger.info(f"Saved targeted QB predictions to {output_file}")