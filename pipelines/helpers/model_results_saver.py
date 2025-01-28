import datetime
import json
import os
import pickle
import torch
import pandas as pd
import numpy as np
from logging import Logger
from typing import Dict, Any, Tuple, List

class ModelResultsSaver:
    """
    A class responsible for saving all model-related results, including:
    - Model artifacts
    - Evaluation metrics
    - Predictions
    - Metadata
    - Error analysis
    """
    def __init__(self, logger: Logger):
        self.logger = logger

    def save_model(self, model: Any, model_type: str, position_dir: str) -> None:
        """Save the trained model based on its type"""
        self.logger.info(f"Saving {model_type} model in {position_dir}")
        
        if model_type == "random_forest":
            model_path = os.path.join(position_dir, 'model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        elif model_type == "lightgbm":
            model_path = os.path.join(position_dir, 'model.txt')
            model.booster_.save_model(model_path)
        elif model_type == "linear":
            model_path = os.path.join(position_dir, 'model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        elif model_type == "neural_network":
            model_path = os.path.join(position_dir, 'model_state.pth')
            torch.save(model.state_dict(), model_path)

    def save_metrics(self, metrics: Dict[str, float], results_dir: str) -> None:
        """Save model performance metrics"""
        metrics_path = os.path.join(results_dir, 'metrics.txt')
        with open(metrics_path, 'w') as f:
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")

    def save_detailed_results(self, y_test: pd.Series, y_pred: np.ndarray, 
                            player_ids: pd.Series, position: str, 
                            results_dir: str) -> pd.DataFrame:
        """Save detailed prediction results and error analysis"""
        # Calculate residuals and create results DataFrame
        residuals = y_test - y_pred
        results_df = pd.DataFrame({
            'master_player_id': player_ids,
            'actual': y_test,
            'predicted': y_pred,
            'residual': residuals,
            'abs_residual': np.abs(residuals),
            'percent_error': np.abs(residuals / y_test) * 100
        })
        
        # Add error percentile rank
        results_df['error_percentile'] = results_df['abs_residual'].rank(pct=True) * 100
        
        # Save all predictions
        results_df.to_csv(os.path.join(results_dir, 'predictions_detailed.csv'), index=False)
        
        # Save largest errors
        top_residuals = results_df.nlargest(10, 'abs_residual')
        top_residuals.to_csv(os.path.join(results_dir, 'largest_errors.csv'), index=False)
        
        # Calculate and save error distribution statistics
        self._save_error_statistics(results_df, results_dir)
        
        return results_df

    def _save_error_statistics(self, results_df: pd.DataFrame, results_dir: str) -> None:
        """Save error distribution statistics"""
        error_stats = {
            'mean_abs_error': results_df['abs_residual'].mean(),
            'median_abs_error': results_df['abs_residual'].median(),
            'error_std': results_df['residual'].std(),
            'mean_percent_error': results_df['percent_error'].mean(),
            'median_percent_error': results_df['percent_error'].median(),
            '90th_percentile_error': results_df['abs_residual'].quantile(0.9),
            '95th_percentile_error': results_df['abs_residual'].quantile(0.95)
        }
        
        with open(os.path.join(results_dir, 'error_statistics.txt'), 'w') as f:
            f.write("Error Distribution Statistics:\n")
            for stat, value in error_stats.items():
                f.write(f"{stat}: {value:.2f}\n")

    def save_run_metadata(self, model: Any, model_type: str, test_season: int,
                         metrics: Dict[str, float], position_dir: str) -> None:
        """Save run metadata including model parameters and performance metrics"""
        metadata = {
            'run_timestamp': datetime.datetime.now().isoformat(),
            'test_season': test_season,
            'model_type': model_type,
            'metrics': metrics
        }
        
        # Add model parameters for applicable model types
        if model_type in ["random_forest", "lightgbm", "linear"]:
            metadata['model_parameters'] = model.get_params()
        
        metadata_path = os.path.join(position_dir, 'run_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

    def save_specific_player_predictions(self, player_ids: List[str], 
                                       predictions_df: pd.DataFrame,
                                       output_dir: str) -> None:
        """Save predictions for specific players"""
        filtered_predictions = predictions_df[
            predictions_df['master_player_id'].isin(player_ids)
        ]
        predictions_path = os.path.join(output_dir, 'player_predictions.csv')
        filtered_predictions.to_csv(predictions_path, index=False)