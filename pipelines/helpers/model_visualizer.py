import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


class ModelVisualizer:
    """Utility class for creating model evaluation visualizations"""
    plt.ioff()
    
    @staticmethod
    def create_scatter_plot(y_test, y_pred, r2_test, position, results_dir):
        """
        Create a scatter plot of predicted vs actual values with R² score
        
        Parameters:
        -----------
        y_test : array-like
            True target values
        y_pred : array-like
            Predicted target values
        r2_test : float
            R-squared score
        position : str
            Player position (QB, RB, WR)
        results_dir : str
            Directory to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot
        plt.scatter(y_test, y_pred, alpha=0.5)
        
        # Add perfect prediction line
        max_val = max(max(y_test), max(y_pred))
        min_val = min(min(y_test), min(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        # Add labels and title
        target_label = {
            'QB': 'Passing Yards',
            'RB': 'Rushing Yards',
            'WR': 'Receiving Yards'
        }.get(position, 'Yards')
        
        plt.xlabel(f'Actual {target_label}')
        plt.ylabel(f'Predicted {target_label}')
        plt.title(f'{position} Model Predictions\nR² = {r2_test:.3f}')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'prediction_scatter.png'))
        plt.close()

    @staticmethod
    def create_residual_plot(y_test, y_pred, position, model_type, results_dir):
        """
        Create residual plots to analyze prediction errors
        
        Parameters:
        -----------
        y_test : array-like
            True target values
        y_pred : array-like
            Predicted target values
        position : str
            Player position (QB, RB, WR)
        model_type : str
            Type of model used for predictions
        results_dir : str
            Directory to save the plot
        """
        residuals = y_test - y_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot of residuals vs predicted values
        ax1.scatter(y_pred, residuals, alpha=0.5)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted Values')
        ax1.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(y_pred, residuals, 1)
        p = np.poly1d(z)
        ax1.plot(y_pred, p(y_pred), "b--", alpha=0.8, label='Trend')
        ax1.legend()
        
        # Histogram of residuals
        ax2.hist(residuals, bins=30, edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--')
        ax2.set_xlabel('Residual Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Residuals')
        ax2.grid(True, alpha=0.3)
        
        target_label = {
            'QB': 'Passing Yards',
            'RB': 'Rushing Yards',
            'WR': 'Receiving Yards'
        }.get(position, 'Yards')
        plt.suptitle(f'{position} {model_type.title()} Model Residual Analysis\nTarget: {target_label}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'residuals.png'))
        plt.close()

    @staticmethod
    def create_feature_importance_plot(model, X_test, position, model_type, results_dir):
        """
        Create feature importance plot for supported model types
        
        Parameters:
        -----------
        model : object
            Trained model instance
        X_test : pd.DataFrame
            Test features
        position : str
            Player position
        model_type : str
            Type of model used
        results_dir : str
            Directory to save the plot
        """
        if model_type in ["random_forest", "lightgbm"]:
            feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(feature_importance)), feature_importance['importance'])
            plt.xticks(range(len(feature_importance)), feature_importance['feature'], 
                      rotation=45, ha='right')
            plt.title(f'{position} Model: Feature Importance ({model_type.title()})')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'feature_importance.png'))
            plt.close()
            
            # Save feature importance to text file
            with open(os.path.join(results_dir, 'feature_importance.txt'), 'w') as f:
                for index, row in feature_importance.iterrows():
                    f.write(f"{row['feature']}: {row['importance']}\n")