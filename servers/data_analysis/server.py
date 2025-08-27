"""
Simplified MCP Tool for Data Analysis, following FastMCP server format.

This tool performs:
1. Data Preprocessing: Drop rows with missing values.
2. Feature Engineering: Pearson correlation analysis.
3. Model Training: XGBoost regression with hyperparameter optimization.
4. Model Interpretation: SHAP analysis.

Usage:
- User uploads a CSV file.
- The tool processes the data and returns results.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tempfile
import os
import base64
from typing import Dict, Any, Optional, List
import argparse
from mcp.server.fastmcp import FastMCP


# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def parse_args():
    """Parse command line arguments for MCP server."""
    parser = argparse.ArgumentParser(description="Data Analysis MCP Server")
    parser.add_argument('--port', type=int, default=50001, help='Server port (default: 50001)')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    try:
        args = parser.parse_args()
    except SystemExit:
        class Args:
            port = 50001
            host = '0.0.0.0'
            log_level = 'INFO'
        args = Args()
    return args


args = parse_args()
mcp = FastMCP("data_analysis", port=args.port, host=args.host)


class SimplifiedDataAnalysisTool:
    """A simplified tool for data analysis with XGBoost and SHAP."""

    def __init__(self):
        self.data = None
        self.processed_data = None
        self.model = None
        self.shap_explainer = None
        self.shap_values = None
        self.target_column = None

    def load_csv_data(self, csv_content: str, encoding: str = 'utf-8') -> Dict[str, Any]:
        """
        Load CSV data from content (can be base64 encoded or raw string).

        Args:
            csv_content (str): CSV data content.
            encoding (str): Encoding of the CSV data.

        Returns:
            Dict[str, Any]: Result of the operation.
        """
        try:
            # Try to decode base64
            try:
                csv_data = base64.b64decode(csv_content).decode(encoding)
            except:
                csv_data = csv_content

            # Create a temporary file and read
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
                temp_file.write(csv_data)
                temp_file_path = temp_file.name

            # Read CSV
            self.data = pd.read_csv(temp_file_path)
            os.unlink(temp_file_path)

            return {
                'success': True,
                'message': f'Successfully loaded CSV data with shape {self.data.shape}',
                'data_shape': self.data.shape
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'Error loading CSV data: {str(e)}'
            }

    def preprocess_data(self) -> Dict[str, Any]:
        """
        Preprocess data by dropping rows with missing values.

        Returns:
            Dict[str, Any]: Result of the operation.
        """
        if self.data is None:
            return {'success': False, 'message': 'No data loaded. Please load CSV data first.'}

        try:
            initial_shape = self.data.shape
            self.processed_data = self.data.dropna()
            final_shape = self.processed_data.shape

            return {
                'success': True,
                'message': 'Data preprocessing completed successfully',
                'initial_shape': initial_shape,
                'final_shape': final_shape,
                'rows_dropped': initial_shape[0] - final_shape[0]
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'Error in data preprocessing: {str(e)}'
            }

    def feature_engineering(self, target_column: str) -> Dict[str, Any]:
        """
        Perform Pearson correlation analysis.

        Args:
            target_column (str): Name of the target column.

        Returns:
            Dict[str, Any]: Result of the operation including correlation matrix.
        """
        if self.processed_data is None:
            return {'success': False, 'message': 'No processed data available. Please preprocess data first.'}

        if target_column not in self.processed_data.columns:
            return {'success': False, 'message': f'Target column "{target_column}" not found in data.'}

        try:
            self.target_column = target_column
            numeric_df = self.processed_data.select_dtypes(include=[np.number])
            correlation_matrix = numeric_df.corr()

            # Convert correlation matrix to a serializable format
            corr_dict = correlation_matrix.to_dict()

            return {
                'success': True,
                'message': 'Feature engineering (correlation analysis) completed successfully',
                'correlation_matrix': corr_dict,
                'target_correlations': {k: float(v) for k, v in correlation_matrix[target_column].abs().sort_values(ascending=False).to_dict().items()}
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'Error in feature engineering: {str(e)}'
            }

    def train_model(self, target_column: str, n_iter: int = 10) -> Dict[str, Any]:
        """
        Train an XGBoost regression model with hyperparameter optimization.

        Args:
            target_column (str): Name of the target column.
            n_iter (int): Number of iterations for RandomizedSearchCV.

        Returns:
            Dict[str, Any]: Result of the operation including model performance.
        """
        if self.processed_data is None:
            return {'success': False, 'message': 'No processed data available. Please preprocess data first.'}

        if target_column not in self.processed_data.columns:
            return {'success': False, 'message': f'Target column "{target_column}" not found in data.'}

        try:
            X = self.processed_data.drop(columns=[target_column])
            y = self.processed_data[target_column]

            # Ensure X contains only numeric columns
            X = X.select_dtypes(include=[np.number])

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_SEED
            )

            # Define XGBoost model
            xgb_model = xgb.XGBRegressor(random_state=RANDOM_SEED)

            # Define hyperparameter search space
            param_distributions = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }

            # Perform hyperparameter optimization
            search = RandomizedSearchCV(
                xgb_model,
                param_distributions,
                n_iter=n_iter,
                cv=5,
                scoring='r2',
                n_jobs=-1,
                random_state=RANDOM_SEED
            )

            search.fit(X_train, y_train)
            self.model = search.best_estimator_

            # Evaluate model
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            return {
                'success': True,
                'message': 'Model training completed successfully',
                'best_params': search.best_params_,
                'mse': float(mse),
                'r2_score': float(r2),
                'train_shape': X_train.shape,
                'test_shape': X_test.shape
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'Error in model training: {str(e)}'
            }

    def interpret_model(self, sample_size: int = 100) -> Dict[str, Any]:
        """
        Perform SHAP analysis on the trained model.

        Args:
            sample_size (int): Number of samples to use for SHAP analysis.

        Returns:
            Dict[str, Any]: Result of the operation including SHAP values.
        """
        if self.model is None:
            return {'success': False, 'message': 'No trained model available. Please train a model first.'}

        if self.processed_data is None or self.target_column is None:
            return {'success': False, 'message': 'Processed data or target column not available.'}

        try:
            X = self.processed_data.drop(columns=[self.target_column])
            X = X.select_dtypes(include=[np.number])

            # Sample data for SHAP analysis if dataset is large
            if len(X) > sample_size:
                X_sample = X.sample(n=sample_size, random_state=RANDOM_SEED)
            else:
                X_sample = X

            # Create SHAP explainer
            self.shap_explainer = shap.TreeExplainer(self.model)
            self.shap_values = self.shap_explainer.shap_values(X_sample)

            # Calculate mean absolute SHAP values for feature importance
            mean_shap_values = np.abs(self.shap_values).mean(0)
            feature_importance = dict(zip(X_sample.columns, mean_shap_values))

            return {
                'success': True,
                'message': 'Model interpretation (SHAP analysis) completed successfully',
                'feature_importance': {k: float(v) for k, v in feature_importance.items()},
                'sample_size': len(X_sample)
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'Error in model interpretation: {str(e)}'
            }

    def run_full_analysis(self, csv_content: str, target_column: str, encoding: str = 'utf-8') -> Dict[str, Any]:
        """
        Run the full analysis pipeline.

        Args:
            csv_content (str): CSV data content.
            target_column (str): Name of the target column.
            encoding (str): Encoding of the CSV data.

        Returns:
            Dict[str, Any]: Results of all steps in the analysis.
        """
        results = {}

        # Load data
        load_result = self.load_csv_data(csv_content, encoding)
        results['load_data'] = load_result
        if not load_result['success']:
            return results

        # Preprocess data
        preprocess_result = self.preprocess_data()
        results['preprocess_data'] = preprocess_result
        if not preprocess_result['success']:
            return results

        # Feature engineering
        feature_result = self.feature_engineering(target_column)
        results['feature_engineering'] = feature_result
        if not feature_result['success']:
            return results

        # Train model
        train_result = self.train_model(target_column)
        results['train_model'] = train_result
        if not train_result['success']:
            return results

        # Interpret model
        interpret_result = self.interpret_model()
        results['interpret_model'] = interpret_result

        return results


@mcp.tool()
def analyze_csv_data(csv_content: str, target_column: str) -> Dict[str, Any]:
    """
    Analyze CSV data using simplified data analysis pipeline.

    Args:
        csv_content (str): CSV data content (can be base64 encoded).
        target_column (str): Name of the target column for prediction.

    Returns:
        Dict[str, Any]: Results of the analysis.
    """
    tool = SimplifiedDataAnalysisTool()
    return tool.run_full_analysis(csv_content, target_column)


if __name__ == "__main__":
    # Get transport type from environment variable, default to SSE
    transport_type = os.getenv('MCP_TRANSPORT', 'sse')
    mcp.run(transport=transport_type)
