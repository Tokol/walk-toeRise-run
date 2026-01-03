"""
RANDOM FOREST MODULE FOR INTEGRATION WITH MAIN APP.PY
=====================================================
This module provides Random Forest functionality to be called from the main app.
It uses ALL 9 features (4 motion + 5 shape) for activity recognition.
FIXED: Cross-validation uses ONLY training data to prevent data leakage.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import plotly.graph_objects as go

import time


class RandomForestModel:
    """
    Random Forest classifier for activity recognition.
    Uses ALL 9 features (4 motion + 5 shape) for prediction.
    """

    def __init__(self, n_estimators=100, random_state=42):
        """
        Initialize Random Forest model.

        Parameters:
        -----------
        n_estimators : int
            Number of trees in the forest
        random_state : int
            Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )
        self.is_trained = False
        self.training_time = None
        self.prediction_time = None

        # ALL 9 features for Random Forest
        self.feature_names = [
            # 4 Motion features
            "mean_magnitude",
            "rms_magnitude",
            "std_magnitude",
            "p2p_magnitude",
            # 5 Shape features
            "kurtosis_magnitude",
            "skewness_magnitude",
            "median_magnitude",
            "iqr_magnitude",
            "peak_count"
        ]

    def select_features(self, features_df):
        """
        Select ALL 9 features for Random Forest.

        Parameters:
        -----------
        features_df : pandas DataFrame
            Complete features dataframe

        Returns:
        --------
        X_rf : pandas DataFrame
            Features with all 9 columns
        y : pandas Series
            Target labels
        """
        # Verify required features exist
        missing_features = [f for f in self.feature_names if f not in features_df.columns]
        if missing_features:
            raise ValueError(f"Missing required features for Random Forest: {missing_features}")

        # Select all 9 features
        X_rf = features_df[self.feature_names].copy()
        y = features_df["activity"].copy()

        return X_rf, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data for Random Forest.
        Note: Random Forest doesn't need feature scaling!

        Parameters:
        -----------
        X : pandas DataFrame
            Features (9 features)
        y : pandas Series
            Labels
        test_size : float
            Proportion for testing
        random_state : int
            Random seed

        Returns:
        --------
        Dictionary containing split data
        """
        # Split data (stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.feature_names
        }

    def train_model(self, X_train, y_train):
        """
        Train the Random Forest model.

        Parameters:
        -----------
        X_train : pandas DataFrame
            Training features
        y_train : pandas Series
            Training labels

        Returns:
        --------
        training_time : float
            Time taken to train
        """
        start_time = time.time()

        self.model.fit(X_train, y_train)
        self.is_trained = True

        self.training_time = time.time() - start_time
        return self.training_time

    def predict(self, X):
        """
        Make predictions using trained Random Forest.

        Parameters:
        -----------
        X : pandas DataFrame
            Features to predict

        Returns:
        --------
        predictions : numpy array
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction!")

        start_time = time.time()
        predictions = self.model.predict(X)
        self.prediction_time = time.time() - start_time

        return predictions

    def evaluate(self, X_test, y_test, y_pred=None):
        """
        Evaluate model performance.

        Parameters:
        -----------
        X_test : pandas DataFrame
            Test features
        y_test : pandas Series
            True test labels
        y_pred : numpy array, optional
            Predictions

        Returns:
        --------
        Dictionary containing all evaluation metrics
        """
        if y_pred is None:
            y_pred = self.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)

        # Per-class accuracy
        classes = np.unique(y_test)
        per_class_accuracy = []
        for i, cls in enumerate(classes):
            class_mask = (y_test == cls)
            if np.sum(class_mask) > 0:
                class_acc = accuracy_score(y_test[class_mask], y_pred[class_mask])
                per_class_accuracy.append(class_acc)

        # Feature importance
        feature_importance = self.model.feature_importances_

        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm,
            'classification_report': report,
            'per_class_accuracy': dict(zip(classes, per_class_accuracy)),
            'feature_importance': dict(zip(self.feature_names, feature_importance)),
            'predictions': y_pred,
            'true_labels': y_test,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time
        }

    def cross_validate(self, X_train, y_train, cv=5):
        """
        Perform cross-validation.
        IMPORTANT: Uses ONLY training data to prevent data leakage.

        Parameters:
        -----------
        X_train : pandas DataFrame
            Training features ONLY
        y_train : pandas Series
            Training labels ONLY
        cv : int
            Number of folds

        Returns:
        --------
        Dictionary with cross-validation results
        """
        # Create a new model for CV to avoid data leakage
        cv_model = RandomForestClassifier(
            n_estimators=self.model.n_estimators,
            random_state=self.model.random_state
        )

        # Perform cross-validation on training data
        cv_scores = cross_val_score(
            cv_model, X_train, y_train,
            cv=cv, scoring='accuracy'
        )

        return {
            'cv_scores': cv_scores,
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'min_score': cv_scores.min(),
            'max_score': cv_scores.max()
        }


def create_rf_performance_plots(metrics, data_info):
    """
    Create Plotly visualizations for Random Forest performance.

    Parameters:
    -----------
    metrics : dict
        RF evaluation metrics
    data_info : dict
        Dataset information

    Returns:
    --------
    Dictionary of Plotly figures
    """
    figures = {}

    # 1. Metrics Bar Chart
    metrics_names = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    metrics_values = [
        metrics['accuracy'],
        metrics['f1_score'],
        metrics['precision'],
        metrics['recall']
    ]

    fig_metrics = go.Figure(data=[
        go.Bar(
            x=metrics_names,
            y=metrics_values,
            marker_color=['blue', 'green', 'orange', 'red'],
            text=[f'{val:.3f}' for val in metrics_values],
            textposition='auto'
        )
    ])

    fig_metrics.update_layout(
        title='Random Forest Performance Metrics',
        yaxis=dict(range=[0, 1]),
        height=400
    )

    figures['metrics_bar'] = fig_metrics

    # 2. Confusion Matrix
    cm = metrics['confusion_matrix']
    classes = np.unique(metrics['true_labels'])
    class_names = [cls.replace('_', ' ').title() for cls in classes]

    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Greens',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 12},
        hoverongaps=False
    ))

    fig_cm.update_layout(
        title='Random Forest Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        height=500
    )

    figures['confusion_matrix'] = fig_cm

    # 3. Feature Importance
    feature_importance = metrics['feature_importance']
    features = list(feature_importance.keys())
    importance_values = list(feature_importance.values())

    # Sort by importance
    sorted_idx = np.argsort(importance_values)[::-1]
    features_sorted = [features[i].replace('_', ' ').title() for i in sorted_idx]
    values_sorted = [importance_values[i] for i in sorted_idx]

    fig_fi = go.Figure(data=[
        go.Bar(
            x=features_sorted,
            y=values_sorted,
            marker_color='lightblue',
            text=[f'{val:.3f}' for val in values_sorted],
            textposition='auto'
        )
    ])

    fig_fi.update_layout(
        title='Random Forest Feature Importance',
        xaxis_title='Features',
        yaxis_title='Importance Score',
        height=500
    )

    figures['feature_importance'] = fig_fi

    # 4. Timing
    times = ['Training', 'Prediction (20 samples)']
    time_values = [metrics['training_time'], metrics['prediction_time']]

    fig_times = go.Figure(data=[
        go.Bar(
            x=times,
            y=time_values,
            marker_color=['blue', 'green'],
            text=[f'{t:.3f}s' for t in time_values],
            textposition='auto'
        )
    ])

    fig_times.update_layout(
        title='Random Forest Computational Performance',
        yaxis_title='Time (seconds)',
        height=350
    )

    figures['timing'] = fig_times

    return figures


def create_rf_cv_plot(cv_results):
    """
    Create plot for cross-validation results.

    Parameters:
    -----------
    cv_results : dict
        Cross-validation results

    Returns:
    --------
    Plotly figure
    """
    fig = go.Figure(data=[
        go.Bar(
            x=[f'Fold {i + 1}' for i in range(len(cv_results['cv_scores']))],
            y=cv_results['cv_scores'],
            marker_color='lightgreen',
            text=[f'{score:.3f}' for score in cv_results['cv_scores']],
            textposition='auto'
        )
    ])

    fig.add_hline(
        y=cv_results['mean_score'],
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {cv_results['mean_score']:.3f}",
        annotation_position="bottom right"
    )

    fig.update_layout(
        title=f'5-Fold Cross-Validation on Training Data (Mean: {cv_results["mean_score"]:.3f} ± {cv_results["std_score"]:.3f})',
        xaxis_title='Fold',
        yaxis_title='Accuracy',
        yaxis=dict(range=[0, 1]),
        height=400
    )

    return fig


def run_rf_pipeline(features_df, test_size=0.2, n_estimators=100, random_state=42):
    """
    Complete Random Forest pipeline.
    FIXED: Cross-validation uses ONLY training data to prevent data leakage.

    Parameters:
    -----------
    features_df : pandas DataFrame
        Features dataframe
    test_size : float
        Proportion for testing
    n_estimators : int
        Number of trees
    random_state : int
        Random seed

    Returns:
    --------
    Dictionary containing all RF results
    """
    print("🌲 Starting Random Forest pipeline...")

    # Initialize RF model
    rf = RandomForestModel(n_estimators=n_estimators, random_state=random_state)

    # Step 1: Select all 9 features
    print("📊 Step 1: Selecting 9 features for Random Forest...")
    X_rf, y = rf.select_features(features_df)

    # Step 2: Split data (no scaling needed for RF)
    print("🔪 Step 2: Splitting data...")
    data_splits = rf.split_data(X_rf, y, test_size=test_size, random_state=random_state)

    # Step 3: Train model
    print("🏋️ Step 3: Training Random Forest...")
    training_time = rf.train_model(data_splits['X_train'], data_splits['y_train'])

    # Step 4: Evaluate
    print("📈 Step 4: Evaluating on test set...")
    metrics = rf.evaluate(data_splits['X_test'], data_splits['y_test'])

    # Step 5: Cross-validation on TRAINING DATA ONLY (prevents data leakage)
    print("🔄 Step 5: Performing cross-validation on TRAINING data only...")
    cv_results = rf.cross_validate(
        data_splits['X_train'],  # Use training features only
        data_splits['y_train'],  # Use training labels only
        cv=5
    )

    # Create visualizations
    print("🎨 Step 6: Creating visualizations...")
    data_info = {
        'samples': len(features_df),
        'train_samples': len(data_splits['X_train']),
        'test_samples': len(data_splits['X_test']),
        'features_used': len(rf.feature_names),
        'classes': len(np.unique(y))
    }

    figures = create_rf_performance_plots(metrics, data_info)
    figures['cv_plot'] = create_rf_cv_plot(cv_results)

    # Compile results
    results = {
        'model': rf,
        'data_splits': data_splits,
        'metrics': metrics,
        'cv_results': cv_results,
        'figures': figures,
        'data_info': data_info,
        'parameters': {
            'n_estimators': n_estimators,
            'test_size': test_size,
            'random_state': random_state,
            'features_used': rf.feature_names
        }
    }

    print("✅ Random Forest pipeline completed!")

    return results


def test_rf_module():
    """Quick test to verify the module works."""
    print("🧪 Testing Random Forest module...")

    # Create dummy data
    np.random.seed(42)
    n_samples = 100

    # Create dummy features (all 9 features)
    dummy_data = {
        'mean_magnitude': np.random.randn(n_samples) + 10,
        'rms_magnitude': np.random.randn(n_samples) + 12,
        'std_magnitude': np.random.randn(n_samples) + 2,
        'p2p_magnitude': np.random.randn(n_samples) + 5,
        'kurtosis_magnitude': np.random.randn(n_samples),
        'skewness_magnitude': np.random.randn(n_samples),
        'median_magnitude': np.random.randn(n_samples) + 9,
        'iqr_magnitude': np.random.randn(n_samples) + 3,
        'peak_count': np.random.randint(1, 10, n_samples),
        'activity': np.random.choice(['walking', 'toe_rise', 'running'], n_samples)
    }

    features_df = pd.DataFrame(dummy_data)

    # Run RF pipeline
    results = run_rf_pipeline(features_df)

    print(f"📊 Accuracy: {results['metrics']['accuracy']:.3f}")
    print(f"⏱️ Training time: {results['metrics']['training_time']:.3f}s")
    print(f"📈 CV Mean Score: {results['cv_results']['mean_score']:.3f}")

    return results


if __name__ == "__main__":
    # Run test if module is executed directly
    test_results = test_rf_module()
    print("✅ Random Forest module test passed!")