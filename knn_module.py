"""
KNN MODULE FOR INTEGRATION WITH MAIN APP.PY
===========================================
This module provides KNN functionality to be called from the main app.
It uses ONLY 4 motion features for activity recognition.
FIXED: Cross-validation uses ONLY training data to prevent data leakage.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import plotly.graph_objects as go

import time


class KNNModel:
    """
    K-Nearest Neighbors classifier for activity recognition.
    Uses only 4 motion features for prediction.
    """

    def __init__(self, n_neighbors=5):
        """
        Initialize KNN model.

        Parameters:
        -----------
        n_neighbors : int
            Number of neighbors to use (k-value)
        """
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.scaler = StandardScaler()  # KNN REQUIRES feature scaling!
        self.is_trained = False
        self.training_time = None
        self.prediction_time = None

        # ONLY these 4 motion features for KNN
        self.feature_names = [
            "mean_magnitude",
            "rms_magnitude",
            "std_magnitude",
            "p2p_magnitude"
        ]

    def select_features(self, features_df):
        """
        Select ONLY the 4 motion features for KNN.

        Parameters:
        -----------
        features_df : pandas DataFrame
            Complete features dataframe with all 9 features

        Returns:
        --------
        X_knn : pandas DataFrame
            Features with only 4 motion columns
        y : pandas Series
            Target labels
        """
        # Verify required features exist
        missing_features = [f for f in self.feature_names if f not in features_df.columns]
        if missing_features:
            raise ValueError(f"Missing required features for KNN: {missing_features}")

        # Select only 4 motion features
        X_knn = features_df[self.feature_names].copy()
        y = features_df["activity"].copy()

        return X_knn, y

    def split_and_scale(self, X, y, test_size=0.2, random_state=42):
        """
        Split data and scale features for KNN.

        Parameters:
        -----------
        X : pandas DataFrame
            Features (4 motion features)
        y : pandas Series
            Labels
        test_size : float
            Proportion for testing (default: 0.2 = 20%)
        random_state : int
            Random seed for reproducibility

        Returns:
        --------
        Dictionary containing all split and scaled data
        """
        # Split data (stratified to maintain class distribution)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y  # Important for imbalanced classes
        )

        # Scale features (CRITICAL for KNN - distance-based!)
        # Fit scaler on TRAINING data only (prevent data leakage)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)  # Use same transformation

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'feature_names': self.feature_names
        }

    def train_model(self, X_train_scaled, y_train):
        """
        Train the KNN model.

        Parameters:
        -----------
        X_train_scaled : numpy array
            Scaled training features
        y_train : numpy array
            Training labels

        Returns:
        --------
        training_time : float
            Time taken to train the model (seconds)
        """
        start_time = time.time()

        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        self.training_time = time.time() - start_time
        return self.training_time

    def predict(self, X_scaled):
        """
        Make predictions using trained KNN model.

        Parameters:
        -----------
        X_scaled : numpy array
            Scaled features

        Returns:
        --------
        predictions : numpy array
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction!")

        start_time = time.time()
        predictions = self.model.predict(X_scaled)
        self.prediction_time = time.time() - start_time

        return predictions

    def evaluate(self, X_test_scaled, y_test, y_pred=None):
        """
        Evaluate model performance.

        Parameters:
        -----------
        X_test_scaled : numpy array
            Scaled test features
        y_test : numpy array
            True test labels
        y_pred : numpy array, optional
            Predictions (will compute if not provided)

        Returns:
        --------
        Dictionary containing all evaluation metrics
        """
        if y_pred is None:
            y_pred = self.predict(X_test_scaled)

        # Calculate multiple metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')  # Weighted for imbalanced data
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Classification report (detailed per-class metrics)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Per-class accuracy
        classes = np.unique(y_test)
        per_class_accuracy = []
        for i, cls in enumerate(classes):
            class_mask = (y_test == cls)
            if np.sum(class_mask) > 0:
                class_acc = accuracy_score(y_test[class_mask], y_pred[class_mask])
                per_class_accuracy.append(class_acc)

        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm,
            'classification_report': report,
            'per_class_accuracy': dict(zip(classes, per_class_accuracy)),
            'predictions': y_pred,
            'true_labels': y_test,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time
        }

    def cross_validate(self, X_train, y_train, cv=5):
        """
        Perform cross-validation for more reliable performance estimate.
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
        # Create a new scaler for CV (don't use the one fitted on full training data)
        cv_scaler = StandardScaler()

        # Scale the training data for CV
        X_train_scaled = cv_scaler.fit_transform(X_train)

        # Perform cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train,
            cv=cv, scoring='accuracy'
        )

        return {
            'cv_scores': cv_scores,
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'min_score': cv_scores.min(),
            'max_score': cv_scores.max()
        }

    def find_optimal_k(self, X_train_scaled, y_train, X_test_scaled, y_test, k_range=range(1, 16, 2)):
        """
        Find optimal number of neighbors (k).
        Uses training data for fitting, test data for evaluation.

        Parameters:
        -----------
        X_train_scaled, y_train : training data
        X_test_scaled, y_test : testing data
        k_range : range of k values to test

        Returns:
        --------
        Dictionary with optimal k analysis
        """
        train_scores = []
        test_scores = []

        for k in k_range:
            # Create and train temporary model
            temp_knn = KNeighborsClassifier(n_neighbors=k)
            temp_knn.fit(X_train_scaled, y_train)

            # Evaluate
            train_pred = temp_knn.predict(X_train_scaled)
            test_pred = temp_knn.predict(X_test_scaled)

            train_scores.append(accuracy_score(y_train, train_pred))
            test_scores.append(accuracy_score(y_test, test_pred))

        # Find optimal k (highest test accuracy)
        optimal_k = k_range[np.argmax(test_scores)]
        optimal_test_score = test_scores[np.argmax(test_scores)]

        return {
            'k_values': list(k_range),
            'train_scores': train_scores,
            'test_scores': test_scores,
            'optimal_k': optimal_k,
            'optimal_test_score': optimal_test_score,
            'current_k': self.model.n_neighbors,
            'current_test_score': test_scores[list(k_range).index(self.model.n_neighbors)]
        }


# ==========================
# VISUALIZATION FUNCTIONS (for app.py to use)
# ==========================
def create_knn_performance_plots(metrics, data_info):
    """
    Create Plotly visualizations for KNN performance.

    Parameters:
    -----------
    metrics : dict
        KNN evaluation metrics
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
        title='KNN Performance Metrics',
        yaxis=dict(range=[0, 1]),
        height=400
    )

    figures['metrics_bar'] = fig_metrics

    # 2. Confusion Matrix Heatmap
    cm = metrics['confusion_matrix']
    classes = np.unique(metrics['true_labels'])
    class_names = [cls.replace('_', ' ').title() for cls in classes]

    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 12},
        hoverongaps=False
    ))

    fig_cm.update_layout(
        title='KNN Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        height=500
    )

    figures['confusion_matrix'] = fig_cm

    # 3. Per-Class Accuracy
    per_class_acc = metrics['per_class_accuracy']

    fig_class_acc = go.Figure(data=[
        go.Bar(
            x=[cls.replace('_', ' ').title() for cls in per_class_acc.keys()],
            y=list(per_class_acc.values()),
            marker_color=['blue', 'orange', 'red'],
            text=[f'{acc:.1%}' for acc in per_class_acc.values()],
            textposition='auto'
        )
    ])

    fig_class_acc.update_layout(
        title='KNN Accuracy per Activity Class',
        yaxis=dict(tickformat='.0%', range=[0, 1]),
        height=400
    )

    figures['per_class_accuracy'] = fig_class_acc

    # 4. Training/Prediction Time Comparison
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
        title='KNN Computational Performance',
        yaxis_title='Time (seconds)',
        height=350
    )

    figures['timing'] = fig_times

    return figures


def create_knn_optimal_k_plot(optimal_k_results):
    """
    Create plot for finding optimal k value.

    Parameters:
    -----------
    optimal_k_results : dict
        Results from find_optimal_k method

    Returns:
    --------
    Plotly figure
    """
    fig = go.Figure()

    # Training accuracy line
    fig.add_trace(go.Scatter(
        x=optimal_k_results['k_values'],
        y=optimal_k_results['train_scores'],
        mode='lines+markers',
        name='Training Accuracy',
        line=dict(color='blue', width=2)
    ))

    # Testing accuracy line
    fig.add_trace(go.Scatter(
        x=optimal_k_results['k_values'],
        y=optimal_k_results['test_scores'],
        mode='lines+markers',
        name='Testing Accuracy',
        line=dict(color='red', width=2)
    ))

    # Highlight optimal k
    fig.add_vline(
        x=optimal_k_results['optimal_k'],
        line_dash="dash",
        line_color="green",
        annotation_text=f"Optimal k={optimal_k_results['optimal_k']}",
        annotation_position="top right"
    )

    fig.update_layout(
        title='Finding Optimal k: Training vs Testing Accuracy',
        xaxis_title='Number of Neighbors (k)',
        yaxis_title='Accuracy',
        yaxis=dict(tickformat='.0%'),
        height=500
    )

    return fig


def create_knn_cv_plot(cv_results):
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


# ==========================
# MAIN KNN PIPELINE FUNCTION (for app.py to call)
# ==========================
def run_knn_pipeline(features_df, test_size=0.2, n_neighbors=5, random_state=42):
    """
    Complete KNN pipeline - to be called from app.py.
    FIXED: Cross-validation uses ONLY training data to prevent data leakage.

    Parameters:
    -----------
    features_df : pandas DataFrame
        Features dataframe from main app (with all 9 features)
    test_size : float
        Proportion for testing (default: 0.2)
    n_neighbors : int
        Number of neighbors for KNN
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    Dictionary containing all KNN results, metrics, and visualizations
    """
    print("🚀 Starting KNN pipeline...")

    # Initialize KNN model
    knn = KNNModel(n_neighbors=n_neighbors)

    # Step 1: Select only 4 motion features for KNN
    print("📊 Step 1: Selecting 4 motion features for KNN...")
    X_knn, y = knn.select_features(features_df)

    # Step 2: Split and scale data
    print("🔪 Step 2: Splitting and scaling data...")
    data_splits = knn.split_and_scale(X_knn, y, test_size=test_size, random_state=random_state)

    # Step 3: Train model
    print("🏋️ Step 3: Training KNN model...")
    training_time = knn.train_model(data_splits['X_train_scaled'], data_splits['y_train'])

    # Step 4: Evaluate on test set
    print("📈 Step 4: Evaluating on test set...")
    metrics = knn.evaluate(data_splits['X_test_scaled'], data_splits['y_test'])

    # Step 5: Cross-validation on TRAINING DATA ONLY (prevents data leakage)
    print("🔄 Step 5: Performing cross-validation on TRAINING data only...")
    cv_results = knn.cross_validate(
        data_splits['X_train'],  # Use training features only
        data_splits['y_train'],  # Use training labels only
        cv=5
    )

    # Step 6: Find optimal k
    print("🔧 Step 6: Finding optimal k value...")
    optimal_k_results = knn.find_optimal_k(
        data_splits['X_train_scaled'], data_splits['y_train'],
        data_splits['X_test_scaled'], data_splits['y_test']
    )

    # Create visualizations
    print("🎨 Step 7: Creating visualizations...")
    data_info = {
        'samples': len(features_df),
        'train_samples': len(data_splits['X_train']),
        'test_samples': len(data_splits['X_test']),
        'features_used': len(knn.feature_names),
        'classes': len(np.unique(y))
    }

    figures = create_knn_performance_plots(metrics, data_info)
    figures['optimal_k_plot'] = create_knn_optimal_k_plot(optimal_k_results)
    figures['cv_plot'] = create_knn_cv_plot(cv_results)

    # Compile all results
    results = {
        'model': knn,
        'data_splits': data_splits,
        'metrics': metrics,
        'cv_results': cv_results,
        'optimal_k_results': optimal_k_results,
        'figures': figures,
        'data_info': data_info,
        'parameters': {
            'n_neighbors': n_neighbors,
            'test_size': test_size,
            'random_state': random_state,
            'features_used': knn.feature_names
        }
    }

    print("✅ KNN pipeline completed successfully!")

    return results


# ==========================
# QUICK TEST FUNCTION (for debugging)
# ==========================
def test_knn_module():
    """Quick test to verify the module works."""
    print("🧪 Testing KNN module...")

    # Create dummy data
    np.random.seed(42)
    n_samples = 100

    # Create dummy features (4 motion features + 5 shape features)
    dummy_data = {
        'mean_magnitude': np.random.randn(n_samples) + 10,
        'rms_magnitude': np.random.randn(n_samples) + 12,
        'std_magnitude': np.random.randn(n_samples) + 2,
        'p2p_magnitude': np.random.randn(n_samples) + 5,
        'kurtosis_magnitude': np.random.randn(n_samples),  # Shape feature
        'skewness_magnitude': np.random.randn(n_samples),  # Shape feature
        'activity': np.random.choice(['walking', 'toe_rise', 'running'], n_samples)
    }

    features_df = pd.DataFrame(dummy_data)

    # Run KNN pipeline
    results = run_knn_pipeline(features_df)

    print(f"📊 Accuracy: {results['metrics']['accuracy']:.3f}")
    print(f"⏱️ Training time: {results['metrics']['training_time']:.3f}s")
    print(f"🎯 Optimal k: {results['optimal_k_results']['optimal_k']}")
    print(f"📈 CV Mean Score: {results['cv_results']['mean_score']:.3f}")

    return results


if __name__ == "__main__":
    # Run test if module is executed directly
    test_results = test_knn_module()
    print("✅ KNN module test passed!")