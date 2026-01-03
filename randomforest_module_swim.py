"""
RANDOM FOREST MODULE FOR SWIMMING APP
=====================================================
This module provides Random Forest functionality for swimming stroke recognition.
It uses ALL 16 features (8 motion + 8 shape) for swimming stroke classification.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import plotly.graph_objects as go

import time


class RandomForestModelSwim:
    """
    Random Forest classifier for swimming stroke recognition.
    Uses ALL 16 features (8 motion + 8 shape) for prediction.
    """

    def __init__(self, n_estimators=100, random_state=42):
        """
        Initialize Random Forest model for swimming.

        Parameters:
        -----------
        n_estimators : int
            Number of trees in the forest
        random_state : int
            Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1  # Use all CPU cores for faster training
        )
        self.is_trained = False
        self.training_time = None
        self.prediction_time = None

        # ALL 16 features for Random Forest (swimming)
        self.feature_names = [
            # 8 Motion features
            "acc_mean",
            "acc_rms",
            "acc_std",
            "acc_p2p",
            "gyro_mean",
            "gyro_rms",
            "gyro_std",
            "gyro_p2p",
            # 8 Shape features
            "pitch_kurtosis",
            "pitch_skewness",
            "pitch_peak_count",
            "roll_asymmetry",
            "stroke_frequency",
            "stroke_rhythm_cv",
            "gyro_kurtosis",
            "gyro_skewness"
        ]

    def select_features(self, features_df):
        """
        Select ALL features for Random Forest (swimming).

        Parameters:
        -----------
        features_df : pandas DataFrame
            Complete features dataframe

        Returns:
        --------
        X_rf : pandas DataFrame
            Features with all columns
        y : pandas Series
            Target labels
        """
        # Check which features are available
        available_features = [f for f in self.feature_names if f in features_df.columns]
        
        if len(available_features) < 8:  # Need at least 8 features
            # Try to find alternative feature names
            alt_feature_mapping = {
                # Motion features alternatives
                "acc_mean": ["mean_magnitude", "mean_acc", "acceleration_mean"],
                "acc_rms": ["rms_magnitude", "rms_acc", "acceleration_rms"],
                "acc_std": ["std_magnitude", "std_acc", "acceleration_std"],
                "acc_p2p": ["p2p_magnitude", "p2p_acc", "acceleration_p2p"],
                "gyro_mean": ["mean_gyro", "gyroscope_mean"],
                "gyro_rms": ["rms_gyro", "gyroscope_rms"],
                "gyro_std": ["std_gyro", "gyroscope_std"],
                "gyro_p2p": ["p2p_gyro", "gyroscope_p2p"],
                # Shape features alternatives
                "pitch_kurtosis": ["pitch_kurt", "kurtosis_pitch"],
                "pitch_skewness": ["pitch_skew", "skewness_pitch"],
                "gyro_kurtosis": ["gyro_kurt", "kurtosis_gyro"],
                "gyro_skewness": ["gyro_skew", "skewness_gyro"]
            }
            
            # Try to find alternative names
            found_features = []
            for feature in self.feature_names:
                if feature not in available_features and feature in alt_feature_mapping:
                    for alt_name in alt_feature_mapping[feature]:
                        if alt_name in features_df.columns:
                            # Don't modify the original dataframe, just track it
                            found_features.append(feature)
                            break
        
        if len(available_features) < 4:  # Even more lenient for RF
            # Use whatever features are available (excluding non-feature columns)
            non_feature_cols = ["activity", "segment_start", "segment_end", "segment_index", "time", "timestamp"]
            all_features = [col for col in features_df.columns if col not in non_feature_cols]
            available_features = all_features
            
            if len(available_features) < 2:
                raise ValueError(f"Not enough features for Random Forest. Found: {available_features}")
        
        # Use available features
        X_rf = features_df[available_features].copy()
        y = features_df["activity"].copy()
        
        # Update feature names to match what we're actually using
        self.feature_names = available_features
        
        return X_rf, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data for Random Forest.
        Note: Random Forest doesn't need feature scaling!

        Parameters:
        -----------
        X : pandas DataFrame
            Features
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

    def predict_proba(self, X):
        """
        Get prediction probabilities.

        Parameters:
        -----------
        X : pandas DataFrame
            Features to predict

        Returns:
        --------
        probabilities : numpy array
            Prediction probabilities for each class
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction!")

        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test, y_pred=None):
        """
        Evaluate model performance.

        Parameters:
        -----------
        X_test : pandas DataFrame
            Test features
        y_test : numpy array or pandas Series
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
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        else:
            feature_importance = dict(zip(self.feature_names, [0] * len(self.feature_names)))

        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm,
            'classification_report': report,
            'per_class_accuracy': dict(zip(classes, per_class_accuracy)),
            'feature_importance': feature_importance,
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
        # Use StratifiedKFold to maintain class distribution
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Perform cross-validation on training data
        cv_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=skf,
            scoring='accuracy',
            n_jobs=-1  # Use all CPU cores
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
    if feature_importance:
        features = list(feature_importance.keys())
        importance_values = list(feature_importance.values())

        # Sort by importance (descending)
        sorted_idx = np.argsort(importance_values)[::-1]
        features_sorted = [features[i] for i in sorted_idx]
        values_sorted = [importance_values[i] for i in sorted_idx]

        # Take top 15 features for readability
        top_n = min(15, len(features_sorted))
        features_top = features_sorted[:top_n]
        values_top = values_sorted[:top_n]

        # Create display names
        features_display = [f.replace('_', ' ').title() for f in features_top]

        fig_fi = go.Figure(data=[
            go.Bar(
                x=features_display,
                y=values_top,
                marker_color='lightblue',
                text=[f'{val:.3f}' for val in values_top],
                textposition='auto'
            )
        ])

        fig_fi.update_layout(
            title=f'Random Forest Feature Importance (Top {top_n})',
            xaxis_title='Features',
            yaxis_title='Importance Score',
            height=500,
            xaxis=dict(tickangle=45)
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


def run_rf_pipeline_swim(features_df, test_size=0.2, n_estimators=100, random_state=42):
    """
    Complete Random Forest pipeline for swimming.

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
    print("🏊‍♂️ Starting Random Forest pipeline for swimming...")

    # Initialize RF model
    rf = RandomForestModelSwim(n_estimators=n_estimators, random_state=random_state)

    # Step 1: Select features
    print("📊 Step 1: Selecting features for Random Forest...")
    X_rf, y = rf.select_features(features_df)
    
    print(f"   Using {len(rf.feature_names)} features: {rf.feature_names}")

    # Step 2: Split data (no scaling needed for RF)
    print("🔪 Step 2: Splitting data...")
    data_splits = rf.split_data(X_rf, y, test_size=test_size, random_state=random_state)
    print(f"   Training samples: {len(data_splits['X_train'])}")
    print(f"   Testing samples: {len(data_splits['X_test'])}")

    # Step 3: Train model
    print("🏋️ Step 3: Training Random Forest...")
    training_time = rf.train_model(data_splits['X_train'], data_splits['y_train'])
    print(f"   Training time: {training_time:.3f}s")

    # Step 4: Evaluate
    print("📈 Step 4: Evaluating on test set...")
    metrics = rf.evaluate(data_splits['X_test'], data_splits['y_test'])
    print(f"   Test accuracy: {metrics['accuracy']:.3f}")

    # Step 5: Cross-validation on TRAINING DATA ONLY (prevents data leakage)
    print("🔄 Step 5: Performing cross-validation on TRAINING data only...")
    cv_results = rf.cross_validate(
        data_splits['X_train'],  # Use training features only
        data_splits['y_train'],  # Use training labels only
        cv=5
    )
    print(f"   CV mean accuracy: {cv_results['mean_score']:.3f}")

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

    print("✅ Random Forest pipeline for swimming completed!")
    return results


def test_rf_module_swim():
    """Quick test to verify the swimming RF module works."""
    print("🧪 Testing Random Forest module for swimming...")

    # Create dummy swimming data
    np.random.seed(42)
    n_samples = 200  # More samples for better RF performance

    # Create dummy swimming features with some correlation patterns
    dummy_data = {
        # Motion features
        "acc_mean": np.random.randn(n_samples) + 10,
        "acc_rms": np.random.randn(n_samples) + 12,
        "acc_std": np.random.randn(n_samples) + 2,
        "acc_p2p": np.random.randn(n_samples) + 5,
        "gyro_mean": np.random.randn(n_samples) + 8,
        "gyro_rms": np.random.randn(n_samples) + 10,
        "gyro_std": np.random.randn(n_samples) + 1.5,
        "gyro_p2p": np.random.randn(n_samples) + 4,
        # Shape features
        "pitch_kurtosis": np.random.randn(n_samples),
        "pitch_skewness": np.random.randn(n_samples),
        "pitch_peak_count": np.random.randint(1, 10, n_samples),
        "roll_asymmetry": np.random.randn(n_samples) * 0.5,
        "stroke_frequency": np.random.randn(n_samples) + 1.5,
        "stroke_rhythm_cv": np.random.randn(n_samples) * 0.3 + 0.2,
        "gyro_kurtosis": np.random.randn(n_samples),
        "gyro_skewness": np.random.randn(n_samples),
    }
    
    # Create meaningful patterns for different strokes
    stroke_patterns = {
        'butterfly': {'acc_mean': 12, 'stroke_frequency': 2.0},
        'backstroke': {'gyro_mean': 9, 'roll_asymmetry': -0.3},
        'breaststroke': {'acc_p2p': 7, 'stroke_frequency': 1.0},
        'freestyle': {'acc_rms': 13, 'stroke_frequency': 1.5}
    }
    
    # Generate labels with patterns
    activities = []
    for i in range(n_samples):
        stroke = np.random.choice(['butterfly', 'backstroke', 'breaststroke', 'freestyle'])
        activities.append(stroke)
        # Add some pattern based on stroke
        pattern = stroke_patterns[stroke]
        for feature, value in pattern.items():
            if feature in dummy_data:
                dummy_data[feature][i] += np.random.randn() * 0.5 + value
    
    dummy_data["activity"] = activities
    features_df = pd.DataFrame(dummy_data)

    # Run RF pipeline
    results = run_rf_pipeline_swim(features_df, n_estimators=50)

    print(f"📊 Accuracy: {results['metrics']['accuracy']:.3f}")
    print(f"⏱️ Training time: {results['metrics']['training_time']:.3f}s")
    print(f"📈 CV Mean Score: {results['cv_results']['mean_score']:.3f}")
    print(f"🎯 Features used: {len(results['model'].feature_names)}")
    
    # Show top 3 features by importance
    feature_importance = results['metrics']['feature_importance']
    if feature_importance:
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        print("🏆 Top 3 Important Features:")
        for i, (feature, importance) in enumerate(sorted_features[:3], 1):
            print(f"  {i}. {feature}: {importance:.3f}")

    return results


if __name__ == "__main__":
    # Run test if module is executed directly
    test_results = test_rf_module_swim()
    print("✅ Random Forest module for swimming test passed!")