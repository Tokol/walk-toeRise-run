"""
CODE INSIGHTS: HOW WALKING ACTIVITY ANALYSIS WORKS
==================================================
This section shows the most important code snippets that make
the walking activity recognition system work.
"""

import streamlit as st


class CodeInsightsSection:
    """
    Educational section showing critical code snippets from the walking activity app.
    Each snippet explains key concepts in activity analytics.
    """

    def __init__(self):
        self.snippets = self._create_snippets()

    def _create_snippets(self):
        """Create all code snippets with explanations."""
        return {
            'feature_extraction': [
                {
                    'title': "📊 Detecting Step Patterns from Acceleration",
                    'code': '''def extract_step_features(acceleration_magnitude):
    # Find peaks = each step/foot contact
    peaks, _ = find_peaks(acceleration_magnitude,
                          height=1.5,      # Minimum peak height
                          distance=10)      # Minimum steps per second
    
    if len(peaks) >= 2:
        # Calculate step frequency (steps per second)
        time_range = len(acceleration_magnitude) / 100  # Assuming 100Hz sampling
        step_freq = len(peaks) / time_range
        
        # Calculate step regularity
        peak_intervals = np.diff(peaks)
        step_regularity = np.std(peak_intervals) / np.mean(peak_intervals)
    else:
        step_freq = 0
        step_regularity = 0
    
    return step_freq, step_regularity''',
                    'explanation': {
                        'what': "Detects steps by finding acceleration peaks",
                        'walking': "Walking: ~1.8-2.2 steps/sec, Running: ~2.5-3.5 steps/sec",
                        'importance': "Step frequency is the key to distinguishing walking from running"
                    },
                    'output_example': "Walking: 1.9 Hz, Running: 2.8 Hz, Toe-rise: irregular"
                },
                {
                    'title': "⚡ Measuring Movement Intensity (RMS)",
                    'code': '''def calculate_activity_intensity(acceleration_data):
    # Root Mean Square (RMS) - measures overall movement energy
    rms_value = np.sqrt(np.mean(acceleration_data ** 2))
    
    # Peak-to-Peak - measures movement range
    p2p_value = np.max(acceleration_data) - np.min(acceleration_data)
    
    # Standard Deviation - measures movement variability
    std_value = np.std(acceleration_data)
    
    return {
        'rms': rms_value,      # Overall energy
        'p2p': p2p_value,      # Movement range
        'std': std_value       # Variability
    }''',
                    'explanation': {
                        'what': "Calculates different measures of movement intensity",
                        'walking': "Running has highest RMS, Toe-rise has highest p2p",
                        'importance': "Different activities have distinct intensity profiles"
                    },
                    'output_example': "Walking RMS: ~1.5, Running RMS: ~2.5, Toe-rise RMS: ~1.8"
                },
                {
                    'title': "📈 Analyzing Movement Distribution (Shape Features)",
                    'code': '''def analyze_movement_shape(acceleration_data):
    # Kurtosis - how peaked/flat the distribution is
    kurt = kurtosis(acceleration_data)
    
    # Skewness - asymmetry of the distribution
    skew = skew(acceleration_data)
    
    # IQR - spread of middle 50% (robust to outliers)
    q1 = np.percentile(acceleration_data, 25)
    q3 = np.percentile(acceleration_data, 75)
    iqr = q3 - q1
    
    return {
        'kurtosis': kurt,   # High = sharp peaks (toe-rise)
        'skewness': skew,   # Positive = right-skewed
        'iqr': iqr          # Spread of typical values
    }''',
                    'explanation': {
                        'what': "Analyzes the statistical shape of movement patterns",
                        'walking': "Toe-rise has high kurtosis (sharp peaks), Running is symmetric",
                        'importance': "Shape features capture movement quality, not just quantity"
                    },
                    'output_example': "Toe-rise kurtosis: >5, Walking kurtosis: ~0, Running kurtosis: ~1"
                }
            ],
            'knn_algorithm': [
                {
                    'title': "🎯 KNN Feature Selection for Activity Recognition",
                    'code': '''# KNN uses only MOTION features (4 total)
motion_features = [
    "mean_magnitude",    # Average movement intensity
    "rms_magnitude",     # Root mean square (energy)
    "std_magnitude",     # Movement variability  
    "p2p_magnitude"      # Movement range
]

# Select only these 4 features for KNN
X_knn = features_df[motion_features].copy()

# Why not use shape features in KNN?
# Shape features (kurtosis, skewness, etc.) have different scales
# KNN's distance calculations get confused by mixed scales''',
                    'explanation': {
                        'what': "KNN works best with simple, scaled motion features",
                        'walking': "Compares 'how much' movement, not movement patterns",
                        'importance': "Too many features confuse KNN's nearest neighbor search"
                    },
                    'output_example': "Uses 4 motion features: mean, RMS, std, peak-to-peak"
                },
                {
                    'title': "⚖️ Why Feature Scaling is CRITICAL for KNN",
                    'code': '''# WITHOUT SCALING - Features have different units!
features_before_scaling = {
    "mean_magnitude": 1.52,    # m/s²
    "rms_magnitude": 1.87,     # m/s²  
    "peak_count": 3,           # count
    "kurtosis_magnitude": 0.5  # unitless
}

# KNN distance = √[(1.52-1.87)² + (3-0.5)²] 
# → Peak count dominates because it's larger!

# WITH SCALING - All features comparable
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Now all features have mean=0, std=1
# KNN can compare them fairly!''',
                    'explanation': {
                        'what': "Normalizes all features to same scale (mean=0, std=1)",
                        'walking': "Makes step count comparable to acceleration intensity",
                        'importance': "Without scaling, KNN accuracy drops 20-40%"
                    },
                    'output_example': "Before scaling: 70% accuracy, After scaling: 90% accuracy"
                },
                {
                    'title': "🔍 Finding Optimal k for Your Data",
                    'code': '''def find_optimal_k(X_train, y_train, cv_folds=5):
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    
    k_values = range(1, 16, 2)  # Test odd k values: 1, 3, 5, ..., 15
    cv_scores = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=cv_folds)
        cv_scores.append(np.mean(scores))
    
    # Find k with highest CV accuracy
    optimal_k = k_values[np.argmax(cv_scores)]
    
    return optimal_k, cv_scores''',
                    'explanation': {
                        'what': "Uses cross-validation to find the best number of neighbors",
                        'walking': "Small k overfits noise, large k misses patterns",
                        'importance': "Optimal k balances noise sensitivity and pattern recognition"
                    },
                    'output_example': "Optimal k found: 5 (tested k=1,3,5,7,9,11,13,15)"
                }
            ],
            'random_forest': [
                {
                    'title': "🌳 Random Forest Uses All Features for Deeper Insights",
                    'code': '''# Random Forest uses ALL 9 features
all_features = [
    # Motion Features (4) - What you're doing
    "mean_magnitude", "rms_magnitude", "std_magnitude", "p2p_magnitude",
    
    # Shape Features (5) - How you're doing it  
    "kurtosis_magnitude",   # Movement sharpness
    "skewness_magnitude",   # Movement asymmetry
    "median_magnitude",     # Typical value
    "iqr_magnitude",        # Spread (robust)
    "peak_count"           # Step/activity count
]

# RF can handle different scales naturally
X_rf = features_df[all_features].copy()

# Each tree uses random subsets of features
# This reduces overfitting and reveals feature importance''',
                    'explanation': {
                        'what': "RF works with complete movement profile",
                        'walking': "Analyzes both movement intensity AND movement quality",
                        'importance': "Captures complex activity patterns that simple features miss"
                    },
                    'output_example': "Uses all 9 features: 4 motion + 5 shape features"
                },
                {
                    'title': "🏆 RF Feature Importance: What Really Matters",
                    'code': '''# After training Random Forest
rf_model.fit(X_train, y_train)

# Get feature importance scores
feature_importance = rf_model.feature_importances_

# Create dictionary for interpretation
importance_dict = dict(zip(feature_names, feature_importance))

# Sort by importance
sorted_importance = sorted(importance_dict.items(), 
                          key=lambda x: x[1], 
                          reverse=True)

# Example results for walking activities:
# [
#     ("rms_magnitude", 0.32),     # Most important - overall energy
#     ("peak_count", 0.25),        # Step frequency
#     ("p2p_magnitude", 0.18),     # Movement range
#     ("kurtosis_magnitude", 0.12), # Movement sharpness
#     ... other features
# ]''',
                    'explanation': {
                        'what': "Shows which features most influence classification",
                        'walking': "RMS (energy) is usually most important for activity recognition",
                        'importance': "Tells you what to measure if you want to build a simple classifier"
                    },
                    'output_example': "Top features: RMS (32%), Peak count (25%), P2P (18%)"
                },
                {
                    'title': "⚡ Random Forest Hyperparameter Optimization",
                    'code': '''def optimize_random_forest(X_train, y_train):
    from sklearn.model_selection import GridSearchCV
    
    param_grid = {
        'n_estimators': [50, 100, 200],      # Number of trees
        'max_depth': [None, 10, 20, 30],     # Tree depth
        'min_samples_split': [2, 5, 10],     # Minimum samples to split
        'min_samples_leaf': [1, 2, 4]        # Minimum samples per leaf
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_params_, grid_search.best_score_''',
                    'explanation': {
                        'what': "Systematically tests parameter combinations",
                        'walking': "Finds optimal tree depth, number of trees, etc.",
                        'importance': "Can improve accuracy 5-15% over default parameters"
                    },
                    'output_example': "Best params: 200 trees, max_depth=20, accuracy: 94%"
                }
            ],
            'comparison': [
                {
                    'title': "🤔 KNN vs Random Forest: Which to Choose?",
                    'code': '''# SIDE-BY-SIDE COMPARISON FOR ACTIVITY RECOGNITION
# ==================== KNN ===================  ============== RF ================
# Features:     4 motion features              # 9 features (motion + shape)
# Scaling:     REQUIRED                        # Not required
# Training:    Instant (stores data)           # Slow (builds trees)
# Prediction:  Slow (searches all data)        # Fast (tree traversal)
# Interpret:   "Similar to example X"          # "Feature Y is important"
# Best for:    Quick prototypes                # Production systems
# Accuracy:    85-90%                          # 90-95%
# Speed:       Fast training, slow prediction  # Slow training, fast prediction''',
                    'explanation': {
                        'what': "Different algorithms for different use cases",
                        'walking': "KNN: Quick testing, RF: Final deployment",
                        'importance': "Choose based on your needs: speed vs accuracy vs interpretability"
                    },
                    'output_example': "KNN: 87% accuracy in 0.1s, RF: 92% accuracy in 2s"
                },
                {
                    'title': "🎯 When to Use Each Algorithm",
                    'code': '''# DECISION TREE FOR ACTIVITY RECOGNITION

def choose_algorithm(requirements):
    if requirements["speed"] == "fast" and requirements["data"] == "small":
        return "KNN"  # Quick results with small data
        
    elif requirements["accuracy"] == "high" and requirements["interpret"] == "yes":
        return "Random Forest"  # Best accuracy + feature importance
        
    elif requirements["real_time"] == "yes":
        # For real-time prediction on mobile/watch
        if requirements["battery"] == "critical":
            return "KNN"  # Lighter prediction
        else:
            return "Random Forest"  # Better accuracy
            
    else:
        return "Random Forest"  # Default choice

# Example: Smartwatch app
requirements = {
    "speed": "medium",
    "accuracy": "high", 
    "interpret": "yes",      # Show user "improve your stride"
    "real_time": "yes",
    "battery": "moderate",
    "data": "medium"
}

choice = choose_algorithm(requirements)  # Returns "Random Forest"''',
                    'explanation': {
                        'what': "Guidelines for selecting the right algorithm",
                        'walking': "Mobile apps: KNN for battery, RF for accuracy",
                        'importance': "The right tool depends on your specific application"
                    },
                    'output_example': "Smart fitness tracker: Random Forest (accuracy > battery)"
                }
            ],
            'practical_applications': [
                {
                    'title': "🏃‍♂️ Real-World Application: Step Counter Algorithm",
                    'code': '''def simple_step_counter(acceleration_magnitude, sampling_rate=100):
    # 1. Apply low-pass filter to remove noise
    from scipy.signal import butter, filtfilt
    b, a = butter(3, 3/(sampling_rate/2), btype='low')
    filtered_signal = filtfilt(b, a, acceleration_magnitude)
    
    # 2. Find peaks (steps)
    peaks, _ = find_peaks(filtered_signal, 
                         height=1.0,      # Minimum step threshold
                         distance=sampling_rate/3)  # Max 3 steps/sec
    
    # 3. Classify activity based on step frequency
    step_freq = len(peaks) / (len(filtered_signal) / sampling_rate)
    
    if step_freq < 1.5:
        activity = "standing"
    elif step_freq < 2.5:
        activity = "walking" 
    elif step_freq < 4.0:
        activity = "running"
    else:
        activity = "unknown"
    
    return len(peaks), activity, step_freq''',
                    'explanation': {
                        'what': "Simple algorithm for step counting and activity classification",
                        'walking': "Used in fitness trackers and smartwatches",
                        'importance': "Shows how research translates to real products"
                    },
                    'output_example': "Output: 42 steps, walking, 1.8 Hz"
                },
                {
                    'title': "📱 Building a Mobile Activity Classifier",
                    'code': '''# Simplified mobile implementation
class MobileActivityClassifier:
    def __init__(self, model_type="knn"):
        if model_type == "knn":
            self.model = KNeighborsClassifier(n_neighbors=5)
            self.scaler = StandardScaler()
        else:
            self.model = RandomForestClassifier(n_estimators=50)
            self.scaler = None
    
    def predict_activity(self, acceleration_window):
        # Extract features from 2-second window
        features = self.extract_features(acceleration_window)
        
        # Scale if using KNN
        if self.scaler:
            features = self.scaler.transform([features])
        
        # Predict activity
        prediction = self.model.predict(features)[0]
        
        # Confidence score
        if hasattr(self.model, "predict_proba"):
            confidence = np.max(self.model.predict_proba(features)[0])
        else:
            confidence = 1.0
        
        return prediction, confidence
    
    def extract_features(self, data):
        # Simple features for mobile efficiency
        return [
            np.mean(data),      # Mean
            np.std(data),       # Std
            np.max(data) - np.min(data),  # Peak-to-peak
            len(find_peaks(data)[0])      # Peak count
        ]''',
                    'explanation': {
                        'what': "Lightweight classifier for mobile devices",
                        'walking': "Can run on smartwatch with limited compute",
                        'importance': "Demonstrates practical deployment considerations"
                    },
                    'output_example': "Prediction: running, Confidence: 92%"
                }
            ]
        }

    def display_snippet(self, category, snippet_idx):
        """Display a single code snippet with explanation."""
        snippet = self.snippets[category][snippet_idx]

        st.markdown(f"### {snippet['title']}")

        # Code display
        with st.expander("📝 View Code", expanded=True):
            st.code(snippet['code'], language='python')

        # Explanation
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**What this does:**\n{snippet['explanation']['what']}")
        with col2:
            st.success(f"**Activity meaning:**\n{snippet['explanation']['walking']}")
        with col3:
            st.warning(f"**Why it matters:**\n{snippet['explanation']['importance']}")

        # Output example
        if snippet['output_example']:
            st.markdown(f"**Example output:** `{snippet['output_example']}`")

        st.divider()

    def display_category(self, category_name, category_label):
        """Display all snippets in a category."""
        st.markdown(f"## {category_label}")

        for i in range(len(self.snippets[category_name])):
            self.display_snippet(category_name, i)

    def display_all(self):
        """Display the complete code insights section."""
        st.title("🔍 Code Insights: How Activity Recognition Works")
        st.markdown("""
        This section shows the **most important code** that makes the walking activity
        recognition system work. Each snippet explains a key concept in movement analytics.
        """)

        # Feature Extraction
        self.display_category(
            'feature_extraction',
            "📊 Feature Extraction: From Acceleration to Activity Metrics"
        )

        # KNN Algorithm
        self.display_category(
            'knn_algorithm',
            "🎯 K-Nearest Neighbors: Simple but Effective"
        )

        # Random Forest
        self.display_category(
            'random_forest',
            "🌳 Random Forest: The Power of Ensemble Learning"
        )

        # Comparison
        self.display_category(
            'comparison',
            "🤔 Algorithm Comparison: Making the Right Choice"
        )

        # Practical Applications
        self.display_category(
            'practical_applications',
            "🏃‍♂️ Practical Applications: From Research to Real Products"
        )

        # Summary
        st.markdown("---")
        st.markdown("""
        ### 🎓 Key Takeaways:

        1. **Feature Engineering is Key**: Good features make classification easy
        2. **Different Tools for Different Jobs**:
           - **KNN**: Quick prototyping, small datasets
           - **Random Forest**: Production systems, feature insights
        3. **Think About Deployment**: Mobile vs server, battery vs accuracy
        4. **Start Simple**: Mean, RMS, and peak count often work surprisingly well
        
        **Next Steps:**
        - Try building your own step counter
        - Experiment with different features
        - Consider how you'd deploy this on a smartwatch
        
        Keep moving and stay active! 🚶‍♂️👣🏃‍♂️
        """)


def add_code_insights_section():
    """Add the code insights section to the walking app."""
    insights = CodeInsightsSection()
    insights.display_all()
