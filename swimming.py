import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import kurtosis, skew
from feature_description_dashboard import FeatureDescriptionDashboard

# Import custom modules
try:
    from knn_module_swim import run_knn_pipeline_swim as run_knn_pipeline
except ImportError:
    # Create fallback function
    def run_knn_pipeline(features_df, test_size=0.2, n_neighbors=5, random_state=42):
        st.error("❌ knn_module_swim.py not found. Please create the swimming KNN module.")
        return None

try:
    from randomforest_module_swim import run_rf_pipeline_swim as run_rf_pipeline
except ImportError:
    # Create fallback function  
    def run_rf_pipeline(features_df, test_size=0.2, n_estimators=100, random_state=42):
        st.error("❌ randomforest_module_swim.py not found. Please create the swimming Random Forest module.")
        return None

import pickle

# Add this after your imports but before the main app code

st.set_page_config(
    page_title="Swimming Stroke Recognition",
    page_icon="assets/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded"  # expand sidebar
)

SWIMMING_FEATURE_INFO = [
    # Motion Features (8 features)
    {
        "Feature": "acc_mean",
        "Type": "Motion",
        "Signal Source": "Acceleration magnitude",
        "Meaning": "Average acceleration magnitude in the 2-second window",
        "Why Useful": "Indicates overall swimming intensity - higher for powerful strokes like butterfly"
    },
    {
        "Feature": "acc_rms",
        "Type": "Motion", 
        "Signal Source": "Acceleration magnitude",
        "Meaning": "Root Mean Square (RMS) of acceleration",
        "Why Useful": "Measures swimming energy/power - distinguishes high-energy strokes"
    },
    {
        "Feature": "acc_std",
        "Type": "Motion",
        "Signal Source": "Acceleration magnitude",
        "Meaning": "Standard deviation of acceleration",
        "Why Useful": "Measures acceleration variability - indicates stroke smoothness"
    },
    {
        "Feature": "acc_p2p",
        "Type": "Motion",
        "Signal Source": "Acceleration magnitude",
        "Meaning": "Peak-to-peak range (max - min) of acceleration",
        "Why Useful": "Shows acceleration extremes - high for explosive strokes"
    },
    {
        "Feature": "gyro_mean",
        "Type": "Motion",
        "Signal Source": "Gyroscope magnitude",
        "Meaning": "Average rotation magnitude in the 2-second window",
        "Why Useful": "Indicates overall body rotation - higher for rotational strokes"
    },
    {
        "Feature": "gyro_rms",
        "Type": "Motion",
        "Signal Source": "Gyroscope magnitude",
        "Meaning": "Root Mean Square (RMS) of rotation",
        "Why Useful": "Measures rotation energy - distinguishes stroke rotation patterns"
    },
    {
        "Feature": "gyro_std",
        "Type": "Motion",
        "Signal Source": "Gyroscope magnitude",
        "Meaning": "Standard deviation of rotation",
        "Why Useful": "Measures rotation consistency - low for smooth strokes"
    },
    {
        "Feature": "gyro_p2p",
        "Type": "Motion",
        "Signal Source": "Gyroscope magnitude",
        "Meaning": "Peak-to-peak range of rotation",
        "Why Useful": "Shows rotation extremes - identifies stroke reversals"
    },
    
    # Shape Features (8 features)
    {
        "Feature": "pitch_kurtosis",
        "Type": "Shape",
        "Signal Source": "Pitch angle",
        "Meaning": "Kurtosis (tailedness) of pitch distribution",
        "Why Useful": "Indicates sharpness of pitch changes - high for butterfly undulation"
    },
    {
        "Feature": "pitch_skewness",
        "Type": "Shape",
        "Signal Source": "Pitch angle",
        "Meaning": "Skewness (asymmetry) of pitch distribution",
        "Why Useful": "Shows pitch direction bias - different for each stroke"
    },
    {
        "Feature": "pitch_peak_count",
        "Type": "Shape",
        "Signal Source": "Pitch angle",
        "Meaning": "Number of pitch peaks in the 2-second window",
        "Why Useful": "Counts body undulations - stroke cycle indicator"
    },
    {
        "Feature": "roll_asymmetry",
        "Type": "Shape",
        "Signal Source": "Roll angle",
        "Meaning": "Difference between right and left roll averages",
        "Why Useful": "Measures body roll symmetry - distinguishes asymmetric strokes"
    },
    {
        "Feature": "stroke_frequency",
        "Type": "Shape",
        "Signal Source": "Gyroscope Z-axis",
        "Meaning": "Estimated stroke cycles per second (Hz)",
        "Why Useful": "Stroke rate - varies between stroke types"
    },
    {
        "Feature": "stroke_rhythm_cv",
        "Type": "Shape",
        "Signal Source": "Gyroscope Z-axis",
        "Meaning": "Coefficient of variation in stroke timing",
        "Why Useful": "Measures stroke rhythm consistency - low for breaststroke"
    },
    {
        "Feature": "gyro_kurtosis",
        "Type": "Shape",
        "Signal Source": "Gyroscope magnitude",
        "Meaning": "Kurtosis of rotation distribution",
        "Why Useful": "Indicates rotation pattern sharpness"
    },
    {
        "Feature": "gyro_skewness",
        "Type": "Shape",
        "Signal Source": "Gyroscope magnitude",
        "Meaning": "Skewness of rotation distribution",
        "Why Useful": "Shows rotation direction bias"
    }
]


# ==========================
# PAGE CONFIG
# ==========================
st.title("🏊‍♂️ Medley Swimming Stroke Recognition")
st.markdown("""
This tool analyzes **Apple Watch IMU data** from medley swimming, extracts stroke-specific features, 
and compares **K-Nearest Neighbors (KNN)** vs **Random Forest** algorithms for swimming stroke classification.
""")

# ==========================
# CUSTOM FUNCTIONS FOR SWIMMING
# ==========================
def compute_magnitude_for_swimming(df):
    """Compute magnitude for acceleration and gyroscope for swimming data"""
    # Check column names and create standard ones
    df = df.copy()
    
    # Standardize column names if needed
    column_mapping = {
        "seconds_elapsed": "Time (s)",
        "accelerationX": "Acc_X",
        "accelerationY": "Acc_Y", 
        "accelerationZ": "Acc_Z",
        "rotationRateX": "Gyro_X",
        "rotationRateY": "Gyro_Y",
        "rotationRateZ": "Gyro_Z",
        "pitch": "Pitch",
        "roll": "Roll",
        "yaw": "Yaw"
    }
    
    # Rename columns if they exist
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df.rename(columns={old_name: new_name}, inplace=True)
    
    # Compute magnitudes
    if all(col in df.columns for col in ["Acc_X", "Acc_Y", "Acc_Z"]):
        df["acc_magnitude"] = np.sqrt(df["Acc_X"]**2 + df["Acc_Y"]**2 + df["Acc_Z"]**2)
    
    if all(col in df.columns for col in ["Gyro_X", "Gyro_Y", "Gyro_Z"]):
        df["gyro_magnitude"] = np.sqrt(df["Gyro_X"]**2 + df["Gyro_Y"]**2 + df["Gyro_Z"]**2)
    
    return df

def extract_swimming_features(df, segment_duration, activities):
    """Extract swimming-specific features"""
    features = []
    
    # Get time column
    time_col = "Time (s)" if "Time (s)" in df.columns else "seconds_elapsed"
    
    start_time = df[time_col].min()
    end_time = df[time_col].max()
    
    for seg_start in np.arange(start_time, end_time, segment_duration):
        seg_end = seg_start + segment_duration
        
        seg_df = df[
            (df[time_col] >= seg_start) &
            (df[time_col] < seg_end)
        ]
        
        if seg_df.empty:
            continue
        
        # Check if we have the required columns
        has_acc = "acc_magnitude" in seg_df.columns
        has_gyro = "gyro_magnitude" in seg_df.columns
        has_pitch = "Pitch" in seg_df.columns or "pitch" in seg_df.columns
        has_roll = "Roll" in seg_df.columns or "roll" in seg_df.columns
        has_gyro_z = "Gyro_Z" in seg_df.columns or "rotationRateZ" in seg_df.columns
        
        # Activity label - check which activity this segment belongs to
        activity_label = None
        for activity in activities:
            if activity["start"] <= seg_start < activity["end"]:
                activity_label = activity["name"]
                break
        
        if activity_label is None:
            # Segment doesn't belong to any defined activity
            continue
        
        # Initialize feature dictionary
        feature_dict = {
            "segment_start": seg_start,
            "segment_end": seg_end,
            "activity": activity_label
        }
        
        # ==========================
        # MOTION FEATURES (8 features)
        # ==========================
        if has_acc:
            acc_mag = seg_df["acc_magnitude"]
            feature_dict.update({
                "acc_mean": acc_mag.mean(),
                "acc_rms": np.sqrt(np.mean(acc_mag**2)),
                "acc_std": acc_mag.std(),
                "acc_p2p": acc_mag.max() - acc_mag.min(),
            })
        else:
            # If no acceleration data, fill with zeros
            feature_dict.update({
                "acc_mean": 0,
                "acc_rms": 0,
                "acc_std": 0,
                "acc_p2p": 0,
            })
        
        if has_gyro:
            gyro_mag = seg_df["gyro_magnitude"]
            feature_dict.update({
                "gyro_mean": gyro_mag.mean(),
                "gyro_rms": np.sqrt(np.mean(gyro_mag**2)),
                "gyro_std": gyro_mag.std(),
                "gyro_p2p": gyro_mag.max() - gyro_mag.min(),
            })
        else:
            # If no gyro data, fill with zeros
            feature_dict.update({
                "gyro_mean": 0,
                "gyro_rms": 0,
                "gyro_std": 0,
                "gyro_p2p": 0,
            })
        
        # ==========================
        # SHAPE FEATURES (8 features)
        # ==========================
        if has_pitch:
            pitch_col = "Pitch" if "Pitch" in seg_df.columns else "pitch"
            pitch_seg = seg_df[pitch_col].values
            feature_dict.update({
                "pitch_kurtosis": kurtosis(pitch_seg) if len(pitch_seg) > 3 else 0,
                "pitch_skewness": skew(pitch_seg) if len(pitch_seg) > 3 else 0,
                "pitch_peak_count": len(find_peaks(pitch_seg)[0]) if len(pitch_seg) > 1 else 0,
            })
        else:
            feature_dict.update({
                "pitch_kurtosis": 0,
                "pitch_skewness": 0,
                "pitch_peak_count": 0,
            })
        
        if has_roll:
            roll_col = "Roll" if "Roll" in seg_df.columns else "roll"
            roll_seg = seg_df[roll_col].values
            # Roll asymmetry
            if len(roll_seg) > 0 and (roll_seg > 0).any() and (roll_seg < 0).any():
                roll_asym = np.mean(roll_seg[roll_seg > 0]) - np.mean(roll_seg[roll_seg < 0])
            else:
                roll_asym = 0
            feature_dict["roll_asymmetry"] = roll_asym
        else:
            feature_dict["roll_asymmetry"] = 0
        
        # Stroke features from gyro Z
        if has_gyro_z:
            gyro_z_col = "Gyro_Z" if "Gyro_Z" in seg_df.columns else "rotationRateZ"
            gyro_z_seg = seg_df[gyro_z_col].values
            
            # Stroke frequency
            peaks, _ = find_peaks(gyro_z_seg)
            stroke_freq = len(peaks) / segment_duration if len(gyro_z_seg) > 0 else 0
            
            # Stroke rhythm regularity
            if len(peaks) > 2:
                intervals = np.diff(peaks)
                rhythm_cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0
            else:
                rhythm_cv = 0
            
            feature_dict.update({
                "stroke_frequency": stroke_freq,
                "stroke_rhythm_cv": rhythm_cv,
            })
        else:
            feature_dict.update({
                "stroke_frequency": 0,
                "stroke_rhythm_cv": 0,
            })
        
        # Additional gyro shape features
        if has_gyro:
            gyro_mag = seg_df["gyro_magnitude"].values
            feature_dict.update({
                "gyro_kurtosis": kurtosis(gyro_mag) if len(gyro_mag) > 3 else 0,
                "gyro_skewness": skew(gyro_mag) if len(gyro_mag) > 3 else 0,
            })
        else:
            feature_dict.update({
                "gyro_kurtosis": 0,
                "gyro_skewness": 0,
            })
        
        features.append(feature_dict)
    
    features_df = pd.DataFrame(features)
    return features_df


# ==========================
# CSV UPLOAD (in session state)
# ==========================
if 'df' not in st.session_state:
    st.session_state.df = None
    st.session_state.features_df = None
    st.session_state.time_ranges = None
    st.session_state.knn_results = None
    st.session_state.rf_results = None

uploaded_file = st.file_uploader(
    "📤 Upload Swimming CSV file (CSV only)",
    type=["csv"]
)

# Initialize or reset session state when new file is uploaded
if uploaded_file is not None and st.session_state.df is None:
    df = pd.read_csv(uploaded_file)
    
    # Check for required columns (flexible for swimming data)
    required_cols_options = [
        ["seconds_elapsed", "accelerationX", "accelerationY", "accelerationZ"],
        ["Time (s)", "Acc_X", "Acc_Y", "Acc_Z"],
        ["time", "ax", "ay", "az"],
        ["timestamp", "accelerometerAccelerationX", "accelerometerAccelerationY", "accelerometerAccelerationZ"]  # Apple Watch format
    ]
    
    has_required_cols = False
    for cols in required_cols_options:
        if all(col in df.columns for col in cols):
            has_required_cols = True
            break
    
    if not has_required_cols:
        st.error(
            "❌ CSV must contain time and acceleration columns. Expected columns like: seconds_elapsed, accelerationX, accelerationY, accelerationZ or Apple Watch format"
        )
        st.stop()
    
    st.session_state.df = compute_magnitude_for_swimming(df)
    st.success("✅ Apple Watch medley swimming CSV loaded successfully!")
    
    # Reset features and results when new file is uploaded
    st.session_state.features_df = None
    st.session_state.time_ranges = None
    st.session_state.knn_results = None
    st.session_state.rf_results = None

# If no file is uploaded, show info and stop
if uploaded_file is None and st.session_state.df is None:
    st.info("📝 Please upload a swimming CSV file to continue.")
    st.stop()

# If we have data in session state, continue
if st.session_state.df is not None:
    df = st.session_state.df
    
    # ==========================
    # SIDEBAR FOR CONFIGURATION
    # ==========================
    with st.sidebar:
        st.sidebar.image("assets/logo.png", width=150)
        st.sidebar.markdown("---")
        
        st.header("⚙️ Configuration")
        
        st.subheader("Data Processing")
        segment_duration = st.slider(
            "Segment Duration (seconds)",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.5,
            help="Window size for feature extraction"
        )
        
        st.subheader("ML Algorithm Settings")
        test_size = st.slider(
            "Test Set Size (%)",
            min_value=10,
            max_value=40,
            value=20,
            help="Percentage of data for testing"
        ) / 100
        
        k_value = st.slider(
            "KNN: Number of Neighbors (k)",
            min_value=1,
            max_value=15,
            value=5,
            step=2,
            help="Number of neighbors for KNN algorithm"
        )
        
        n_trees = st.slider(
            "Random Forest: Number of Trees",
            min_value=10,
            max_value=200,
            value=100,
            step=10,
            help="Number of trees in the Random Forest"
        )
        
        st.markdown("---")
        st.info("""
        **Medley Analysis Strategy:**
        - **KNN:** Uses 8 motion features for fast classification
        - **Random Forest:** Uses all 16 features (8 motion + 8 shape) for detailed stroke analysis
        - **Data Source:** Apple Watch IMU sensors
        """)
    
    # ==========================
    # SWIMMING STROKE SELECTION (NEW SECTION)
    # ==========================
    st.header("📅 Step 1: Define Swimming Stroke Time Ranges")
    
    # Stroke Type Selection
    stroke_mode = st.radio(
        "Choose stroke type:",
        ["Default Strokes (butterfly, backstroke, breaststroke, freestyle)", "Custom Strokes (define your own)"],
        horizontal=True,
        key="stroke_mode"
    )
    
    st.markdown("---")
    
    # Initialize custom strokes in session state
    if 'custom_strokes' not in st.session_state:
        st.session_state.custom_strokes = [
            {"name": "butterfly", "start": 18.0, "end": 30.0},
            {"name": "backstroke", "start": 35.0, "end": 53.0},
            {"name": "breaststroke", "start": 57.0, "end": 82.0},
            {"name": "freestyle", "start": 87.0, "end": 105.0}
        ]
    
    # Initialize default preset selection
    if 'default_preset' not in st.session_state:
        st.session_state.default_preset = "Ellen Williamsson"
    
    # ==========================
    # DEFAULT STROKES MODE
    # ==========================
    if stroke_mode == "Default Strokes (butterfly, backstroke, breaststroke, freestyle)":
        col1, col2 = st.columns([3, 1])
        
        with col1:
            preset = st.selectbox(
                "Choose preset time ranges or select Custom to define your own:",
                ["Ellen Williamsson", "Custom"],
                key="preset_selector"
            )
            st.session_state.default_preset = preset
        
        with col2:
            st.write("")
            st.write("")
            if st.button("🔄 Reset to Preset", key="reset_preset"):
                st.session_state.features_df = None
                st.session_state.time_ranges = None
                st.rerun()
        
        # Preset defaults (using your specified times)
        if preset == "Ellen Williamsson":
            defaults = {
                "butterfly_start": 18,
                "butterfly_end": 30,
                "backstroke_start": 35,
                "backstroke_end": 53,
                "breaststroke_start": 57,
                "breaststroke_end": 82,
                "freestyle_start": 87,
                "freestyle_end": 105,
            }
        else:  # Custom
            defaults = {
                "butterfly_start": 0,
                "butterfly_end": 0,
                "backstroke_start": 0,
                "backstroke_end": 0,
                "breaststroke_start": 0,
                "breaststroke_end": 0,
                "freestyle_start": 0,
                "freestyle_end": 0,
            }
        
        # Time Range Inputs for Default Strokes
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🦋 Butterfly")
            butterfly_start = st.number_input(
                "Butterfly start (s)",
                min_value=0.0,
                value=float(defaults["butterfly_start"]),
                step=1.0,
                format="%.0f",
                key="butterfly_start_default"
            )
            butterfly_end = st.number_input(
                "Butterfly end (s)",
                min_value=0.0,
                value=float(defaults["butterfly_end"]),
                step=1.0,
                format="%.0f",
                key="butterfly_end_default"
            )
            # Store in session state
            if len(st.session_state.custom_strokes) >= 1:
                st.session_state.custom_strokes[0] = {"name": "butterfly", "start": butterfly_start, "end": butterfly_end}
            
            st.markdown("### 🐸 Breaststroke")
            breaststroke_start = st.number_input(
                "Breaststroke start (s)",
                min_value=0.0,
                value=float(defaults["breaststroke_start"]),
                step=1.0,
                format="%.0f",
                key="breaststroke_start_default"
            )
            breaststroke_end = st.number_input(
                "Breaststroke end (s)",
                min_value=0.0,
                value=float(defaults["breaststroke_end"]),
                step=1.0,
                format="%.0f",
                key="breaststroke_end_default"
            )
            # Store in session state
            if len(st.session_state.custom_strokes) >= 3:
                st.session_state.custom_strokes[2] = {"name": "breaststroke", "start": breaststroke_start, "end": breaststroke_end}
        
        with col2:
            st.markdown("### 🔄 Backstroke")
            backstroke_start = st.number_input(
                "Backstroke start (s)",
                min_value=0.0,
                value=float(defaults["backstroke_start"]),
                step=1.0,
                format="%.0f",
                key="backstroke_start_default"
            )
            backstroke_end = st.number_input(
                "Backstroke end (s)",
                min_value=0.0,
                value=float(defaults["backstroke_end"]),
                step=1.0,
                format="%.0f",
                key="backstroke_end_default"
            )
            # Store in session state
            if len(st.session_state.custom_strokes) >= 2:
                st.session_state.custom_strokes[1] = {"name": "backstroke", "start": backstroke_start, "end": backstroke_end}
            
            st.markdown("### 🏊‍♂️ Freestyle")
            freestyle_start = st.number_input(
                "Freestyle start (s)",
                min_value=0.0,
                value=float(defaults["freestyle_start"]),
                step=1.0,
                format="%.0f",
                key="freestyle_start_default"
            )
            freestyle_end = st.number_input(
                "Freestyle end (s)",
                min_value=0.0,
                value=float(defaults["freestyle_end"]),
                step=1.0,
                format="%.0f",
                key="freestyle_end_default"
            )
            # Store in session state
            if len(st.session_state.custom_strokes) >= 4:
                st.session_state.custom_strokes[3] = {"name": "freestyle", "start": freestyle_start, "end": freestyle_end}
        
        # Ensure we have exactly 4 strokes for default mode
        if len(st.session_state.custom_strokes) > 4:
            st.session_state.custom_strokes = st.session_state.custom_strokes[:4]
        
        # Store time ranges for visualization
        st.session_state.time_ranges = {
            "butterfly_start": butterfly_start,
            "butterfly_end": butterfly_end,
            "backstroke_start": backstroke_start,
            "backstroke_end": backstroke_end,
            "breaststroke_start": breaststroke_start,
            "breaststroke_end": breaststroke_end,
            "freestyle_start": freestyle_start,
            "freestyle_end": freestyle_end
        }
    
    # ==========================
    # CUSTOM STROKES MODE
    # ==========================
    else:
        st.subheader("🆕 Define Your Custom Swimming Strokes")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            num_strokes = st.number_input(
                "Number of strokes:",
                min_value=2,
                max_value=10,
                value=len(st.session_state.custom_strokes),
                step=1,
                key="num_strokes"
            )
        
        with col2:
            st.write("")
            st.write("")
            if st.button("🔄 Reset Strokes", key="reset_custom"):
                st.session_state.custom_strokes = [
                    {"name": "stroke_1", "start": 0.0, "end": 0.0},
                    {"name": "stroke_2", "start": 0.0, "end": 0.0},
                    {"name": "stroke_3", "start": 0.0, "end": 0.0},
                    {"name": "stroke_4", "start": 0.0, "end": 0.0}
                ]
                st.rerun()
        
        # Adjust custom strokes list based on number
        if num_strokes > len(st.session_state.custom_strokes):
            # Add new strokes
            for i in range(len(st.session_state.custom_strokes), num_strokes):
                st.session_state.custom_strokes.append(
                    {"name": f"stroke_{i + 1}", "start": 0.0, "end": 0.0}
                )
        elif num_strokes < len(st.session_state.custom_strokes):
            # Remove extra strokes
            st.session_state.custom_strokes = st.session_state.custom_strokes[:num_strokes]
        
        # Display custom stroke inputs
        st.markdown("### 📝 Stroke Details")
        
        # Create columns for strokes (2 per row)
        num_cols = 2
        for i in range(0, len(st.session_state.custom_strokes), num_cols):
            cols = st.columns(num_cols)
            for col_idx in range(num_cols):
                stroke_idx = i + col_idx
                if stroke_idx < len(st.session_state.custom_strokes):
                    with cols[col_idx]:
                        st.markdown(f"#### Stroke {stroke_idx + 1}")
                        
                        # Stroke name
                        stroke_name = st.text_input(
                            "Stroke name:",
                            value=st.session_state.custom_strokes[stroke_idx]["name"],
                            key=f"stroke_name_{stroke_idx}"
                        )
                        
                        # Start time
                        start_time = st.number_input(
                            "Start time (s):",
                            min_value=0.0,
                            value=float(st.session_state.custom_strokes[stroke_idx]["start"]),
                            step=1.0,
                            format="%.0f",
                            key=f"stroke_start_{stroke_idx}"
                        )
                        
                        # End time
                        end_time = st.number_input(
                            "End time (s):",
                            min_value=0.0,
                            value=float(st.session_state.custom_strokes[stroke_idx]["end"]),
                            step=1.0,
                            format="%.0f",
                            key=f"stroke_end_{stroke_idx}"
                        )
                        
                        # Update session state
                        st.session_state.custom_strokes[stroke_idx] = {
                            "name": stroke_name,
                            "start": start_time,
                            "end": end_time
                        }
        
        # Quick preset button
        st.markdown("### ⚡ Quick Preset")
        
        if st.button("📋 Ellen Williamsson Preset", use_container_width=True):
            if len(st.session_state.custom_strokes) >= 4:
                st.session_state.custom_strokes[0] = {"name": "butterfly", "start": 18.0, "end": 30.0}
                st.session_state.custom_strokes[1] = {"name": "backstroke", "start": 35.0, "end": 53.0}
                st.session_state.custom_strokes[2] = {"name": "breaststroke", "start": 57.0, "end": 82.0}
                st.session_state.custom_strokes[3] = {"name": "freestyle", "start": 87.0, "end": 105.0}
                st.rerun()
    
    # ==========================
    # PROCESS DATA BUTTON
    # ==========================
    st.markdown("---")
    
    if st.button("🔧 Process Data & Extract Features", type="primary", key="process_button"):
        with st.spinner("Processing data and extracting swimming features..."):
            # Clear previous features and results
            st.session_state.features_df = None
            st.session_state.knn_results = None
            st.session_state.rf_results = None
            
            # Get strokes from session state
            strokes = st.session_state.custom_strokes
            
            # Validate strokes
            valid_strokes = []
            for stroke in strokes:
                name = stroke["name"].strip()
                start = stroke["start"]
                end = stroke["end"]
                
                if not name:
                    st.warning(f"⚠️ Stroke name cannot be empty. Skipping stroke.")
                    continue
                
                if start >= end:
                    st.warning(f"⚠️ Stroke '{name}' has invalid time range ({start}s to {end}s). Skipping.")
                    continue
                
                valid_strokes.append(stroke)
            
            if len(valid_strokes) < 2:
                st.error("❌ Need at least 2 valid strokes with proper time ranges.")
                st.stop()
            
            # Extract swimming-specific features
            features_df = extract_swimming_features(df, segment_duration, valid_strokes)
            
            if features_df.empty:
                st.warning("⚠️ No segments fall within the selected time ranges.")
                st.stop()
            
            # Store in session state
            st.session_state.features_df = features_df
            
            # Store time ranges for visualization
            st.session_state.time_ranges = {}
            for i, stroke in enumerate(valid_strokes):
                st.session_state.time_ranges[f"stroke_{i}_start"] = stroke["start"]
                st.session_state.time_ranges[f"stroke_{i}_end"] = stroke["end"]
                st.session_state.time_ranges[f"stroke_{i}_name"] = stroke["name"]
            
            st.success(f"✅ Extracted {len(features_df)} segments for {len(valid_strokes)} swimming strokes!")
    
    # ==========================
    # DISPLAY RESULTS IF PROCESSED
    # ==========================
    if st.session_state.features_df is not None:
        features_df = st.session_state.features_df
        time_ranges = st.session_state.time_ranges
        
        st.markdown("---")
        st.header("📊 Step 2: Data Exploration & Visualization")
        
        # Get unique strokes for color mapping
        strokes_from_features = list(features_df["activity"].unique())
        num_strokes = len(strokes_from_features)
        
        # Generate colors for swimming strokes
        swimming_colors_matplotlib = {
            "butterfly": "tab:blue",
            "backstroke": "tab:green",
            "breaststroke": "tab:orange",
            "freestyle": "tab:red"
        }
        
        swimming_colors_plotly = {
            "butterfly": "blue",
            "backstroke": "green",
            "breaststroke": "orange",
            "freestyle": "red"
        }
        
        # Create stroke to color mapping for both matplotlib and plotly
        matplotlib_stroke_colors = {}
        plotly_stroke_colors = {}
        stroke_names = {}  # For display names
        
        if stroke_mode == "Default Strokes (butterfly, backstroke, breaststroke, freestyle)":
            # Use predefined colors for default strokes
            matplotlib_stroke_colors = swimming_colors_matplotlib.copy()
            plotly_stroke_colors = swimming_colors_plotly.copy()
            stroke_names = {
                "butterfly": "Butterfly",
                "backstroke": "Backstroke",
                "breaststroke": "Breaststroke",
                "freestyle": "Freestyle"
            }
        else:
            # For custom strokes, map each unique name to a color
            all_stroke_names = set()
            
            # Add from session state custom strokes
            for stroke in st.session_state.custom_strokes:
                all_stroke_names.add(stroke["name"])
            
            # Add from features_df if available
            all_stroke_names.update(features_df["activity"].unique())
            
            # Create mapping
            all_stroke_names = list(all_stroke_names)
            colors = list(swimming_colors_matplotlib.values())
            plotly_colors = list(swimming_colors_plotly.values())
            
            for i, stroke_name in enumerate(all_stroke_names):
                mat_color = colors[i % len(colors)]
                plt_color = plotly_colors[i % len(plotly_colors)]
                
                matplotlib_stroke_colors[stroke_name] = mat_color
                plotly_stroke_colors[stroke_name] = plt_color
                
                # Create display name (capitalize, replace underscores)
                display_name = stroke_name.replace("_", " ").title()
                stroke_names[stroke_name] = display_name
        
        # ==========================
        # 1. RAW SIGNAL PLOT
        # ==========================
        from raw_signal_visualizer import SwimmingSignalVisualizer
        if st.session_state.features_df is not None:
            features_df = st.session_state.features_df
            time_ranges = st.session_state.time_ranges

            # Create visualizer instance
            visualizer = SwimmingSignalVisualizer(
                df=df,  # Raw data
                features_df=features_df,  # Extracted features
                time_ranges=time_ranges,  # Stroke time ranges
                stroke_mode=stroke_mode,  # "Default Strokes" or "Custom Strokes"
                stroke_names_dict=stroke_names  # Optional: custom stroke name mapping
                )

            # Render the complete visualization dashboard
            visualizer.render_enhanced_signal_dashboard()

            # Continue with the rest of your app...
        # ==========================
        # 2. DATA OVERVIEW
        # ==========================
        st.subheader("📋 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", len(features_df))
        
        with col2:
            st.metric("Features Extracted", 16)
        
        with col3:
            st.metric("Strokes", features_df["activity"].nunique())
        
        with col4:
            st.metric("Segment Duration", f"{segment_duration}s")
        
        # Segment Count by Stroke
        st.subheader("📊 Stroke Distribution")
        
        segment_counts = features_df["activity"].value_counts()
        segment_counts_display = pd.DataFrame({
            "Stroke": [stroke_names.get(act, act.replace("_", " ").title()) for act in segment_counts.index],
            "Segments": segment_counts.values,
            "Percentage": (segment_counts.values / segment_counts.sum() * 100).round(1)
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_segments = go.Figure(data=[
                go.Pie(
                    labels=segment_counts_display["Stroke"],
                    values=segment_counts_display["Segments"],
                    hole=0.3,
                    marker_colors=[plotly_stroke_colors.get(act, 'gray') for act in segment_counts.index],
                    textinfo='label+value+percent',
                    hovertemplate="<b>%{label}</b><br>Segments: %{value}<br>Percentage: %{percent}<extra></extra>"
                )
            ])
            
            fig_segments.update_layout(
                title="Segments per Stroke",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig_segments, use_container_width=True)
        
        with col2:
            # Feature Discriminability
            motion_features = ["acc_mean", "acc_rms", "acc_std", "acc_p2p", "gyro_mean", "gyro_rms", "gyro_std", "gyro_p2p"]
            shape_features = ["pitch_kurtosis", "pitch_skewness", "pitch_peak_count", "roll_asymmetry", 
                             "stroke_frequency", "stroke_rhythm_cv", "gyro_kurtosis", "gyro_skewness"]
            all_features = motion_features + shape_features
            
            feature_discriminability = []
            for feature in all_features:
                groups = [features_df[features_df["activity"] == act][feature].values
                          for act in strokes_from_features]
                groups = [g for g in groups if len(g) > 0]
                if len(groups) >= 2:
                    f_stat, _ = stats.f_oneway(*groups)
                    feature_discriminability.append({
                        "Feature": feature.replace('_', ' ').title(),
                        "F_Statistic": f_stat
                    })
            
            disc_df = pd.DataFrame(feature_discriminability)
            disc_df = disc_df.sort_values("F_Statistic", ascending=False)
            
            fig_disc = go.Figure(data=[
                go.Bar(
                    x=disc_df["F_Statistic"],
                    y=disc_df["Feature"],
                    orientation='h',
                    marker_color='lightblue',
                    hovertemplate="<b>%{y}</b><br>F-Statistic: %{x:.2f}<br><extra></extra>"
                )
            ])
            
            fig_disc.update_layout(
                title="Feature Discriminability (Higher = Better Separation)",
                xaxis_title="F-Statistic (ANOVA)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_disc, use_container_width=True)
        
        # ==========================
        # 3. FEATURE SCATTER PLOTS
        # ==========================
        from feature_relationships_swim import SwimmingFeatureRelationships
        feature_visualizer = SwimmingFeatureRelationships(
        features_df=features_df,
        stroke_colors=plotly_stroke_colors,
        stroke_names=stroke_names
                                    )
        feature_visualizer.render_dashboard()
        # ==========================
        # 4. FEATURE TABLE
        # ==========================
        st.subheader("📋 Feature Table (All Segments)")
        
        with st.expander("View Complete Feature Table", expanded=False):
            st.dataframe(features_df, use_container_width=True)
        
        # ==========================
        # 5. EXPORT FEATURES
        # ==========================
        st.subheader("💾 Export Features")
        
        csv_data = features_df.to_csv(index=False).encode("utf-8")
        
        st.download_button(
                label="📥 Download Features as CSV",
                data=csv_data,
                file_name="swimming_features_2s_windows.csv",
                mime="text/csv",
                key="download_csv"
            )
        
        # ==========================
        # 6. MOTION FEATURES EXPLORER
        # ==========================
        
    
        feature_dashboard = FeatureDescriptionDashboard(SWIMMING_FEATURE_INFO)
        feature_dashboard.render_dashboard(features_df)
        
        
        st.markdown("---")
        st.subheader("⚡ Motion Features Explorer")
        
        # Motion features for swimming (8 features)
        motion_features = [
            "acc_mean", "acc_rms", "acc_std", "acc_p2p",
            "gyro_mean", "gyro_rms", "gyro_std", "gyro_p2p"
        ]
        
        # Create tabs for motion features
        motion_tab1, motion_tab2, motion_tab3, motion_tab4 = st.tabs([
            "📈 Feature Trends", "🎯 Stroke Comparison", 
            "📊 Motion Energy Dashboard", "📋 Feature Statistics"
        ])
        
        with motion_tab1:
            selected_motion_feature = st.selectbox(
                "Select motion feature to explore:",
                motion_features,
                key="motion_feature_selector"
            )
            
            # Feature description
            motion_descriptions = {
                "acc_mean": "Average acceleration magnitude in the window",
                "acc_rms": "Root Mean Square - measures acceleration energy/power",
                "acc_std": "Standard deviation - measures acceleration variability",
                "acc_p2p": "Peak-to-peak - range from minimum to maximum acceleration",
                "gyro_mean": "Average gyroscope magnitude in the window",
                "gyro_rms": "Root Mean Square - measures rotation energy/power",
                "gyro_std": "Standard deviation - measures rotation variability",
                "gyro_p2p": "Peak-to-peak - range from minimum to maximum rotation"
            }
            
            st.info(
                f"**{selected_motion_feature.replace('_', ' ').title()}**: {motion_descriptions[selected_motion_feature]}"
            )
            
            # Line plot across time segments
            fig_m1 = go.Figure()
            
            for stroke in strokes_from_features:
                stroke_data = features_df[features_df["activity"] == stroke]
                if not stroke_data.empty:
                    display_name = stroke_names.get(stroke, stroke.replace("_", " ").title())
                    fig_m1.add_trace(go.Scatter(
                        x=stroke_data["segment_start"],
                        y=stroke_data[selected_motion_feature],
                        mode='lines+markers',
                        name=display_name,
                        line=dict(color=plotly_stroke_colors.get(stroke, 'gray'), width=2),
                        marker=dict(size=6),
                        hovertemplate='<b>%{text}</b><br>' +
                                      'Time: %{x:.1f}s<br>' +
                                      'Value: %{y:.3f}<extra></extra>',
                        text=[display_name] * len(stroke_data)
                    ))
            
            fig_m1.update_layout(
                title=f"{selected_motion_feature.replace('_', ' ').title()} Across Time Segments",
                xaxis_title="Segment Start Time (s)",
                yaxis_title=selected_motion_feature.replace('_', ' ').title(),
                height=400,
                hovermode="x unified",
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            st.plotly_chart(fig_m1, use_container_width=True)
        
        with motion_tab2:
            selected_motion_feature_t2 = st.selectbox(
                "Select motion feature for comparison:",
                motion_features,
                key="motion_feature_comparison"
            )
            
            # Create SCATTER plot for comparison
            fig_m2 = go.Figure()
            
            # Add scatter points for each stroke
            for stroke in strokes_from_features:
                stroke_data = features_df[features_df["activity"] == stroke]
                if not stroke_data.empty:
                    display_name = stroke_names.get(stroke, stroke.replace("_", " ").title())
                    fig_m2.add_trace(go.Scatter(
                        x=stroke_data.index,
                        y=stroke_data[selected_motion_feature_t2],
                        mode='markers',
                        name=display_name,
                        marker=dict(
                            size=8,
                            color=plotly_stroke_colors.get(stroke, 'gray'),
                            opacity=0.7,
                            line=dict(width=1, color='white')
                        ),
                        hovertemplate='<b>%{text}</b><br>' +
                                      'Segment Index: %{x}<br>' +
                                      'Value: %{y:.3f}<br>' +
                                      'Time: %{customdata:.1f}s<extra></extra>',
                        text=[display_name] * len(stroke_data),
                        customdata=stroke_data["segment_start"]
                    ))
            
            # Calculate statistics for summary box
            summary_data = {}
            for stroke in strokes_from_features:
                stroke_data = features_df[features_df["activity"] == stroke]
                if not stroke_data.empty:
                    values = stroke_data[selected_motion_feature_t2]
                    display_name = stroke_names.get(stroke, stroke.replace("_", " ").title())
                    summary_data[display_name] = {
                        "stroke": stroke,
                        "mean": values.mean(),
                        "std": values.std(),
                        "min": values.min(),
                        "max": values.max(),
                        "count": len(values)
                    }
            
            fig_m2.update_layout(
                title=f"{selected_motion_feature_t2.replace('_', ' ').title()} - Stroke Comparison",
                xaxis_title="Segment Index (Order in Dataset)",
                yaxis_title=selected_motion_feature_t2.replace('_', ' ').title(),
                height=500,
                hovermode="closest",
                showlegend=True
            )
            
            st.plotly_chart(fig_m2, use_container_width=True)
            
            # Summary Box
            st.subheader("📊 Comparison Summary")
            
            if summary_data:
                # Find which stroke has highest mean
                highest_stroke_name = max(summary_data.items(), key=lambda x: x[1]["mean"])
                lowest_stroke_name = min(summary_data.items(), key=lambda x: x[1]["mean"])
                
                highest_stroke = highest_stroke_name[0]
                lowest_stroke = lowest_stroke_name[0]
                
                # Calculate percentage difference
                if lowest_stroke_name[1]["mean"] > 0:
                    pct_diff = ((highest_stroke_name[1]["mean"] - lowest_stroke_name[1]["mean"]) /
                                lowest_stroke_name[1]["mean"]) * 100
                else:
                    pct_diff = 0
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Highest Average",
                        value=f"{highest_stroke}",
                        delta=f"{highest_stroke_name[1]['mean']:.3f}"
                    )
                
                with col2:
                    st.metric(
                        label="Lowest Average",
                        value=f"{lowest_stroke}",
                        delta=f"{lowest_stroke_name[1]['mean']:.3f}"
                    )
                
                with col3:
                    st.metric(
                        label="Difference",
                        value=f"{pct_diff:.1f}%",
                        delta=f"{highest_stroke_name[1]['mean'] - lowest_stroke_name[1]['mean']:.3f}"
                    )
                
                # Variability comparison
                st.subheader("📈 Variability Comparison")
                variability_data = []
                for display_name, stats in summary_data.items():
                    variability_data.append({
                        "Stroke": display_name,
                        "Samples": stats["count"],
                        "Average": f"{stats['mean']:.3f}",
                        "Std Dev": f"{stats['std']:.3f}",
                        "Min": f"{stats['min']:.3f}",
                        "Max": f"{stats['max']:.3f}",
                        "Range": f"{stats['max'] - stats['min']:.3f}"
                    })
                
                variability_df = pd.DataFrame(variability_data)
                st.dataframe(variability_df, use_container_width=True)
        
        with motion_tab3:
            # Motion Energy Dashboard
            
            # 1. Feature Correlation Matrix
            st.subheader("🔗 Motion Feature Correlations")
            
            corr_matrix = features_df[motion_features].corr().round(3)
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns.str.replace('_', ' ').str.title(),
                y=corr_matrix.columns.str.replace('_', ' ').str.title(),
                colorscale='RdBu',
                zmin=-1,
                zmax=1,
                text=corr_matrix.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig_corr.update_layout(
                title="Correlation Between Motion Features",
                height=400,
                xaxis_title="Features",
                yaxis_title="Features"
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # 2. Cumulative Energy Plot
            st.subheader("🔋 Cumulative Motion Energy")
            
            # Sort by time and calculate cumulative RMS
            sorted_df = features_df.sort_values("segment_start")
            sorted_df['cumulative_acc_rms'] = sorted_df['acc_rms'].cumsum()
            sorted_df['cumulative_gyro_rms'] = sorted_df['gyro_rms'].cumsum()
            
            fig_energy = go.Figure()
            
            # Add cumulative acceleration energy line
            fig_energy.add_trace(go.Scatter(
                x=sorted_df["segment_start"],
                y=sorted_df["cumulative_acc_rms"],
                mode='lines',
                name='Cumulative Acc RMS',
                line=dict(color='blue', width=3),
                fill='tozeroy',
                fillcolor='rgba(0, 0, 255, 0.1)'
            ))
            
            # Add cumulative gyroscope energy line
            fig_energy.add_trace(go.Scatter(
                x=sorted_df["segment_start"],
                y=sorted_df["cumulative_gyro_rms"],
                mode='lines',
                name='Cumulative Gyro RMS',
                line=dict(color='green', width=3),
                fill='tozeroy',
                fillcolor='rgba(0, 255, 0, 0.1)'
            ))
            
            fig_energy.update_layout(
                title="Cumulative Motion Energy Over Time",
                xaxis_title="Segment Start Time (s)",
                yaxis_title="Cumulative RMS",
                height=400,
                hovermode="x unified"
            )
            
            st.plotly_chart(fig_energy, use_container_width=True)
            
            # 3. Stroke Energy Summary
            st.subheader("⚡ Stroke Energy Comparison")
            
            # Calculate average RMS per stroke
            stroke_acc_rms = features_df.groupby("activity")["acc_rms"].agg(['mean', 'std', 'count']).round(3)
            stroke_acc_rms = stroke_acc_rms.reset_index()
            stroke_acc_rms['stroke_display'] = stroke_acc_rms['activity'].apply(
                lambda x: stroke_names.get(x, x.replace('_', ' ').title())
            )
            
            # Sort by mean RMS (highest to lowest)
            stroke_acc_rms = stroke_acc_rms.sort_values('mean', ascending=False)
            
            fig_bar = go.Figure()
            
            fig_bar.add_trace(go.Bar(
                x=stroke_acc_rms['stroke_display'],
                y=stroke_acc_rms['mean'],
                error_y=dict(
                    type='data',
                    array=stroke_acc_rms['std'],
                    visible=True
                ),
                marker_color=[plotly_stroke_colors.get(stroke, 'gray') for stroke in stroke_acc_rms['activity']],
                text=stroke_acc_rms['mean'].round(3),
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>' +
                              'Average Acc RMS: %{y:.3f}<br>' +
                              'Std: %{customdata:.3f}<br>' +
                              'Samples: %{text}<extra></extra>',
                customdata=stroke_acc_rms['std']
            ))
            
            fig_bar.update_layout(
                title="Average Acceleration Energy (RMS) by Stroke",
                xaxis_title="Stroke",
                yaxis_title="Average Acc RMS",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with motion_tab4:
            # Feature statistics table for motion features
            st.subheader("Motion Feature Statistics")
            
            motion_stats_data = []
            for feature in motion_features:
                stats = features_df[feature].describe()
                motion_stats_data.append({
                    "Feature": feature.replace('_', ' ').title(),
                    "Count": int(stats['count']),
                    "Mean": round(stats['mean'], 3),
                    "Std": round(stats['std'], 3),
                    "Min": round(stats['min'], 3),
                    "25%": round(stats['25%'], 3),
                    "Median": round(stats['50%'], 3),
                    "75%": round(stats['75%'], 3),
                    "Max": round(stats['max'], 3)
                })
            
            motion_stats_df = pd.DataFrame(motion_stats_data)
            st.dataframe(motion_stats_df, use_container_width=True)
            
            # Stroke-wise statistics for motion features
            st.subheader("Stroke-wise Motion Statistics")
            
            # Calculate statistics
            motion_stroke_stats = (
                features_df.groupby("activity")[motion_features]
                .agg(['mean', 'std'])
                .round(3)
            )
            
            # Flatten multi-level columns
            motion_stroke_stats.columns = [
                f"{feature}_{stat}"
                for feature, stat in motion_stroke_stats.columns
            ]
            
            # Create display version with formatted index
            motion_stroke_stats_display = motion_stroke_stats.copy()
            motion_stroke_stats_display.index = (
                motion_stroke_stats_display.index
                .map(lambda x: stroke_names.get(x, x.replace('_', ' ').title()))
            )
            
            # Display with better formatting
            st.dataframe(
                motion_stroke_stats_display,
                use_container_width=True
            )
        
        # ==========================
        # 7. SHAPE FEATURE EXPLORER
        # ==========================
        st.markdown("---")
        st.subheader("📊 Shape Feature Explorer")
        
        # Define shape features for swimming (8 features)
        shape_features = [
            "pitch_kurtosis", "pitch_skewness", "pitch_peak_count",
            "roll_asymmetry", "stroke_frequency", "stroke_rhythm_cv",
            "gyro_kurtosis", "gyro_skewness"
        ]
        
        # Create tabs for shape features
        shape_tab1, shape_tab2, shape_tab3 = st.tabs(["📈 Feature Trends", "📊 Distribution", "📋 Feature Table"])
        
        with shape_tab1:
            selected_shape_feature = st.selectbox(
                "Select shape feature to explore:",
                shape_features,
                key="shape_feature_selector"
            )
            
            # Feature description
            shape_descriptions = {
                "pitch_kurtosis": "Measures tailedness/sharpness of pitch distribution. High kurtosis = sharp peaks, heavy tails",
                "pitch_skewness": "Measures asymmetry of pitch movement. Positive = right-skewed, Negative = left-skewed",
                "pitch_peak_count": "Number of pitch peaks in the 2-second window",
                "roll_asymmetry": "Difference between left and right body roll. Positive = right bias, Negative = left bias",
                "stroke_frequency": "Estimated stroke cycles per second",
                "stroke_rhythm_cv": "Consistency of time between strokes. Lower = more regular rhythm",
                "gyro_kurtosis": "Measures tailedness/sharpness of rotation distribution",
                "gyro_skewness": "Measures asymmetry of rotation patterns"
            }
            
            st.info(
                f"**{selected_shape_feature.replace('_', ' ').title()}**: {shape_descriptions[selected_shape_feature]}"
            )
            
            # Line plot across time segments
            fig_shape1 = go.Figure()
            
            for stroke in strokes_from_features:
                stroke_data = features_df[features_df["activity"] == stroke]
                if not stroke_data.empty:
                    display_name = stroke_names.get(stroke, stroke.replace("_", " ").title())
                    fig_shape1.add_trace(go.Scatter(
                        x=stroke_data["segment_start"],
                        y=stroke_data[selected_shape_feature],
                        mode='lines+markers',
                        name=display_name,
                        line=dict(color=plotly_stroke_colors.get(stroke, 'gray'), width=2),
                        marker=dict(size=6)
                    ))
            
            fig_shape1.update_layout(
                title=f"{selected_shape_feature.replace('_', ' ').title()} Across Time Segments",
                xaxis_title="Segment Start Time (s)",
                yaxis_title=selected_shape_feature.replace('_', ' ').title(),
                height=400,
                hovermode="x unified",
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            st.plotly_chart(fig_shape1, use_container_width=True)
        
        with shape_tab2:
            # Boxplot by stroke
            fig_shape2 = go.Figure()
            
            for stroke in strokes_from_features:
                stroke_data = features_df[features_df["activity"] == stroke]
                if not stroke_data.empty:
                    display_name = stroke_names.get(stroke, stroke.replace("_", " ").title())
                    fig_shape2.add_trace(go.Box(
                        y=stroke_data[selected_shape_feature],
                        name=display_name,
                        marker_color=plotly_stroke_colors.get(stroke, 'gray'),
                        boxmean=True
                    ))
            
            fig_shape2.update_layout(
                title=f"Distribution of {selected_shape_feature.replace('_', ' ').title()} by Stroke",
                yaxis_title=selected_shape_feature.replace('_', ' ').title(),
                xaxis_title="Stroke",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_shape2, use_container_width=True)
            
            # Add histogram view
            col1, col2 = st.columns(2)
            
            with col1:
                fig_shape3 = px.histogram(
                    features_df,
                    x=selected_shape_feature,
                    nbins=20,
                    title=f"Overall Distribution",
                    labels={selected_shape_feature: selected_shape_feature.replace('_', ' ').title()},
                    height=350
                )
                st.plotly_chart(fig_shape3, use_container_width=True)
            
            with col2:
                fig_shape4 = px.histogram(
                    features_df,
                    x=selected_shape_feature,
                    color="activity",
                    barmode="overlay",
                    nbins=20,
                    title=f"Distribution by Stroke",
                    color_discrete_map=plotly_stroke_colors,
                    labels={
                        selected_shape_feature: selected_shape_feature.replace('_', ' ').title(),
                        "activity": "Stroke"
                    },
                    height=350
                )
                st.plotly_chart(fig_shape4, use_container_width=True)
        
        with shape_tab3:
            # Feature statistics table
            st.subheader("Shape Feature Statistics")
            
            shape_stats_data = []
            for feature in shape_features:
                stats = features_df[feature].describe()
                shape_stats_data.append({
                    "Feature": feature.replace('_', ' ').title(),
                    "Count": int(stats['count']),
                    "Mean": round(stats['mean'], 3),
                    "Std": round(stats['std'], 3),
                    "Min": round(stats['min'], 3),
                    "25%": round(stats['25%'], 3),
                    "Median": round(stats['50%'], 3),
                    "75%": round(stats['75%'], 3),
                    "Max": round(stats['max'], 3)
                })
            
            shape_stats_df = pd.DataFrame(shape_stats_data)
            st.dataframe(shape_stats_df, use_container_width=True)
            
            # Stroke-wise statistics function
            def display_stroke_statistics(df, features, title, stroke_names_dict=None):
                """Display formatted stroke statistics for given features"""
                st.subheader(title)
                
                # Calculate statistics
                stats = (
                    df.groupby("activity")[features]
                    .agg(['mean', 'std'])
                    .round(3)
                )
                
                # Flatten columns
                stats.columns = [f"{feat}_{stat}" for feat, stat in stats.columns]
                
                # Format stroke names
                if stroke_names_dict:
                    stats.index = stats.index.map(
                        lambda x: stroke_names_dict.get(x, x.replace('_', ' ').title())
                    )
                
                # Display
                st.dataframe(stats, use_container_width=True)
                
                return stats
            
            # Usage
            motion_stats = display_stroke_statistics(
                features_df, 
                motion_features, 
                "Stroke-wise Motion Statistics",
                stroke_names
            )
            
            shape_stats = display_stroke_statistics(
                features_df,
                shape_features,
                "Stroke-wise Shape Statistics",
                stroke_names
            )
            
            st.success("✅ Feature extraction complete! Ready for machine learning.")
        
        # ==========================
        # 8. MACHINE LEARNING COMPARISON
        # ==========================
        st.markdown("---")
        st.header("🤖 Step 3: Machine Learning Algorithm Comparison")
        
        st.info(f"""
        **Algorithm Configuration:**
        - **Test Set:** {test_size * 100:.0f}% of data
        - **KNN:** k={k_value} neighbors, 8 motion features
        - **Random Forest:** {n_trees} trees, all 16 features (8 motion + 8 shape)
        - **Strokes:** {', '.join([stroke_names.get(a, a.replace('_', ' ').title()) for a in strokes_from_features])}
        """)
        
        if st.button("🚀 Run KNN vs Random Forest Comparison", type="primary", key="run_algorithms"):
            with st.spinner("Training and evaluating models..."):
                
                # Clear previous results
                st.session_state.knn_results = None
                st.session_state.rf_results = None
                
                try:
                    # Run KNN pipeline
                    with st.spinner("Training KNN..."):
                        knn_results = run_knn_pipeline(
                            features_df=features_df,
                            test_size=test_size,
                            n_neighbors=k_value,
                            random_state=42
                        )
                        st.session_state.knn_results = knn_results
                    
                    # Run Random Forest pipeline
                    with st.spinner("Training Random Forest..."):
                        rf_results = run_rf_pipeline(
                            features_df=features_df,
                            test_size=test_size,
                            n_estimators=n_trees,
                            random_state=42
                        )
                        st.session_state.rf_results = rf_results
                    
                    st.success("✅ Both models trained successfully!")
                
                except Exception as e:
                    st.error(f"❌ Error during model training: {str(e)}")
                    st.stop()
        
        # ==========================
        # 9. DISPLAY ML RESULTS
        # ==========================
        if st.session_state.knn_results is not None and st.session_state.rf_results is not None:
            knn_results = st.session_state.knn_results
            rf_results = st.session_state.rf_results
            
            st.markdown("---")
            st.header("📊 Model Performance Results")
            
            # Performance Metrics Comparison
            st.subheader("🎯 Performance Metrics Comparison")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                delta_knn = knn_results['metrics']['accuracy'] - rf_results['metrics']['accuracy']
                st.metric(
                    "KNN Accuracy",
                    f"{knn_results['metrics']['accuracy']:.3f}",
                    delta=f"{delta_knn:+.3f}" if delta_knn != 0 else None,
                    delta_color="inverse" if delta_knn < 0 else "normal"
                )
            
            with col2:
                delta_rf = rf_results['metrics']['accuracy'] - knn_results['metrics']['accuracy']
                st.metric(
                    "Random Forest Accuracy",
                    f"{rf_results['metrics']['accuracy']:.3f}",
                    delta=f"{delta_rf:+.3f}" if delta_rf != 0 else None,
                    delta_color="inverse" if delta_rf < 0 else "normal"
                )
            
            with col3:
                st.metric(
                    "KNN F1 Score",
                    f"{knn_results['metrics']['f1_score']:.3f}",
                    help="Harmonic mean of precision and recall"
                )
            
            with col4:
                st.metric(
                    "Random Forest F1 Score",
                    f"{rf_results['metrics']['f1_score']:.3f}",
                    help="Harmonic mean of precision and recall"
                )
            
            # Determine winner
            knn_acc = knn_results['metrics']['accuracy']
            rf_acc = rf_results['metrics']['accuracy']
            
            if rf_acc > knn_acc:
                winner = "🎖️ Random Forest"
                improvement = f"{(rf_acc - knn_acc) * 100:.1f}%"
                recommendation = f"**Use Random Forest** - it's {improvement} more accurate!"
                winner_color = "🟢"
            elif knn_acc > rf_acc:
                winner = "🎖️ KNN"
                improvement = f"{(knn_acc - rf_acc) * 100:.1f}%"
                recommendation = f"**Use KNN** - it's {improvement} more accurate and faster!"
                winner_color = "🔵"
            else:
                winner = "🤝 Tie"
                recommendation = "Both algorithms perform similarly. Choose KNN for speed or Random Forest for interpretability."
                winner_color = "🟡"
            
            # Winner announcement
            st.info(f"""
            {winner_color} **Winner:** {winner}
            
            **Recommendation:** {recommendation}
            
            **Key Differences:**
            - **KNN Training Time:** {knn_results['metrics']['training_time']:.3f}s
            - **RF Training Time:** {rf_results['metrics']['training_time']:.3f}s
            - **KNN Features:** 8 motion features
            - **RF Features:** 16 features (8 motion + 8 shape)
            - **Number of Strokes:** {len(strokes_from_features)}
            """)
            
            # ==========================
            # VISUALIZATIONS TABS
            # ==========================
            st.subheader("📈 Visualizations")

            tab1, tab2, tab3, tab4 = st.tabs(
                ["📊 Metrics", "🎭 Confusion Matrices", "🎯 Feature Importance", "⏱️ Performance"])

            with tab1:
                col1, col2 = st.columns(2)

                with col1:
                    # Update metric bar chart with correct labels
                    knn_metrics_fig = knn_results['figures']['metrics_bar']
                    # Update stroke labels if needed
                    if hasattr(knn_metrics_fig, 'data'):
                        st.plotly_chart(knn_metrics_fig, use_container_width=True)
                    st.caption("KNN Performance Metrics")

                with col2:
                    rf_metrics_fig = rf_results['figures']['metrics_bar']
                    # Update stroke labels if needed
                    if hasattr(rf_metrics_fig, 'data'):
                        st.plotly_chart(rf_metrics_fig, use_container_width=True)
                    st.caption("Random Forest Performance Metrics")

            with tab2:
                col1, col2 = st.columns(2)

                with col1:
                    knn_cm_fig = knn_results['figures']['confusion_matrix']
                    # Update confusion matrix labels
                    if hasattr(knn_cm_fig, 'layout'):
                        # Update x and y axis labels with display names
                        updated_labels = []
                        for label in knn_cm_fig.layout.xaxis.ticktext or []:
                            display_label = stroke_names.get(label, label.replace('_', ' ').title())
                            updated_labels.append(display_label)

                        if updated_labels:
                            knn_cm_fig.update_xaxes(ticktext=updated_labels, tickvals=list(range(len(updated_labels))))
                            knn_cm_fig.update_yaxes(ticktext=updated_labels, tickvals=list(range(len(updated_labels))))

                    st.plotly_chart(knn_cm_fig, use_container_width=True)
                    st.caption("KNN Confusion Matrix")

                with col2:
                    rf_cm_fig = rf_results['figures']['confusion_matrix']
                    # Update confusion matrix labels
                    if hasattr(rf_cm_fig, 'layout'):
                        # Update x and y axis labels with display names
                        updated_labels = []
                        for label in rf_cm_fig.layout.xaxis.ticktext or []:
                            display_label = stroke_names.get(label, label.replace('_', ' ').title())
                            updated_labels.append(display_label)

                        if updated_labels:
                            rf_cm_fig.update_xaxes(ticktext=updated_labels, tickvals=list(range(len(updated_labels))))
                            rf_cm_fig.update_yaxes(ticktext=updated_labels, tickvals=list(range(len(updated_labels))))

                    st.plotly_chart(rf_cm_fig, use_container_width=True)
                    st.caption("Random Forest Confusion Matrix")

            with tab3:
                if 'feature_importance' in rf_results['figures']:
                    rf_fi_fig = rf_results['figures']['feature_importance']
                    st.plotly_chart(rf_fi_fig, use_container_width=True)
                    st.caption("Random Forest Feature Importance")

                # Show top features
                if 'feature_importance' in rf_results['metrics']:
                    feature_importance = rf_results['metrics']['feature_importance']
                    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]

                    st.subheader("🏆 Top 5 Most Important Features")
                    for i, (feature, importance) in enumerate(top_features, 1):
                        feature_name = feature.replace("_", " ").title()
                        st.progress(float(importance), text=f"{i}. {feature_name}: {importance:.3f}")

            with tab4:
                col1, col2 = st.columns(2)

                with col1:
                    # Timing comparison
                    timing_data = pd.DataFrame({
                        'Metric': ['Training Time', 'Prediction Time (20 samples)'],
                        'KNN': [
                            knn_results['metrics']['training_time'],
                            knn_results['metrics']['prediction_time']
                        ],
                        'Random Forest': [
                            rf_results['metrics']['training_time'],
                            rf_results['metrics']['prediction_time']
                        ]
                    })

                    fig_timing = px.bar(
                        timing_data.melt(id_vars='Metric'),
                        x='Metric',
                        y='value',
                        color='variable',
                        barmode='group',
                        title='Computational Performance',
                        labels={'value': 'Time (seconds)', 'variable': 'Algorithm'}
                    )
                    st.plotly_chart(fig_timing, use_container_width=True)

                with col2:
                    # Cross-validation results
                    cv_data = pd.DataFrame({
                        'Algorithm': ['KNN', 'Random Forest'],
                        'Mean CV Accuracy': [
                            knn_results['cv_results']['mean_score'],
                            rf_results['cv_results']['mean_score']
                        ],
                        'Std Deviation': [
                            knn_results['cv_results']['std_score'],
                            rf_results['cv_results']['std_score']
                        ]
                    })

                    fig_cv = px.bar(
                        cv_data,
                        x='Algorithm',
                        y='Mean CV Accuracy',
                        error_y='Std Deviation',
                        title='Cross-Validation Results',
                        labels={'Mean CV Accuracy': 'Mean Accuracy', 'Algorithm': ''},
                        color='Algorithm',
                        color_discrete_map={'KNN': 'blue', 'Random Forest': 'green'}
                    )
                    fig_cv.update_layout(yaxis=dict(range=[0, 1]))
                    st.plotly_chart(fig_cv, use_container_width=True)

            # ============================================
            # PERFORMANCE SUMMARY CARD
            # ============================================
            # Calculate values needed
            total = len(features_df)
            train_count = len(knn_results['data_splits']['X_train'])
            test_count = len(knn_results['data_splits']['X_test'])
            train_pct = (train_count / total) * 100
            test_pct = (test_count / total) * 100

            st.markdown("---")
            st.subheader("🏆 Performance Summary")

            # Get metrics
            knn_acc = knn_results['metrics']['accuracy']
            rf_acc = rf_results['metrics']['accuracy']
            knn_f1 = knn_results['metrics']['f1_score']
            rf_f1 = rf_results['metrics']['f1_score']
            knn_time = knn_results['metrics']['training_time']
            rf_time = rf_results['metrics']['training_time']
            knn_pred_time = knn_results['metrics']['prediction_time']
            rf_pred_time = rf_results['metrics']['prediction_time']

            # Create summary table
            summary_data = {
                'Metric': ['🎯 Accuracy', '⚖️ F1 Score', '⏱️ Training Time', '⚡ Prediction Time', '📊 Features Used',
                           '🏊‍♂️ Strokes'],
                'KNN': [
                    f"{knn_acc:.3f}",
                    f"{knn_f1:.3f}",
                    f"{knn_time:.3f}s",
                    f"{knn_pred_time:.3f}s",
                    "8 motion features",
                    f"{len(strokes_from_features)} strokes"
                ],
                'Random Forest': [
                    f"{rf_acc:.3f}",
                    f"{rf_f1:.3f}",
                    f"{rf_time:.3f}s",
                    f"{rf_pred_time:.3f}s",
                    "16 features (motion + shape)",
                    f"{len(strokes_from_features)} strokes"
                ],
                'Winner': [
                    '✅ KNN' if knn_acc > rf_acc else ('✅ RF' if rf_acc > knn_acc else '🤝 Tie'),
                    '✅ KNN' if knn_f1 > rf_f1 else ('✅ RF' if rf_f1 > knn_f1 else '🤝 Tie'),
                    '✅ KNN' if knn_time < rf_time else ('✅ RF' if rf_time < knn_time else '🤝 Tie'),
                    '✅ KNN' if knn_pred_time < rf_pred_time else ('✅ RF' if rf_pred_time < knn_pred_time else '🤝 Tie'),
                    '✅ RF' if rf_acc > knn_acc else '✅ KNN',
                    '🤝 Same'
                ]
            }

            summary_df = pd.DataFrame(summary_data)

            # Display with nice formatting
            st.dataframe(
                summary_df,
                column_config={
                    "Metric": st.column_config.TextColumn("Metric", width="medium"),
                    "KNN": st.column_config.TextColumn("KNN", width="medium"),
                    "Random Forest": st.column_config.TextColumn("Random Forest", width="medium"),
                    "Winner": st.column_config.TextColumn("Winner", width="small")
                },
                hide_index=True,
                use_container_width=True
            )

            # Performance insights
            st.info(f"""
            **📈 Key Insights:**

            1. **Accuracy Difference:** {abs(knn_acc - rf_acc) * 100:.1f}%
               - {'KNN is better for this dataset' if knn_acc > rf_acc else 'Random Forest is better for this dataset'}

            2. **Speed Comparison:**
               - Training: {('KNN is' if knn_time < rf_time else 'Random Forest is')} {abs(knn_time - rf_time):.2f}s {'faster' if knn_time < rf_time else 'slower'}
               - Prediction: {('KNN is' if knn_pred_time < rf_pred_time else 'Random Forest is')} {abs(knn_pred_time - rf_pred_time):.3f}s {'faster' if knn_pred_time < rf_pred_time else 'slower'}

            3. **Stroke Analysis:** You're classifying {len(strokes_from_features)} different swimming strokes
            4. **Recommendation:** {'**Use KNN** for real-time applications' if knn_pred_time < rf_pred_time else '**Use Random Forest** for batch processing'}
            """)

            # ============================================
            # CONFUSION ANALYSIS
            # ============================================
            st.markdown("---")
            st.subheader("🎭 Confusion Analysis")

            col1, col2 = st.columns(2)

            with col1:
                # KNN Confusion Details
                knn_cm = knn_results['metrics']['confusion_matrix']
                knn_true = knn_results['metrics']['true_labels']
                knn_pred = knn_results['metrics']['predictions']

                # Convert labels to display names
                knn_true_display = [stroke_names.get(label, label.replace('_', ' ').title()) for label in knn_true]
                knn_pred_display = [stroke_names.get(label, label.replace('_', ' ').title()) for label in knn_pred]

                # Create detailed confusion DataFrame
                knn_confusion_details = pd.crosstab(
                    pd.Series(knn_true_display, name='True Stroke'),
                    pd.Series(knn_pred_display, name='Predicted Stroke'),
                    margins=True
                )

                st.markdown("##### KNN Confusion Details")
                st.dataframe(knn_confusion_details.style.background_gradient(cmap='Blues'),
                             use_container_width=True)

                # KNN misclassification analysis
                knn_misclassified = knn_true != knn_pred
                if sum(knn_misclassified) > 0:
                    knn_errors = pd.DataFrame({
                        'True': [stroke_names.get(label, label.replace('_', ' ').title()) for label in
                                 knn_true[knn_misclassified]],
                        'Predicted': [stroke_names.get(label, label.replace('_', ' ').title()) for label in
                                      knn_pred[knn_misclassified]],
                        'Count': 1
                    }).groupby(['True', 'Predicted']).count().reset_index()

                    st.markdown("**Most Common KNN Errors:**")
                    for _, row in knn_errors.sort_values('Count', ascending=False).head(3).iterrows():
                        st.write(f"- {row['True']} → {row['Predicted']}: {row['Count']} times")

            with col2:
                # Random Forest Confusion Details
                rf_cm = rf_results['metrics']['confusion_matrix']
                rf_true = rf_results['metrics']['true_labels']
                rf_pred = rf_results['metrics']['predictions']

                # Convert labels to display names
                rf_true_display = [stroke_names.get(label, label.replace('_', ' ').title()) for label in rf_true]
                rf_pred_display = [stroke_names.get(label, label.replace('_', ' ').title()) for label in rf_pred]

                # Create detailed confusion DataFrame
                rf_confusion_details = pd.crosstab(
                    pd.Series(rf_true_display, name='True Stroke'),
                    pd.Series(rf_pred_display, name='Predicted Stroke'),
                    margins=True
                )

                st.markdown("##### Random Forest Confusion Details")
                st.dataframe(rf_confusion_details.style.background_gradient(cmap='Greens'),
                             use_container_width=True)

                # RF misclassification analysis
                rf_misclassified = rf_true != rf_pred
                if sum(rf_misclassified) > 0:
                    rf_errors = pd.DataFrame({
                        'True': [stroke_names.get(label, label.replace('_', ' ').title()) for label in
                                 rf_true[rf_misclassified]],
                        'Predicted': [stroke_names.get(label, label.replace('_', ' ').title()) for label in
                                      rf_pred[rf_misclassified]],
                        'Count': 1
                    }).groupby(['True', 'Predicted']).count().reset_index()

                    st.markdown("**Most Common Random Forest Errors:**")
                    for _, row in rf_errors.sort_values('Count', ascending=False).head(3).iterrows():
                        st.write(f"- {row['True']} → {row['Predicted']}: {row['Count']} times")

            # Overall confusion statistics
            st.markdown("---")
            st.subheader("📊 Confusion Statistics")

            col1, col2, col3 = st.columns(3)

            with col1:
                knn_error_rate = sum(knn_misclassified) / len(knn_true) if len(knn_true) > 0 else 0
                st.metric("KNN Error Rate", f"{knn_error_rate:.1%}",
                          delta=f"{-knn_error_rate * 100:.1f}%" if knn_error_rate < 0.5 else None,
                          delta_color="inverse")

            with col2:
                rf_error_rate = sum(rf_misclassified) / len(rf_true) if len(rf_true) > 0 else 0
                st.metric("Random Forest Error Rate", f"{rf_error_rate:.1%}",
                          delta=f"{-rf_error_rate * 100:.1f}%" if rf_error_rate < 0.5 else None,
                          delta_color="inverse")

            with col3:
                improvement = (rf_error_rate - knn_error_rate) / knn_error_rate * 100 if knn_error_rate > 0 else 0
                st.metric("Relative Improvement",
                          f"{abs(improvement):.1f}%",
                          delta=f"{'KNN' if knn_error_rate < rf_error_rate else 'RF'} makes fewer errors",
                          delta_color="normal" if knn_error_rate < rf_error_rate else "inverse")

            # Most confused strokes
            st.markdown("#### 🎯 Most Challenging Strokes")
            stroke_errors = []

            for stroke in strokes_from_features:
                display_name = stroke_names.get(stroke, stroke.replace('_', ' ').title())
                knn_stroke_error = sum((knn_true == stroke) & knn_misclassified) / sum(knn_true == stroke) if sum(
                    knn_true == stroke) > 0 else 0
                rf_stroke_error = sum((rf_true == stroke) & rf_misclassified) / sum(rf_true == stroke) if sum(
                    rf_true == stroke) > 0 else 0
                stroke_errors.append({
                    'Stroke': display_name,
                    'KNN Error Rate': knn_stroke_error,
                    'RF Error Rate': rf_stroke_error,
                    'Total Samples': sum(knn_true == stroke)
                })

            error_df = pd.DataFrame(stroke_errors)
            st.dataframe(error_df.sort_values('KNN Error Rate', ascending=False),
                         use_container_width=True)

            # ============================================
            # TEST SIZE IMPACT ANALYSIS
            # ============================================
            with st.expander("🔬 Test Size Impact Analysis", expanded=False):
                st.markdown("""
                ### 📊 How Test Size Affects Results

                This analysis shows how different test/train splits would impact your results.
                Smaller test size = more training data but less reliable evaluation.
                """)

                # Simulate different test sizes
                test_sizes = [0.1, 0.2, 0.3, 0.4]
                knn_simulated_scores = []
                rf_simulated_scores = []

                # Use current model to simulate (simplified approach)
                for ts in test_sizes:
                    # Simplified simulation based on current results
                    # In reality, you'd retrain with different splits
                    base_knn_acc = knn_results['cv_results']['mean_score']
                    base_rf_acc = rf_results['cv_results']['mean_score']

                    # Simulate effect: more training data usually helps, but less test data is noisier
                    # This is a simplified educational simulation
                    train_effect = (1 - ts) / 0.8  # Normalize to current 80% train
                    test_noise = ts / 0.2  # Normalize to current 20% test

                    simulated_knn = base_knn_acc * (0.7 + 0.3 * train_effect) / (1 + 0.1 * test_noise)
                    simulated_rf = base_rf_acc * (0.7 + 0.3 * train_effect) / (1 + 0.1 * test_noise)

                    knn_simulated_scores.append(min(simulated_knn, 0.99))
                    rf_simulated_scores.append(min(simulated_rf, 0.99))

                # Create visualization
                fig_test_size = go.Figure()

                fig_test_size.add_trace(go.Scatter(
                    x=[f"{int(ts * 100)}% Test\n{int((1 - ts) * 100)}% Train" for ts in test_sizes],
                    y=knn_simulated_scores,
                    mode='lines+markers',
                    name='KNN (Simulated)',
                    line=dict(color='blue', width=2),
                    marker=dict(size=8)
                ))

                fig_test_size.add_trace(go.Scatter(
                    x=[f"{int(ts * 100)}% Test\n{int((1 - ts) * 100)}% Train" for ts in test_sizes],
                    y=rf_simulated_scores,
                    mode='lines+markers',
                    name='Random Forest (Simulated)',
                    line=dict(color='green', width=2),
                    marker=dict(size=8)
                ))

                # Highlight current test size
                current_idx = test_sizes.index(test_size)
                fig_test_size.add_vline(
                    x=current_idx,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Your choice: {test_size * 100}% Test",
                    annotation_position="top right"
                )

                fig_test_size.update_layout(
                    title='Simulated Impact of Test Size on Accuracy',
                    yaxis_title='Estimated Accuracy',
                    yaxis=dict(tickformat='.0%', range=[0.5, 1.0]),
                    xaxis_title='Test Size / Train Size',
                    height=400,
                    hovermode="x unified"
                )

                st.plotly_chart(fig_test_size, use_container_width=True)

                # Recommendations based on dataset size
                st.markdown("#### 💡 Recommendations")

                if len(features_df) < 100:
                    st.warning(f"""
                    **Small Dataset Alert:** You have only **{len(features_df)}** segments.

                    **Recommendation:** Use **30-40% test size** for more reliable evaluation,
                    even though it means less training data. With small datasets, reliable
                    evaluation is more important than maximum training data.
                    """)
                elif len(features_df) < 300:
                    st.info(f"""
                    **Medium Dataset:** You have **{len(features_df)}** segments.

                    **Recommendation:** **20-30% test size** is ideal. You have enough data
                    for both decent training and reliable testing.
                    """)
                else:
                    st.success(f"""
                    **Large Dataset:** You have **{len(features_df)}** segments.

                    **Recommendation:** **10-20% test size** works well. You can afford to
                    use more data for training while still having a large enough test set.
                    """)

                st.markdown(f"""
                **📚 Rule of Thumb:**
                - **Test size:** At least 20-30 samples per stroke class
                - **Training size:** At least 50-100 samples per stroke class
                - **Your current split:** {test_count} test samples, {train_count} training samples
                - **Samples per stroke:** {len(features_df) // len(strokes_from_features)} average
                """)

            # ============================================
            # DATA SPLIT EXPLANATION (ALL 3 LEVELS)
            # ============================================
            # --- LEVEL 1: Visual Summary ---
            st.markdown("---")
            st.subheader("📊 Data Split & Methodology")

            col1, col2 = st.columns([2, 1])

            with col1:
                # Visual split diagram
                fig_split = go.Figure(data=[
                    go.Indicator(
                        mode="gauge+number",
                        value=train_pct,
                        title={'text': f"Training Data ({train_count} segments)"},
                        domain={'x': [0, 0.48], 'y': [0, 1]},
                        gauge={'axis': {'range': [0, 100]},
                               'bar': {'color': "blue"},
                               'steps': [
                                   {'range': [0, train_pct], 'color': "lightblue"},
                                   {'range': [train_pct, 100], 'color': "lightgray"}]
                               }
                    ),
                    go.Indicator(
                        mode="gauge+number",
                        value=test_pct,
                        title={'text': f"Test Data ({test_count} segments)"},
                        domain={'x': [0.52, 1], 'y': [0, 1]},
                        gauge={'axis': {'range': [0, 100]},
                               'bar': {'color': "red"},
                               'steps': [
                                   {'range': [0, test_pct], 'color': "lightcoral"},
                                   {'range': [test_pct, 100], 'color': "lightgray"}]
                               }
                    )
                ])

                fig_split.update_layout(height=250)
                st.plotly_chart(fig_split, use_container_width=True)

            with col2:
                st.info(f"""
                **Your Settings:**
                - Test Size: **{test_size * 100}%**
                - Total Segments: **{total}**
                - Strokes: **{len(strokes_from_features)}**

                ⚠️ **Key Principle:**
                Test data was **NEVER seen** during training!
                """)

            # --- LEVEL 2: Interactive Flow Diagram ---
            st.markdown("---")
            st.subheader("🔍 How We Ensure Fair Testing")

            # Create an interactive flowchart
            flow_data = pd.DataFrame({
                'Step': ['Your CSV Data', 'Split', 'Training Set', 'Test Set', 'Results'],
                'Description': [
                    f'{total} stroke segments',
                    f'Your choice: {test_size * 100}% for testing',
                    f'{train_count} segments for learning',
                    f'{test_count} segments kept secret',
                    'Unbiased performance metrics'
                ],
                'Icon': ['📁', '✂️', '🎓', '🔒', '📈'],
                'Color': ['gray', 'orange', 'blue', 'red', 'green']
            })

            fig_flow = go.Figure(data=[
                go.Scatter(
                    x=flow_data['Step'],
                    y=[1] * len(flow_data),
                    mode='markers+text+lines',
                    marker=dict(size=50, color=flow_data['Color']),
                    text=flow_data['Icon'],
                    textposition="middle center",
                    textfont=dict(size=20),
                    hovertemplate='<b>%{x}</b><br>%{customdata}<extra></extra>',
                    customdata=flow_data['Description']
                )
            ])

            fig_flow.update_layout(
                height=200,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=True),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white'
            )

            st.plotly_chart(fig_flow, use_container_width=True)

            # Add timeline explanation below
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                **🎓 Training Phase**
                - Model learns patterns
                - Cross-validation practice
                - Hyperparameter tuning
                """)
            with col2:
                st.markdown("""
                **🔒 Test Phase**
                - Data kept secret
                - One-time evaluation
                - Real-world simulation
                """)
            with col3:
                st.markdown("""
                **✅ Why This Matters**
                - No data leakage
                - Realistic accuracy
                - Trustworthy results
                """)

            # --- LEVEL 3: Detailed Educational Section ---
            with st.expander("📚 Learn More: Data Splitting in ML", expanded=False):
                tab1, tab2, tab3 = st.tabs(["Why Split?", "Cross-Validation", "Best Practices"])

                with tab1:
                    st.markdown(f"""
                    ### 🎯 The 80/20 Rule in Machine Learning

                    **Analogy:** Imagine preparing for a swimming competition:
                    - **Training ({train_pct:.0f}%):** Practice strokes you train
                    - **Testing ({test_pct:.0f}%):** Competition strokes you've never practiced

                    **Without splitting:**
                    > "I memorized all {total} practice strokes and got 100% on the same strokes!"
                    → **Useless** - doesn't test real skill

                    **With splitting:**
                    > "I trained {train_count} strokes, then correctly performed new strokes!"
                    → **{knn_results['metrics']['accuracy']:.1%} real skill** that works on new strokes
                    """)

                    # Simple analogy visualization
                    analogy_df = pd.DataFrame({
                        'Scenario': ['Cheating (No Split)', f'Fair Test ({test_size * 100}% Split)'],
                        'Accuracy': [100, knn_results['metrics']['accuracy'] * 100],
                        'Description': [
                            f'Tests on same {total} segments studied → Overly optimistic',
                            f'Tests on {test_count} new segments → Real-world performance'
                        ]
                    })

                    fig_analogy = px.bar(
                        analogy_df,
                        x='Scenario',
                        y='Accuracy',
                        color='Scenario',
                        text='Accuracy',
                        title='Real vs Fake Accuracy',
                        hover_data=['Description']
                    )
                    fig_analogy.update_layout(yaxis_title="Accuracy (%)")
                    st.plotly_chart(fig_analogy, use_container_width=True)

                with tab2:
                    st.markdown(f"""
                    ### 🔄 Cross-Validation: Practice Tests

                    **Process:**
                    1. Take your **training data only** ({train_count} segments)
                    2. Split it into 5 folds (~{train_count // 5} segments each)
                    3. Train on 4 folds, test on the 5th
                    4. Repeat 5 times with different test folds
                    5. Average the 5 scores

                    **Your Results:**
                    - KNN CV Accuracy: **{knn_results['cv_results']['mean_score']:.3f}**
                    - Random Forest CV Accuracy: **{rf_results['cv_results']['mean_score']:.3f}**

                    **Benefits:**
                    - More reliable than single split
                    - Uses all training data for both training and validation
                    - Better estimate of model performance
                    """)

                    # CV visualization
                    cv_fig = go.Figure()

                    # Add CV fold visualization
                    folds = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']

                    cv_fig.add_trace(go.Bar(
                        x=folds,
                        y=[80, 80, 80, 80, 80],  # Training percentage
                        name='Training',
                        marker_color='blue',
                        text=['Train'] * 5
                    ))

                    cv_fig.add_trace(go.Bar(
                        x=folds,
                        y=[20, 20, 20, 20, 20],  # Validation percentage
                        name='Validation',
                        marker_color='orange',
                        text=['Val'] * 5
                    ))

                    cv_fig.update_layout(
                        title=f'5-Fold Cross-Validation (On {train_count} Training Segments Only)',
                        barmode='stack',
                        height=300,
                        yaxis_title="Percentage",
                        yaxis=dict(range=[0, 100])
                    )
                    st.plotly_chart(cv_fig, use_container_width=True)

                with tab3:
                    st.markdown(f"""
                    ### ⚖️ Choosing Your Test Size

                    **{test_size * 100}% Test Size Analysis:**

                    | Size | Pros | Cons | When to Use |
                    |------|------|------|-------------|
                    | **10-20%** | More training data | Less reliable test | Large datasets |
                    | **20-30%** | Good balance | Moderate | Most projects |
                    | **30-40%** | Very reliable test | Less training | Small datasets |

                    **Your choice ({test_size * 100}%):**
                    - Training segments: **{train_count}**
                    - Test segments: **{test_count}**
                    - Ratio: **{train_count}:{test_count}**
                    - Strokes: **{len(strokes_from_features)}**

                    **Rule of thumb:** At least 20 samples per stroke class in test set
                    """)

            # ==========================
            # MODEL EXPORT SECTION
            # ==========================
            st.markdown("---")
            st.subheader("💾 Export Trained Models")

            col1, col2 = st.columns(2)

            with col1:
                # KNN Model Export
                st.markdown("#### 🤖 KNN Model")
                st.info(f"""
                **Model Details:**
                - Accuracy: {knn_results['metrics']['accuracy']:.3f}
                - Features: 8 motion features
                - Strokes: {len(strokes_from_features)}
                - Optimal k: {knn_results['optimal_k_results']['optimal_k'] if 'optimal_k_results' in knn_results else 'N/A'}
                """)

                # Create KNN model package
                knn_model_package = {
                    'model': knn_results['model'].model,
                    'scaler': knn_results['model'].scaler if hasattr(knn_results['model'], 'scaler') else None,
                    'feature_names': knn_results['model'].feature_names,
                    'accuracy': knn_results['metrics']['accuracy'],
                    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'classes': list(strokes_from_features),
                    'stroke_names': stroke_names,
                    'test_size': test_size,
                    'parameters': knn_results.get('parameters', {})
                }

                # Save to bytes using pickle
                knn_bytes = pickle.dumps(knn_model_package)

                st.download_button(
                    label="📥 Download KNN Model",
                    data=knn_bytes,
                    file_name=f"swimming_knn_model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                    mime="application/octet-stream",
                    help="Save KNN model for future predictions"
                )

            with col2:
                # Random Forest Model Export
                st.markdown("#### 🌲 Random Forest Model")
                st.info(f"""
                **Model Details:**
                - Accuracy: {rf_results['metrics']['accuracy']:.3f}
                - Features: 16 features (8 motion + 8 shape)
                - Strokes: {len(strokes_from_features)}
                - Trees: {n_trees}
                """)

                # Create RF model package
                rf_model_package = {
                    'model': rf_results['model'].model,
                    'feature_names': rf_results['model'].feature_names,
                    'accuracy': rf_results['metrics']['accuracy'],
                    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'classes': list(strokes_from_features),
                    'stroke_names': stroke_names,
                    'n_estimators': n_trees,
                    'test_size': test_size,
                    'parameters': rf_results.get('parameters', {}),
                    'feature_importance': rf_results['metrics'].get('feature_importance', {})
                }

                # Save to bytes using pickle
                rf_bytes = pickle.dumps(rf_model_package)

                st.download_button(
                    label="📥 Download Random Forest Model",
                    data=rf_bytes,
                    file_name=f"swimming_rf_model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                    mime="application/octet-stream",
                    help="Save Random Forest model for future predictions"
                )

            # Combined model export
            st.markdown("#### 📦 Combined Export")

            col1, col2 = st.columns(2)

            with col1:
                # Export model info
                model_info = {
                    'knn_accuracy': knn_results['metrics']['accuracy'],
                    'rf_accuracy': rf_results['metrics']['accuracy'],
                    'best_model': 'KNN' if knn_acc > rf_acc else 'Random Forest',
                    'export_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'dataset_info': {
                        'total_samples': total,
                        'train_samples': train_count,
                        'test_samples': test_count,
                        'test_size': test_size,
                        'num_strokes': len(strokes_from_features),
                        'strokes': list(strokes_from_features)
                    }
                }

                info_df = pd.DataFrame([model_info])
                csv_info = info_df.to_csv(index=False).encode('utf-8')

                st.download_button(
                    label="📋 Download Model Info",
                    data=csv_info,
                    file_name="swimming_model_info.csv",
                    mime="text/csv",
                    help="Summary of trained models"
                )

            with col2:
                # Quick recommendation
                st.info(f"""
                **🎯 Recommendation:**
                {'**Use KNN model** for faster predictions' if knn_pred_time < rf_pred_time else '**Use Random Forest model** for higher accuracy'}

                **Next Steps:**
                1. Download preferred model
                2. Use prediction app for new swimming data
                3. Share model with coaches or researchers
                """)

            # ==========================
            # DOWNLOAD COMPARISON RESULTS
            # ==========================
            st.markdown("---")
            st.subheader("💾 Download Comparison Results")

            # Create comparison dataframe
            comparison_df = pd.DataFrame({
                'Algorithm': ['KNN', 'Random Forest'],
                'Accuracy': [knn_acc, rf_acc],
                'F1_Score': [
                    knn_results['metrics']['f1_score'],
                    rf_results['metrics']['f1_score']
                ],
                'Precision': [
                    knn_results['metrics']['precision'],
                    rf_results['metrics']['precision']
                ],
                'Recall': [
                    knn_results['metrics']['recall'],
                    rf_results['metrics']['recall']
                ],
                'Training_Time_s': [
                    knn_results['metrics']['training_time'],
                    rf_results['metrics']['training_time']
                ],
                'Prediction_Time_s': [
                    knn_results['metrics']['prediction_time'],
                    rf_results['metrics']['prediction_time']
                ],
                'Features_Used': [
                    '8 motion features',
                    '16 features (8 motion + 8 shape)'
                ],
                'CV_Mean_Accuracy': [
                    knn_results['cv_results']['mean_score'],
                    rf_results['cv_results']['mean_score']
                ],
                'CV_Std_Dev': [
                    knn_results['cv_results']['std_score'],
                    rf_results['cv_results']['std_score']
                ],
                'Winner': ['No', 'Yes'] if rf_acc > knn_acc else ['Yes', 'No'] if knn_acc > rf_acc else ['Tie', 'Tie']
            })

            csv_comparison = comparison_df.to_csv(index=False).encode('utf-8')

            col1, col2 = st.columns(2)

            with col1:
                st.download_button(
                    label="📥 Download Comparison Results",
                    data=csv_comparison,
                    file_name="swimming_algorithm_comparison.csv",
                    mime="text/csv",
                    key="download_comparison"
                )

            with col2:
                # Create detailed predictions file
                # Get predictions and true labels
                true_labels = knn_results['metrics']['true_labels']
                knn_preds = knn_results['metrics']['predictions']
                rf_preds = rf_results['metrics']['predictions']

                # Convert to display names
                true_labels_display = [stroke_names.get(label, label.replace('_', ' ').title()) for label in
                                       true_labels]
                knn_preds_display = [stroke_names.get(label, label.replace('_', ' ').title()) for label in knn_preds]
                rf_preds_display = [stroke_names.get(label, label.replace('_', ' ').title()) for label in rf_preds]

                # Create predictions DataFrame without segment starts
                predictions_df = pd.DataFrame({
                    'Test_Sample_Index': range(len(true_labels)),
                    'True_Label': true_labels_display,
                    'KNN_Prediction': knn_preds_display,
                    'RF_Prediction': rf_preds_display,
                    'KNN_Correct': true_labels == knn_preds,
                    'RF_Correct': true_labels == rf_preds
                })

                csv_predictions = predictions_df.to_csv(index=False).encode('utf-8')

                st.download_button(
                    label="📊 Download Predictions",
                    data=csv_predictions,
                    file_name="swimming_model_predictions.csv",
                    mime="text/csv",
                    key="download_predictions"
                )

            st.success("🎉 Medley swimming stroke analysis complete! Results are ready for download.")

        elif st.session_state.features_df is not None:
            st.info("👆 Click 'Run KNN vs Random Forest Comparison' to start machine learning analysis!")

    else:
        # Initial state - no features processed yet
        st.info("👆 Click 'Process Data & Extract Features' to begin feature extraction.")

        # Show raw data preview
        with st.expander("📄 Raw Data Preview", expanded=True):
            st.dataframe(df.head(), use_container_width=True)
            time_col = "Time (s)" if "Time (s)" in df.columns else "seconds_elapsed"
            if time_col in df.columns:
                st.write(f"**Data Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
                st.write(f"**Time Range:** {df[time_col].min():.1f}s to {df[time_col].max():.1f}s")
            else:
                st.write(f"**Data Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    <p>🏊‍♂️ Swimming Stroke Recognition Tool | KNN vs Random Forest Comparison</p>
    <p>📊 Features extracted: 16 features per {segment_duration if 'segment_duration' in locals() else 2}-second window (8 motion + 8 shape)</p>
    <p>🆕 Now with customizable swimming strokes!</p>
</div>
""", unsafe_allow_html=True)