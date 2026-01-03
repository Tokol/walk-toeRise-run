import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from csv_processor import load_and_process_csv, manual_column_mapping, show_data_preview

# Import custom modules
from features import compute_magnitude
try:
    from knn_module import run_knn_pipeline
except ImportError:
    # For Streamlit Cloud or when module is in same directory
    # Create a workaround or include the code directly
    from knn_module import run_knn_pipeline

from randomforest_module import run_rf_pipeline
from feature_relationships import FeatureRelationships 

import pickle


# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Walk-ToeRise-Run Activity Recognition",
    page_icon="assets/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚶‍♂️ Walk, ToeRise, Run Activity Recognition: Feature Extraction & ML Comparison")
st.markdown("""
This tool processes accelerometer data, extracts features, and compares 
**K-Nearest Neighbors (KNN)** vs **Random Forest** algorithms for activity classification.
""")

# ==========================

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
    "📤 Upload CSV file (CSV only)",
    type=["csv"]
)

# Initialize or reset session state when new file is uploaded
if uploaded_file is not None and st.session_state.df is None:
    # Try auto-loading first
    df = load_and_process_csv(uploaded_file)
    
    if df is not None:
        # Success - compute magnitude and store
        st.session_state.df = compute_magnitude(df)
        
        # Reset features and results when new file is uploaded
        st.session_state.features_df = None
        st.session_state.time_ranges = None
        st.session_state.knn_results = None
        st.session_state.rf_results = None
    else:
        # Auto-detection failed, try manual
        st.warning("⚠️ Auto-detection failed. Trying manual mapping...")
        df_raw = pd.read_csv(uploaded_file)
        df = manual_column_mapping(df_raw)
        
        if df is not None:
            st.session_state.df = compute_magnitude(df)
            st.session_state.features_df = None
            st.session_state.time_ranges = None
            st.session_state.knn_results = None
            st.session_state.rf_results = None
        else:
            st.info("👆 Please complete the manual column mapping above.")
            st.stop()

# If no file is uploaded, show info and stop
if uploaded_file is None and st.session_state.df is None:
    st.info("📝 Please upload a CSV file to continue.")
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
            min_value=1.0,
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
        **Feature Strategy:**
        - **KNN:** Uses 4 motion features only
        - **Random Forest:** Uses all 9 features (4 motion + 5 shape)
        """)

    # ==========================
    # ACTIVITY TIME RANGE SELECTION
    # ==========================
    st.header("📅 Step 1: Define Activity Time Ranges")

    # Initialize session state for preset and custom values
    if 'preset_selector' not in st.session_state:
        st.session_state.preset_selector = "Together"
    if 'custom_walking_start' not in st.session_state:
        st.session_state.custom_walking_start = 0.0
    if 'custom_walking_end' not in st.session_state:
        st.session_state.custom_walking_end = 0.0
    if 'custom_toe_start' not in st.session_state:
        st.session_state.custom_toe_start = 0.0
    if 'custom_toe_end' not in st.session_state:
        st.session_state.custom_toe_end = 0.0
    if 'custom_run_start' not in st.session_state:
        st.session_state.custom_run_start = 0.0
    if 'custom_run_end' not in st.session_state:
        st.session_state.custom_run_end = 0.0

    # Preset selection with Custom option
    col1, col2 = st.columns([3, 1])

    with col1:
        preset = st.selectbox(
            "Choose preset time ranges or select Custom to define your own:",
            ["Together", "Williamsson", "Lama", "Custom"],
            key="preset_selector"
        )

    with col2:
        st.write("")
        st.write("")
        if st.button("🔄 Reset to Selected Preset", key="reset_preset"):
            st.session_state.features_df = None
            st.session_state.time_ranges = None
            st.rerun()

    # Preset defaults
    if preset == "Williamsson":
        defaults = {
            "walking_start": 1,
            "walking_end": 99,
            "toe_start": 135,
            "toe_end": 166,
            "run_start": 210,
            "run_end": 274,
        }
    elif preset == "Lama":
        defaults = {
            "walking_start": 1,
            "walking_end": 98,
            "toe_start": 135,
            "toe_end": 166,
            "run_start": 210,
            "run_end": 270,
        }
    elif preset == "Together":
        defaults = {
            "walking_start": 5,
            "walking_end": 89,
            "toe_start": 122,
            "toe_end": 181,
            "run_start": 220,
            "run_end": 274,
        }
    else:  # Custom
        defaults = {
            "walking_start": 0,
            "walking_end": 0,
            "toe_start": 0,
            "toe_end": 0,
            "run_start": 0,
            "run_end": 0,
        }

    # Time Range Inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🚶 Walking")
        walking_start = st.number_input(
            "Walking start (s)",
            min_value=0.0,
            value=float(defaults["walking_start"]),
            step=1.0,
            format="%.0f",
            key="walking_start_input"
        )
        walking_end = st.number_input(
            "Walking end (s)",
            min_value=0.0,
            value=float(defaults["walking_end"]),
            step=1.0,
            format="%.0f",
            key="walking_end_input"
        )
        # Store custom values in session state
        if preset == "Custom":
            st.session_state.custom_walking_start = walking_start
            st.session_state.custom_walking_end = walking_end

    with col2:
        st.markdown("### 👣 Toe-rise")
        toe_start = st.number_input(
            "Toe-rise start (s)",
            min_value=0.0,
            value=float(defaults["toe_start"]),
            step=1.0,
            format="%.0f",
            key="toe_start_input"
        )
        toe_end = st.number_input(
            "Toe-rise end (s)",
            min_value=0.0,
            value=float(defaults["toe_end"]),
            step=1.0,
            format="%.0f",
            key="toe_end_input"
        )
        # Store custom values in session state
        if preset == "Custom":
            st.session_state.custom_toe_start = toe_start
            st.session_state.custom_toe_end = toe_end

    with col3:
        st.markdown("### 🏃 Running")
        run_start = st.number_input(
            "Running start (s)",
            min_value=0.0,
            value=float(defaults["run_start"]),
            step=1.0,
            format="%.0f",
            key="run_start_input"
        )
        run_end = st.number_input(
            "Running end (s)",
            min_value=0.0,
            value=float(defaults["run_end"]),
            step=1.0,
            format="%.0f",
            key="run_end_input"
        )
        # Store custom values in session state
        if preset == "Custom":
            st.session_state.custom_run_start = run_start
            st.session_state.custom_run_end = run_end

    # Store time ranges for visualization
    st.session_state.time_ranges = {
        "walking_start": walking_start,
        "walking_end": walking_end,
        "toe_start": toe_start,
        "toe_end": toe_end,
        "run_start": run_start,
        "run_end": run_end
    }

    # ==========================
    # PROCESS DATA BUTTON
    # ==========================
    st.markdown("---")

    if st.button("🔧 Process Data & Extract Features", type="primary", key="process_button"):
        with st.spinner("Processing data and extracting features..."):
            # Clear previous features and results
            st.session_state.features_df = None
            st.session_state.knn_results = None
            st.session_state.rf_results = None

            # Define activities from time ranges
            activities = [
                {"name": "walking", "start": walking_start, "end": walking_end},
                {"name": "toe_rise", "start": toe_start, "end": toe_end},
                {"name": "running", "start": run_start, "end": run_end}
            ]

            # Validate activities
            valid_activities = []
            for activity in activities:
                name = activity["name"]
                start = activity["start"]
                end = activity["end"]

                if start >= end:
                    st.warning(f"⚠️ Activity '{name}' has invalid time range ({start}s to {end}s). Skipping.")
                    continue

                valid_activities.append(activity)

            if len(valid_activities) < 2:
                st.error("❌ Need at least 2 valid activities with proper time ranges.")
                st.stop()

            # ==========================
            # SEGMENT + LABEL (2s WINDOWS, 9 FEATURES)
            # ==========================
            features = []

            start_time = df["Time (s)"].min()
            end_time = df["Time (s)"].max()

            for seg_start in np.arange(start_time, end_time, segment_duration):
                seg_end = seg_start + segment_duration

                seg_df = df[
                    (df["Time (s)"] >= seg_start) &
                    (df["Time (s)"] < seg_end)
                    ]

                if seg_df.empty:
                    continue

                mag = seg_df["magnitude"]

                # Motion features (4)
                mean_mag = mag.mean()
                rms_mag = np.sqrt(np.mean(mag ** 2))
                std_mag = mag.std()
                p2p_mag = mag.max() - mag.min()

                # Shape features (5)
                kurt_mag = mag.kurt()
                skew_mag = mag.skew()
                median_mag = mag.median()

                q1 = mag.quantile(0.25)
                q3 = mag.quantile(0.75)
                iqr_mag = q3 - q1

                peaks, _ = find_peaks(mag)
                peak_count = len(peaks)

                # Activity label - check which activity this segment belongs to
                activity_label = None
                for activity in valid_activities:
                    if activity["start"] <= seg_start < activity["end"]:
                        activity_label = activity["name"]
                        break

                if activity_label is None:
                    # Segment doesn't belong to any defined activity
                    continue

                features.append({
                    "segment_start": seg_start,
                    "segment_end": seg_end,
                    "mean_magnitude": mean_mag,
                    "rms_magnitude": rms_mag,
                    "std_magnitude": std_mag,
                    "p2p_magnitude": p2p_mag,
                    "kurtosis_magnitude": kurt_mag,
                    "skewness_magnitude": skew_mag,
                    "median_magnitude": median_mag,
                    "iqr_magnitude": iqr_mag,
                    "peak_count": peak_count,
                    "activity": activity_label
                })

            features_df = pd.DataFrame(features)

            if features_df.empty:
                st.warning("⚠️ No segments fall within the selected time ranges.")
                st.stop()

            # Store in session state
            st.session_state.features_df = features_df

            # Store activities
            st.session_state.activities = [
                {"name": "walking", "display": "Walking"},
                {"name": "toe_rise", "display": "Toe-rise"},
                {"name": "running", "display": "Running"}
            ]

            st.success(f"✅ Extracted {len(features_df)} segments for 3 activities!")

    # ==========================
    # DISPLAY RESULTS IF PROCESSED
    # ==========================
    if st.session_state.features_df is not None:
        features_df = st.session_state.features_df
        time_ranges = st.session_state.time_ranges
        activities = st.session_state.activities

        st.markdown("---")
        st.header("📊 Step 2: Data Exploration & Visualization")

        # Get unique activities for color mapping
        activities_from_features = list(features_df["activity"].unique())
        num_activities = len(activities_from_features)

        # Create consistent color mappings
        matplotlib_activity_colors = {
            "walking": "tab:blue",
            "toe_rise": "tab:orange",
            "running": "tab:red"
        }
        plotly_activity_colors = {
            "walking": "blue",
            "toe_rise": "orange",
            "running": "red"
        }
        activity_names = {
            "walking": "Walking",
            "toe_rise": "Toe-rise",
            "running": "Running"
        }

        # ==========================
        # 1. RAW SIGNAL PLOT
        # ==========================
        st.subheader("📈 Raw Signal with Activity Regions")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df["Time (s)"], df["magnitude"], color="black", alpha=0.6, linewidth=0.8)

        # Add activity regions to the plot
        if "walking_start" in time_ranges and "walking_end" in time_ranges:
            ax.axvspan(time_ranges["walking_start"], time_ranges["walking_end"],
                       color="tab:blue", alpha=0.25, label="Walking")

        if "toe_start" in time_ranges and "toe_end" in time_ranges:
            ax.axvspan(time_ranges["toe_start"], time_ranges["toe_end"],
                       color="tab:orange", alpha=0.25, label="Toe-rise")

        if "run_start" in time_ranges and "run_end" in time_ranges:
            ax.axvspan(time_ranges["run_start"], time_ranges["run_end"],
                       color="tab:red", alpha=0.25, label="Running")

        # Add legend
        ax.legend(
            title="Activity",
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            fontsize=9,
            title_fontsize=10
        )

        x_start = int(df["Time (s)"].min())
        x_end = int(df["Time (s)"].max())
        ax.set_xticks(np.arange(x_start, x_end + 1, 10))

        y_max = int(df["magnitude"].max()) + 1
        ax.set_yticks(np.arange(0, y_max, 5))

        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("Magnitude (m/s²)", fontsize=10)
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)

        # ==========================
        # 2. DATA OVERVIEW
        # ==========================
        st.subheader("📋 Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Samples", len(features_df))

        with col2:
            st.metric("Features Extracted", 9)

        with col3:
            st.metric("Activities", features_df["activity"].nunique())

        with col4:
            st.metric("Segment Duration", f"{segment_duration}s")

        # Segment Count by Activity
        st.subheader("📊 Activity Distribution")

        segment_counts = features_df["activity"].value_counts()
        segment_counts_display = pd.DataFrame({
            "Activity": [activity_names.get(act, act.replace("_", " ").title()) for act in segment_counts.index],
            "Segments": segment_counts.values,
            "Percentage": (segment_counts.values / segment_counts.sum() * 100).round(1)
        })

        col1, col2 = st.columns(2)

        with col1:
            fig_segments = go.Figure(data=[
                go.Pie(
                    labels=segment_counts_display["Activity"],
                    values=segment_counts_display["Segments"],
                    hole=0.3,
                    marker_colors=[plotly_activity_colors.get(act, 'gray') for act in segment_counts.index],
                    textinfo='label+value+percent',
                    hovertemplate="<b>%{label}</b><br>Segments: %{value}<br>Percentage: %{percent}<extra></extra>"
                )
            ])

            fig_segments.update_layout(
                title="Segments per Activity",
                height=400,
                showlegend=True
            )

            st.plotly_chart(fig_segments, use_container_width=True)

        with col2:
            # Feature Discriminability
            all_features = ["mean_magnitude", "rms_magnitude", "std_magnitude", "p2p_magnitude",
                            "kurtosis_magnitude", "skewness_magnitude", "median_magnitude",
                            "iqr_magnitude", "peak_count"]

            feature_discriminability = []
            for feature in all_features:
                groups = [features_df[features_df["activity"] == act][feature].values
                          for act in activities_from_features]
                groups = [g for g in groups if len(g) > 0]
                if len(groups) >= 2:
                    f_stat, _ = stats.f_oneway(*groups)
                    feature_discriminability.append({
                        "Feature": feature.replace("_", " ").title(),
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
        st.subheader("🔍 Feature Relationships")

        feature_viz = FeatureRelationships(
        features_df=features_df,
        activity_colors=plotly_activity_colors,
        activity_names=activity_names
        )
        feature_viz.render_dashboard()


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

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                label="📥 Download Features as CSV",
                data=csv_data,
                file_name="features_2s_windows.csv",
                mime="text/csv",
                key="download_csv"
            )

        with col2:
            # Show feature summary
            feature_summary = pd.DataFrame({
                "Feature Type": ["Motion"] * 4 + ["Shape"] * 5,
                "Feature Name": [
                    "Mean Magnitude", "RMS Magnitude", "Std Magnitude", "Peak-to-Peak",
                    "Kurtosis", "Skewness", "Median", "IQR", "Peak Count"
                ],
                "Description": [
                    "Average acceleration",
                    "Root mean square (energy)",
                    "Standard deviation (variability)",
                    "Range from min to max",
                    "Tailedness of distribution",
                    "Asymmetry of distribution",
                    "Middle value",
                    "Interquartile range",
                    "Number of local maxima"
                ]
            })

            with st.expander("📚 Feature Descriptions", expanded=False):
                st.dataframe(feature_summary, use_container_width=True)

        # ==========================
        # 6. MOTION FEATURES EXPLORER
        # ==========================
        st.markdown("---")
        st.subheader("⚡ Motion Features Explorer")

        # Motion features
        motion_features = ["mean_magnitude", "rms_magnitude", "std_magnitude", "p2p_magnitude"]

        # Create tabs for motion features
        motion_tab1, motion_tab2, motion_tab3, motion_tab4 = st.tabs([
            "📈 Feature Trends",
            "🎯 Activity Comparison",
            "📊 Motion Energy Dashboard",
            "📋 Feature Statistics"
        ])

        with motion_tab1:
            selected_motion_feature = st.selectbox(
                "Select motion feature to explore:",
                motion_features,
                key="motion_feature_selector"
            )

            # Feature description
            motion_descriptions = {
                "mean_magnitude": "Average acceleration magnitude in the window",
                "rms_magnitude": "Root Mean Square - measures signal energy/power",
                "std_magnitude": "Standard deviation - measures variability",
                "p2p_magnitude": "Peak-to-peak - range from minimum to maximum"
            }

            st.info(
                f"**{selected_motion_feature.replace('_', ' ').title()}**: {motion_descriptions[selected_motion_feature]}")

            # Line plot across time segments
            fig_m1 = go.Figure()

            for activity in activities_from_features:
                activity_data = features_df[features_df["activity"] == activity]
                if not activity_data.empty:
                    display_name = activity_names.get(activity, activity.replace("_", " ").title())
                    fig_m1.add_trace(go.Scatter(
                        x=activity_data["segment_start"],
                        y=activity_data[selected_motion_feature],
                        mode='lines+markers',
                        name=display_name,
                        line=dict(color=plotly_activity_colors.get(activity, 'gray'), width=2),
                        marker=dict(size=6),
                        hovertemplate='<b>%{text}</b><br>' +
                                      'Time: %{x:.1f}s<br>' +
                                      'Value: %{y:.3f}<extra></extra>',
                        text=[display_name] * len(activity_data)
                    ))

            fig_m1.update_layout(
                title=f"{selected_motion_feature.replace('_', ' ').title()} Across Time Segments",
                xaxis_title="Segment Start Time (s)",
                yaxis_title=selected_motion_feature.replace('_', ' ').title() + " (m/s²)",
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

            # Add scatter points for each activity
            for activity in activities_from_features:
                activity_data = features_df[features_df["activity"] == activity]
                if not activity_data.empty:
                    display_name = activity_names.get(activity, activity.replace("_", " ").title())
                    fig_m2.add_trace(go.Scatter(
                        x=activity_data.index,
                        y=activity_data[selected_motion_feature_t2],
                        mode='markers',
                        name=display_name,
                        marker=dict(
                            size=8,
                            color=plotly_activity_colors.get(activity, 'gray'),
                            opacity=0.7,
                            line=dict(width=1, color='white')
                        ),
                        hovertemplate='<b>%{text}</b><br>' +
                                      'Segment Index: %{x}<br>' +
                                      'Value: %{y:.3f} m/s²<br>' +
                                      'Time: %{customdata:.1f}s<extra></extra>',
                        text=[display_name] * len(activity_data),
                        customdata=activity_data["segment_start"]
                    ))

            # Calculate statistics for summary box
            summary_data = {}
            for activity in activities_from_features:
                activity_data = features_df[features_df["activity"] == activity]
                if not activity_data.empty:
                    values = activity_data[selected_motion_feature_t2]
                    display_name = activity_names.get(activity, activity.replace("_", " ").title())
                    summary_data[display_name] = {
                        "activity": activity,
                        "mean": values.mean(),
                        "std": values.std(),
                        "min": values.min(),
                        "max": values.max(),
                        "count": len(values)
                    }

            fig_m2.update_layout(
                title=f"{selected_motion_feature_t2.replace('_', ' ').title()} - Activity Comparison",
                xaxis_title="Segment Index (Order in Dataset)",
                yaxis_title=selected_motion_feature_t2.replace('_', ' ').title() + " (m/s²)",
                height=500,
                hovermode="closest",
                showlegend=True
            )

            st.plotly_chart(fig_m2, use_container_width=True)

            # Summary Box
            st.subheader("📊 Comparison Summary")

            if summary_data:
                # Find which activity has highest mean
                highest_activity_name = max(summary_data.items(), key=lambda x: x[1]["mean"])
                lowest_activity_name = min(summary_data.items(), key=lambda x: x[1]["mean"])

                highest_activity = highest_activity_name[0]
                lowest_activity = lowest_activity_name[0]

                # Calculate percentage difference
                if lowest_activity_name[1]["mean"] > 0:
                    pct_diff = ((highest_activity_name[1]["mean"] - lowest_activity_name[1]["mean"]) /
                                lowest_activity_name[1]["mean"]) * 100
                else:
                    pct_diff = 0

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        label="Highest Average",
                        value=f"{highest_activity}",
                        delta=f"{highest_activity_name[1]['mean']:.3f} m/s²"
                    )

                with col2:
                    st.metric(
                        label="Lowest Average",
                        value=f"{lowest_activity}",
                        delta=f"{lowest_activity_name[1]['mean']:.3f} m/s²"
                    )

                with col3:
                    st.metric(
                        label="Difference",
                        value=f"{pct_diff:.1f}%",
                        delta=f"{highest_activity_name[1]['mean'] - lowest_activity_name[1]['mean']:.3f} m/s²"
                    )

                # Variability comparison
                st.subheader("📈 Variability Comparison")
                variability_data = []
                for display_name, stats in summary_data.items():
                    variability_data.append({
                        "Activity": display_name,
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
            sorted_df['cumulative_rms'] = sorted_df['rms_magnitude'].cumsum()

            fig_energy = go.Figure()

            # Add cumulative energy line
            fig_energy.add_trace(go.Scatter(
                x=sorted_df["segment_start"],
                y=sorted_df["cumulative_rms"],
                mode='lines',
                name='Cumulative RMS',
                line=dict(color='green', width=3),
                fill='tozeroy',
                fillcolor='rgba(0, 255, 0, 0.1)'
            ))

            fig_energy.update_layout(
                title="Cumulative Motion Energy Over Time",
                xaxis_title="Segment Start Time (s)",
                yaxis_title="Cumulative RMS (m/s²)",
                height=400,
                hovermode="x unified"
            )

            st.plotly_chart(fig_energy, use_container_width=True)

            # 3. Activity Energy Summary
            st.subheader("⚡ Activity Energy Comparison")

            # Calculate average RMS per activity
            activity_rms = features_df.groupby("activity")["rms_magnitude"].agg(['mean', 'std', 'count']).round(3)
            activity_rms = activity_rms.reset_index()
            activity_rms['activity_display'] = activity_rms['activity'].apply(
                lambda x: activity_names.get(x, x.replace('_', ' ').title())
            )

            # Sort by mean RMS (highest to lowest)
            activity_rms = activity_rms.sort_values('mean', ascending=False)

            fig_bar = go.Figure()

            fig_bar.add_trace(go.Bar(
                x=activity_rms['activity_display'],
                y=activity_rms['mean'],
                error_y=dict(
                    type='data',
                    array=activity_rms['std'],
                    visible=True
                ),
                marker_color=[plotly_activity_colors.get(act, 'gray') for act in activity_rms['activity']],
                text=activity_rms['mean'].round(3),
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>' +
                              'Average RMS: %{y:.3f} m/s²<br>' +
                              'Std: %{customdata:.3f} m/s²<br>' +
                              'Samples: %{text}<extra></extra>',
                customdata=activity_rms['std']
            ))

            fig_bar.update_layout(
                title="Average Motion Energy (RMS) by Activity",
                xaxis_title="Activity",
                yaxis_title="Average RMS (m/s²)",
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

            # Activity-wise statistics for motion features
            st.subheader("Activity-wise Motion Statistics")

            # Calculate statistics
            motion_activity_stats = (
                features_df.groupby("activity")[motion_features]
                .agg(['mean', 'std'])
                .round(3)
            )

            # Flatten multi-level columns
            motion_activity_stats.columns = [
                f"{feature}_{stat}"
                for feature, stat in motion_activity_stats.columns
            ]

            # Create display version with formatted index
            motion_activity_stats_display = motion_activity_stats.copy()
            motion_activity_stats_display.index = (
                motion_activity_stats_display.index
                .map(lambda x: activity_names.get(x, x.replace('_', ' ').title()))
            )

            # Display with better formatting
            st.dataframe(
                motion_activity_stats_display,
                use_container_width=True
            )

            # ==========================
            # 7. SHAPE FEATURE EXPLORER
            # ==========================
            st.markdown("---")
            st.subheader("📊 Shape Feature Explorer")

        # Define shape features
        shape_features = [
            "kurtosis_magnitude",
            "skewness_magnitude",
            "median_magnitude",
            "iqr_magnitude",
            "peak_count"
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
                "kurtosis_magnitude": "Measures tailedness/sharpness. High kurtosis = sharp peaks, heavy tails",
                "skewness_magnitude": "Measures asymmetry. Positive = right-skewed, Negative = left-skewed",
                "median_magnitude": "Middle value of magnitude distribution",
                "iqr_magnitude": "Spread of middle 50% of data. Robust to outliers",
                "peak_count": "Number of local maxima in the 2-second window"
            }

            st.info(
                f"**{selected_shape_feature.replace('_', ' ').title()}**: {shape_descriptions[selected_shape_feature]}")

            # Line plot across time segments
            fig_shape1 = go.Figure()

            for activity in activities_from_features:
                activity_data = features_df[features_df["activity"] == activity]
                if not activity_data.empty:
                    display_name = activity_names.get(activity, activity.replace("_", " ").title())
                    fig_shape1.add_trace(go.Scatter(
                        x=activity_data["segment_start"],
                        y=activity_data[selected_shape_feature],
                        mode='lines+markers',
                        name=display_name,
                        line=dict(color=plotly_activity_colors.get(activity, 'gray'), width=2),
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
            # Boxplot by activity
            fig_shape2 = go.Figure()

            for activity in activities_from_features:
                activity_data = features_df[features_df["activity"] == activity]
                if not activity_data.empty:
                    display_name = activity_names.get(activity, activity.replace("_", " ").title())
                    fig_shape2.add_trace(go.Box(
                        y=activity_data[selected_shape_feature],
                        name=display_name,
                        marker_color=plotly_activity_colors.get(activity, 'gray'),
                        boxmean=True
                    ))

            fig_shape2.update_layout(
                title=f"Distribution of {selected_shape_feature.replace('_', ' ').title()} by Activity",
                yaxis_title=selected_shape_feature.replace('_', ' ').title(),
                xaxis_title="Activity",
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
                    title=f"Distribution by Activity",
                    color_discrete_map=plotly_activity_colors,
                    labels={
                        selected_shape_feature: selected_shape_feature.replace('_', ' ').title(),
                        "activity": "Activity"
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

            # Activity-wise statistics
            def display_activity_statistics(df, features, title, activity_names=None):
                """Display formatted activity statistics for given features"""
                st.subheader(title)

                # Calculate statistics
                stats = (
                    df.groupby("activity")[features]
                    .agg(['mean', 'std'])
                    .round(3)
                )

                # Flatten columns
                stats.columns = [f"{feat}_{stat}" for feat, stat in stats.columns]

                # Format activity names
                if activity_names:
                    stats.index = stats.index.map(
                        lambda x: activity_names.get(x, x.replace('_', ' ').title())
                    )

                # Display
                st.dataframe(stats, use_container_width=True)

                return stats

            # Usage
            motion_stats = display_activity_statistics(
                features_df, 
                motion_features, 
                "Activity-wise Motion Statistics",
                activity_names
            )

            shape_stats = display_activity_statistics(
                features_df,
                shape_features,
                "Activity-wise Shape Statistics",
                activity_names
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
        - **KNN:** k={k_value} neighbors, 4 motion features
        - **Random Forest:** {n_trees} trees, all 9 features
        - **Activities:** Walking, Toe-rise, Running
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
            - **KNN Features:** 4 motion features
            - **RF Features:** 9 features (4 motion + 5 shape)
            - **Number of Activities:** 3
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
                    # Update activity labels if needed
                    if hasattr(knn_metrics_fig, 'data'):
                        st.plotly_chart(knn_metrics_fig, use_container_width=True)
                    st.caption("KNN Performance Metrics")

                with col2:
                    rf_metrics_fig = rf_results['figures']['metrics_bar']
                    # Update activity labels if needed
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
                            display_label = activity_names.get(label, label.replace('_', ' ').title())
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
                            display_label = activity_names.get(label, label.replace('_', ' ').title())
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
            # ENHANCEMENT 1: PERFORMANCE SUMMARY CARD
            # ============================================
            # Calculate values needed for all enhancements
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
                           '🎯 Activities'],
                'KNN': [
                    f"{knn_acc:.3f}",
                    f"{knn_f1:.3f}",
                    f"{knn_time:.3f}s",
                    f"{knn_pred_time:.3f}s",
                    "4 motion features",
                    "3 activities"
                ],
                'Random Forest': [
                    f"{rf_acc:.3f}",
                    f"{rf_f1:.3f}",
                    f"{rf_time:.3f}s",
                    f"{rf_pred_time:.3f}s",
                    "9 features (motion + shape)",
                    "3 activities"
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

            3. **Activity Analysis:** You're classifying 3 different activities
            4. **Recommendation:** {'**Use KNN** for real-time applications' if knn_pred_time < rf_pred_time else '**Use Random Forest** for batch processing'}
            """)

            # ============================================
            # ENHANCEMENT 2: CONFUSION ANALYSIS
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
                knn_true_display = [activity_names.get(label, label.replace('_', ' ').title()) for label in knn_true]
                knn_pred_display = [activity_names.get(label, label.replace('_', ' ').title()) for label in knn_pred]

                # Create detailed confusion DataFrame
                knn_confusion_details = pd.crosstab(
                    pd.Series(knn_true_display, name='True Activity'),
                    pd.Series(knn_pred_display, name='Predicted Activity'),
                    margins=True
                )

                st.markdown("##### KNN Confusion Details")
                st.dataframe(knn_confusion_details.style.background_gradient(cmap='Blues'),
                             use_container_width=True)

                # KNN misclassification analysis
                knn_misclassified = knn_true != knn_pred
                if sum(knn_misclassified) > 0:
                    knn_errors = pd.DataFrame({
                        'True': [activity_names.get(label, label.replace('_', ' ').title()) for label in
                                 knn_true[knn_misclassified]],
                        'Predicted': [activity_names.get(label, label.replace('_', ' ').title()) for label in
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
                rf_true_display = [activity_names.get(label, label.replace('_', ' ').title()) for label in rf_true]
                rf_pred_display = [activity_names.get(label, label.replace('_', ' ').title()) for label in rf_pred]

                # Create detailed confusion DataFrame
                rf_confusion_details = pd.crosstab(
                    pd.Series(rf_true_display, name='True Activity'),
                    pd.Series(rf_pred_display, name='Predicted Activity'),
                    margins=True
                )

                st.markdown("##### Random Forest Confusion Details")
                st.dataframe(rf_confusion_details.style.background_gradient(cmap='Greens'),
                             use_container_width=True)

                # RF misclassification analysis
                rf_misclassified = rf_true != rf_pred
                if sum(rf_misclassified) > 0:
                    rf_errors = pd.DataFrame({
                        'True': [activity_names.get(label, label.replace('_', ' ').title()) for label in
                                 rf_true[rf_misclassified]],
                        'Predicted': [activity_names.get(label, label.replace('_', ' ').title()) for label in
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

            # Most confused activities
            st.markdown("#### 🎯 Most Challenging Activities")
            activity_errors = []

            for activity in activities_from_features:
                display_name = activity_names.get(activity, activity.replace("_", " ").title())
                knn_act_error = sum((knn_true == activity) & knn_misclassified) / sum(knn_true == activity) if sum(
                    knn_true == activity) > 0 else 0
                rf_act_error = sum((rf_true == activity) & rf_misclassified) / sum(rf_true == activity) if sum(
                    rf_true == activity) > 0 else 0
                activity_errors.append({
                    'Activity': display_name,
                    'KNN Error Rate': knn_act_error,
                    'RF Error Rate': rf_act_error,
                    'Total Samples': sum(knn_true == activity)
                })

            error_df = pd.DataFrame(activity_errors)
            st.dataframe(error_df.sort_values('KNN Error Rate', ascending=False),
                         use_container_width=True)

            # ============================================
            # ENHANCEMENT 3: TEST SIZE IMPACT ANALYSIS
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
                - **Test size:** At least 20-30 samples per activity class
                - **Training size:** At least 50-100 samples per activity class
                - **Your current split:** {test_count} test samples, {train_count} training samples
                - **Samples per activity:** {len(features_df) // len(activities_from_features)} average
                """)

            # ============================================
            # NEW: DATA SPLIT EXPLANATION (ALL 3 LEVELS)
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
                - Activities: **3**

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
                    f'{total} activity segments',
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

                    **Analogy:** Imagine preparing for a math exam:
                    - **Training ({train_pct:.0f}%):** Textbook problems you study
                    - **Testing ({test_pct:.0f}%):** Exam questions you've never seen

                    **Without splitting:**
                    > "I memorized all {total} textbook problems and got 100% on the same problems!"
                    → **Useless** - doesn't test real understanding

                    **With splitting:**
                    > "I studied {train_count} problems, then correctly solved new problems!"
                    → **{knn_results['metrics']['accuracy']:.1%} real understanding** that works on new problems
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
                    - Activities: **3**

                    **Rule of thumb:** At least 20 samples per activity class in test set
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
                - Features: 4 motion features
                - Activities: 3
                - Optimal k: {knn_results['optimal_k_results']['optimal_k'] if 'optimal_k_results' in knn_results else 'N/A'}
                """)

                # Create KNN model package
                knn_model_package = {
                    'model': knn_results['model'].model,
                    'scaler': knn_results['model'].scaler if hasattr(knn_results['model'], 'scaler') else None,
                    'feature_names': knn_results['model'].feature_names,
                    'accuracy': knn_results['metrics']['accuracy'],
                    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'classes': list(activities_from_features),
                    'activity_names': activity_names,
                    'test_size': test_size,
                    'parameters': knn_results.get('parameters', {})
                }

                # Save to bytes using pickle
                knn_bytes = pickle.dumps(knn_model_package)

                st.download_button(
                    label="📥 Download KNN Model",
                    data=knn_bytes,
                    file_name=f"knn_model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                    mime="application/octet-stream",
                    help="Save KNN model for future predictions"
                )

            with col2:
                # Random Forest Model Export
                st.markdown("#### 🌲 Random Forest Model")
                st.info(f"""
                **Model Details:**
                - Accuracy: {rf_results['metrics']['accuracy']:.3f}
                - Features: 9 features (4 motion + 5 shape)
                - Activities: 3
                - Trees: {n_trees}
                """)

                # Create RF model package
                rf_model_package = {
                    'model': rf_results['model'].model,
                    'feature_names': rf_results['model'].feature_names,
                    'accuracy': rf_results['metrics']['accuracy'],
                    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'classes': list(activities_from_features),
                    'activity_names': activity_names,
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
                    file_name=f"rf_model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl",
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
                        'num_activities': 3,
                        'activities': list(activities_from_features)
                    }
                }

                info_df = pd.DataFrame([model_info])
                csv_info = info_df.to_csv(index=False).encode('utf-8')

                st.download_button(
                    label="📋 Download Model Info",
                    data=csv_info,
                    file_name="model_info.csv",
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
                2. Use prediction app for new data
                3. Share model with colleagues
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
                    '4 motion features',
                    '9 features (4 motion + 5 shape)'
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
                    file_name="algorithm_comparison.csv",
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
                true_labels_display = [activity_names.get(label, label.replace('_', ' ').title()) for label in
                                       true_labels]
                knn_preds_display = [activity_names.get(label, label.replace('_', ' ').title()) for label in knn_preds]
                rf_preds_display = [activity_names.get(label, label.replace('_', ' ').title()) for label in rf_preds]

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
                    file_name="model_predictions.csv",
                    mime="text/csv",
                    key="download_predictions"
                )

            st.success("🎉 Analysis complete! Results are ready for download.")

        elif st.session_state.features_df is not None:
            st.info("👆 Click 'Run KNN vs Random Forest Comparison' to start machine learning analysis!")

    else:
        # Initial state - no features processed yet
        st.info("👆 Click 'Process Data & Extract Features' to begin feature extraction.")

        # Show raw data preview
        with st.expander("📄 Raw Data Preview", expanded=True):
            show_data_preview(df)
            st.write(f"**Data Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.write(f"**Time Range:** {df['Time (s)'].min():.1f}s to {df['Time (s)'].max():.1f}s")

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    <p>🎯 Activity Recognition Tool | KNN vs Random Forest Comparison</p>
    <p>📊 Features extracted: 9 features per {segment_duration if 'segment_duration' in locals() else 2}-second window</p>
    <p>🏃 Activities: Walking, Toe-rise, Running</p>
</div>
""", unsafe_allow_html=True)