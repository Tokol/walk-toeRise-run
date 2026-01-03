# feature_description_dashboard.py
import streamlit as st
import pandas as pd
from typing import List, Dict, Optional

class FeatureDescriptionDashboard:
    """Class to display the feature description dashboard for swimming stroke analysis."""
    
    def __init__(self, feature_info: List[Dict]):
        """
        Initialize the FeatureDescriptionDashboard.
        
        Args:
            feature_info: List of dictionaries containing feature information
        """
        self.feature_info = feature_info
        self.feature_df = pd.DataFrame(feature_info)
        
    def _color_feature_type(self, val: str) -> str:
        """Apply color coding for feature types."""
        if val == "Motion":
            return "background-color: #e6f7ff; color: #0066cc; font-weight: bold;"
        elif val == "Shape":
            return "background-color: #f0f9eb; color: #52c41a; font-weight: bold;"
        return ""
    
    def _color_signal_source(self, val: str) -> str:
        """Apply color coding for signal sources."""
        colors = {
            "Acceleration magnitude": "background-color: #fef7e0; color: #fa8c16;",
            "Gyroscope magnitude": "background-color: #f6ffed; color: #73d13d;",
            "Pitch angle": "background-color: #f9f0ff; color: #9254de;",
            "Roll angle": "background-color: #e6fffb; color: #13c2c2;",
            "Gyroscope Z-axis": "background-color: #fff7e6; color: #fa541c;"
        }
        return colors.get(val, "")
    
    def render_all_features_tab(self, features_df: Optional[pd.DataFrame] = None):
        """Render the 'All Features' tab - displays all 16 features without filtering."""
        st.markdown("### 📋 Complete Feature List (16 Features)")
        
        # Just display all features without any filtering
        display_df = self.feature_df.copy()
        
        # Display count
        st.markdown(f"**Total Features: {len(display_df)}**")
        
        # Apply styling
        styled_df = display_df.style.applymap(self._color_feature_type, subset=["Type"])
        styled_df = styled_df.applymap(self._color_signal_source, subset=["Signal Source"])
        
        # Display the styled dataframe
        st.dataframe(
            styled_df,
            column_config={
                "Feature": st.column_config.TextColumn(
                    "Feature", 
                    width="small",
                    help="Feature name as used in the dataset"
                ),
                "Type": st.column_config.TextColumn(
                    "Type", 
                    width="small",
                    help="Motion: intensity/energy features | Shape: pattern/distribution features"
                ),
                "Signal Source": st.column_config.TextColumn(
                    "Signal Source", 
                    width="medium",
                    help="Which sensor or calculated signal this feature comes from"
                ),
                "Meaning": st.column_config.TextColumn(
                    "Meaning", 
                    width="large",
                    help="What this feature measures mathematically"
                ),
                "Why Useful": st.column_config.TextColumn(
                    "Why Useful", 
                    width="large",
                    help="How this feature helps distinguish swimming strokes"
                )
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Summary statistics
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Features", len(display_df))
        with col2:
            motion_count = len(display_df[display_df["Type"] == "Motion"])
            st.metric("Motion Features", motion_count)
        with col3:
            shape_count = len(display_df[display_df["Type"] == "Shape"])
            st.metric("Shape Features", shape_count)
        with col4:
            signal_sources = display_df["Signal Source"].nunique()
            st.metric("Signal Sources", signal_sources)
    
    def render_stroke_specific_tab(self):
        """Render the 'Stroke-Specific' tab."""
        st.markdown("### 🎯 Which Features Matter Most for Each Stroke")
        
        stroke_features = {
            "Butterfly 🦋": {
                "Type": "Powerful, undulating",
                "Most Important": [
                    {"feature": "acc_rms", "type": "Motion", "source": "Acceleration magnitude", "reason": "High energy from dolphin kicks"},
                    {"feature": "acc_p2p", "type": "Motion", "source": "Acceleration magnitude", "reason": "Large acceleration variations"},
                    {"feature": "pitch_kurtosis", "type": "Shape", "source": "Pitch angle", "reason": "Sharp pitch changes during undulation"}
                ],
                "Typical Values": "High acc_rms (> 2.5), High pitch_kurtosis (> 3)",
                "Key Insight": "The most acceleration-intensive stroke with sharp body undulations"
            },
            "Backstroke 🔄": {
                "Type": "Asymmetric, rotating",
                "Most Important": [
                    {"feature": "roll_asymmetry", "type": "Shape", "source": "Roll angle", "reason": "Uneven body roll from alternating arms"},
                    {"feature": "gyro_mean", "type": "Motion", "source": "Gyroscope magnitude", "reason": "Continuous body rotation"},
                    {"feature": "stroke_frequency", "type": "Shape", "source": "Gyroscope Z-axis", "reason": "Consistent arm cycle timing"}
                ],
                "Typical Values": "High roll_asymmetry (≠ 0), Moderate gyro_mean (1-2 rad/s)",
                "Key Insight": "Characterized by asymmetric body roll and steady rotation"
            },
            "Breaststroke 🐸": {
                "Type": "Symmetric, rhythmic",
                "Most Important": [
                    {"feature": "stroke_rhythm_cv", "type": "Shape", "source": "Gyroscope Z-axis", "reason": "Very regular stroke timing"},
                    {"feature": "gyro_std", "type": "Motion", "source": "Gyroscope magnitude", "reason": "Low rotation variability"},
                    {"feature": "pitch_peak_count", "type": "Shape", "source": "Pitch angle", "reason": "Distinct pitch peaks per stroke cycle"}
                ],
                "Typical Values": "Low stroke_rhythm_cv (< 0.3), Low gyro_std (< 0.5 rad/s)",
                "Key Insight": "The most rhythmically consistent stroke with symmetric motion"
            },
            "Freestyle 🏊‍♂️": {
                "Type": "Continuous, rotating",
                "Most Important": [
                    {"feature": "gyro_rms", "type": "Motion", "source": "Gyroscope magnitude", "reason": "High rotation energy from body roll"},
                    {"feature": "stroke_frequency", "type": "Shape", "source": "Gyroscope Z-axis", "reason": "Higher stroke rate than other strokes"},
                    {"feature": "acc_mean", "type": "Motion", "source": "Acceleration magnitude", "reason": "Continuous moderate acceleration"}
                ],
                "Typical Values": "High gyro_rms (> 1.5 rad/s), High stroke_frequency (> 0.8 Hz)",
                "Key Insight": "Characterized by continuous rotation and highest stroke rate"
            }
        }
        
        for stroke, info in stroke_features.items():
            with st.expander(f"{stroke} - {info['Type']}", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🎯 Key Features:**")
                    for feat_info in info["Most Important"]:
                        # Get full description
                        full_desc = next((item for item in self.feature_info if item["Feature"] == feat_info["feature"]), {})
                        
                        with st.container():
                            st.markdown(f"##### `{feat_info['feature']}`")
                            st.markdown(f"**Type:** `{feat_info['type']}` | **Source:** {feat_info['source']}")
                            st.markdown(f"*{feat_info['reason']}*")
                            if full_desc:
                                with st.expander("📖 More details", expanded=False):
                                    st.markdown(f"**Meaning:** {full_desc.get('Meaning', '')}")
                                    st.markdown(f"**Why Useful:** {full_desc.get('Why Useful', '')}")
                
                with col2:
                    st.markdown("**📊 Stroke Characteristics:**")
                    st.info(f"**Typical Values:** {info['Typical Values']}")
                    st.success(f"**Key Insight:** {info['Key Insight']}")
                    
                    # Add feature type distribution for this stroke
                    motion_count = sum(1 for f in info["Most Important"] if f["type"] == "Motion")
                    shape_count = sum(1 for f in info["Most Important"] if f["type"] == "Shape")
                    
                    st.markdown("**📈 Feature Type Distribution:**")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Motion Features", motion_count)
                    with col_b:
                        st.metric("Shape Features", shape_count)
                    
                    # Signal sources used
                    sources = set(f["source"] for f in info["Most Important"])
                    st.markdown(f"**📡 Signal Sources Used:** {', '.join(sources)}")
    
    def render_feature_explorer_tab(self, features_df: Optional[pd.DataFrame] = None):
        """Render the 'Feature Explorer' tab."""
        st.markdown("### 🔍 Explore Individual Features")
        
        # Enhanced selector with grouping
        feature_options = []
        for feature in self.feature_info:
            feature_options.append({
                "label": f"{feature['Feature']} ({feature['Type']} - {feature['Signal Source']})",
                "value": feature['Feature']
            })
        
        selected_feature_label = st.selectbox(
            "Choose a feature to explore in detail:",
            options=[opt["label"] for opt in feature_options],
            index=0,
            help="Browse through all 16 features with their type and source information",
            key="feature_explorer_select"
        )
        
        if selected_feature_label:
            # Extract feature name from the selected label
            selected_feature = selected_feature_label.split(" (")[0]
            feature_data = next((f for f in self.feature_info if f["Feature"] == selected_feature), None)
            
            if feature_data:
                self._render_feature_detail(feature_data, features_df)
    
    def _render_feature_detail(self, feature_data: Dict, features_df: Optional[pd.DataFrame] = None):
        """Render detailed view for a single feature."""
        st.markdown("---")
        
        # Header with color-coded badges
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"## `{feature_data['Feature']}`")
        with col2:
            type_color = "blue" if feature_data["Type"] == "Motion" else "green"
            st.markdown(f"<span style='background-color:{type_color}; color:white; padding:5px 10px; border-radius:5px; font-weight:bold;'>{feature_data['Type']}</span>", 
                       unsafe_allow_html=True)
        with col3:
            source_color = {
                "Acceleration magnitude": "orange",
                "Gyroscope magnitude": "green", 
                "Pitch angle": "purple",
                "Roll angle": "teal",
                "Gyroscope Z-axis": "red"
            }.get(feature_data["Signal Source"], "gray")
            st.markdown(f"<span style='background-color:{source_color}; color:white; padding:5px 10px; border-radius:5px;'>{feature_data['Signal Source']}</span>", 
                       unsafe_allow_html=True)
        
        # Main content
        col1, col2 = st.columns([1, 2])
        with col1:
            self._render_feature_statistics(feature_data, features_df)
        
        with col2:
            self._render_feature_description(feature_data)
        
        # Quick actions row
    
    def _render_feature_statistics(self, feature_data: Dict, features_df: Optional[pd.DataFrame] = None):
        """Render feature statistics card."""
        st.markdown("### 📊 Feature Statistics")
        
        if features_df is not None and feature_data['Feature'] in features_df.columns:
            feat_stats = features_df[feature_data['Feature']].describe()
            
            st.metric("Mean", f"{feat_stats['mean']:.3f}", 
                     delta="High" if feat_stats['mean'] > features_df[feature_data['Feature']].median() else "Low")
            st.metric("Std Dev", f"{feat_stats['std']:.3f}", 
                     delta="Variable" if feat_stats['std'] > features_df[feature_data['Feature']].std().mean() else "Stable")
            st.metric("Min", f"{feat_stats['min']:.3f}")
            st.metric("Max", f"{feat_stats['max']:.3f}")
            st.metric("25th Percentile", f"{feat_stats['25%']:.3f}")
            st.metric("75th Percentile", f"{feat_stats['75%']:.3f}")
            
            # Variability indicator
            cv = feat_stats['std'] / feat_stats['mean'] if feat_stats['mean'] != 0 else 0
            st.progress(min(cv, 1.0), text=f"Variability (CV): {cv:.2f}")
        else:
            st.info("📝 Process data to see statistics")
        
        # Quick facts
        st.markdown("### ⚡ Quick Facts")
        st.markdown(f"**Feature Type:** {feature_data['Type']}")
        st.markdown(f"**Signal Source:** {feature_data['Signal Source']}")
        
        # Find which strokes use this feature
        related_strokes = self._get_related_strokes(feature_data['Feature'])
        if related_strokes:
            st.markdown(f"**Key for:** {', '.join(related_strokes)}")
    
    def _get_related_strokes(self, feature_name: str) -> List[str]:
        """Get strokes that use this feature as important."""
        stroke_features = {
            "Butterfly 🦋": ["acc_rms", "acc_p2p", "pitch_kurtosis"],
            "Backstroke 🔄": ["roll_asymmetry", "gyro_mean", "stroke_frequency"],
            "Breaststroke 🐸": ["stroke_rhythm_cv", "gyro_std", "pitch_peak_count"],
            "Freestyle 🏊‍♂️": ["gyro_rms", "stroke_frequency", "acc_mean"]
        }
        
        related_strokes = []
        for stroke, features in stroke_features.items():
            if feature_name in features:
                related_strokes.append(stroke.split(" ")[0])  # Remove emoji
        
        return related_strokes
    
    def _render_feature_description(self, feature_data: Dict):
        """Render feature description and technical details."""
        # Detailed description
        st.markdown("### 📖 Detailed Description")
        
        # Meaning with explanation
        st.markdown("**What it measures:**")
        with st.container():
            st.info(feature_data['Meaning'])
        
        # Why useful with context
        st.markdown("**Why it's useful for swimming:**")
        with st.container():
            st.success(feature_data['Why Useful'])
        
        # Technical details
        st.markdown("### 🔧 Technical Details")
        
        # Calculate or statistical method
        calc_methods = {
            "acc_mean": "Simple average of acceleration magnitude",
            "acc_rms": "Root mean square: sqrt(mean(x²))",
            "acc_std": "Standard deviation",
            "acc_p2p": "Peak-to-peak: max(x) - min(x)",
            "gyro_mean": "Average of gyroscope magnitude", 
            "gyro_rms": "Root mean square of rotation",
            "gyro_std": "Standard deviation of rotation",
            "gyro_p2p": "Rotation range",
            "pitch_kurtosis": "Fourth standardized moment (tail heaviness)",
            "pitch_skewness": "Third standardized moment (asymmetry)",
            "pitch_peak_count": "Count of local maxima",
            "roll_asymmetry": "Difference between positive and negative roll means",
            "stroke_frequency": "Peaks per second in gyro Z",
            "stroke_rhythm_cv": "Coefficient of variation of peak intervals",
            "gyro_kurtosis": "Kurtosis of rotation magnitude",
            "gyro_skewness": "Skewness of rotation magnitude"
        }
        
        st.markdown(f"**Calculation:** {calc_methods.get(feature_data['Feature'], 'Statistical measure')}")
        
        # Units if applicable
        units = {
            "acc_": "g (9.81 m/s²)",
            "gyro_": "rad/s",
            "pitch_": "degrees",
            "roll_": "degrees", 
            "stroke_frequency": "Hz (strokes per second)",
            "stroke_rhythm_cv": "unitless (ratio)"
        }
        
        unit = next((v for k, v in units.items() if feature_data['Feature'].startswith(k)), "unitless")
        st.markdown(f"**Units:** {unit}")
        
        # Visualization guide
        st.markdown("### 📊 Visualization Guide")
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            if feature_data['Type'] == "Motion":
                st.markdown("**Primary View:**")
                st.markdown("'Motion Features Explorer' → 'Feature Trends' tab")
                st.markdown("**Also See:**")
                st.markdown("- 'Motion Energy Dashboard'")
                st.markdown("- 'Stroke Comparison' tab")
            else:
                st.markdown("**Primary View:**")
                st.markdown("'Shape Feature Explorer' → 'Feature Trends' tab")
                st.markdown("**Also See:**")
                st.markdown("- 'Distribution' tab for box plots")
                st.markdown("- 'Feature Table' for statistics")
        
        with col_v2:
            # Related features
            related_features = []
            same_type = [f for f in self.feature_info if f['Type'] == feature_data['Type'] and f['Feature'] != feature_data['Feature']]
            same_source = [f for f in self.feature_info if f['Signal Source'] == feature_data['Signal Source'] and f['Feature'] != feature_data['Feature']]
            
            st.markdown("**Related Features:**")
            if same_type[:3]:
                st.markdown(f"*Same type:* {', '.join([f['Feature'] for f in same_type[:3]])}")
            if same_source[:3]:
                st.markdown(f"*Same source:* {', '.join([f['Feature'] for f in same_source[:3]])}")
    
    
    def render_dashboard(self, features_df: Optional[pd.DataFrame] = None):
        """Render the complete feature description dashboard."""
        st.markdown("---")
        st.header("📘 Feature Description Dashboard")
        
        # Create tabs for different views
        feat_tab1, feat_tab2, feat_tab3 = st.tabs([
            "📋 All Features", 
            "🏊‍♂️ Stroke-Specific", 
            "🔍 Feature Explorer"
        ])
        
        with feat_tab1:
            self.render_all_features_tab(features_df)
        
        with feat_tab2:
            self.render_stroke_specific_tab()
        
        with feat_tab3:
            self.render_feature_explorer_tab(features_df)