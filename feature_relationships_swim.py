import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st


class SwimmingFeatureRelationships:
    """
    Class for creating swimming-specific feature relationship visualizations.
    
    This class generates various 2D scatter plots showing relationships between
    extracted features, which help understand swimming stroke biomechanics.
    """
    
    def __init__(self, features_df, stroke_colors=None, stroke_names=None):
        """
        Initialize the Feature Relationships visualizer.
        
        Parameters:
        -----------
        features_df : pandas DataFrame
            DataFrame containing extracted features with 'activity' column
        stroke_colors : dict, optional
            Mapping of stroke names to colors for consistent coloring
        stroke_names : dict, optional
            Mapping of stroke codes to display names
        """
        self.features_df = features_df
        self.stroke_colors = stroke_colors or {}
        self.stroke_names = stroke_names or {}
        self.strokes = features_df['activity'].unique() if 'activity' in features_df.columns else []
        
        # Feature descriptions for tooltips
        self.feature_descriptions = {
            # Motion features (8)
            "acc_mean": "Average acceleration magnitude in the window",
            "acc_rms": "Root Mean Square - measures acceleration energy/power",
            "acc_std": "Standard deviation - measures acceleration variability",
            "acc_p2p": "Peak-to-peak - range from minimum to maximum acceleration",
            "gyro_mean": "Average gyroscope magnitude in the window",
            "gyro_rms": "Root Mean Square - measures rotation energy/power",
            "gyro_std": "Standard deviation - measures rotation variability",
            "gyro_p2p": "Peak-to-peak - range from minimum to maximum rotation",
            
            # Shape features (8)
            "pitch_kurtosis": "Tailedness/sharpness of pitch distribution",
            "pitch_skewness": "Asymmetry of pitch movement",
            "pitch_peak_count": "Number of pitch peaks in the window",
            "roll_asymmetry": "Difference between left/right body roll",
            "stroke_frequency": "Estimated stroke cycles per second",
            "stroke_rhythm_cv": "Consistency of stroke timing (lower = more regular)",
            "gyro_kurtosis": "Tailedness/sharpness of rotation distribution",
            "gyro_skewness": "Asymmetry of rotation patterns"
        }
        
        # Swimming biomechanics insights
        self.biomechanics_insights = {
            ("stroke_frequency", "roll_asymmetry"): 
                "Butterfly: High frequency, symmetric roll | Breaststroke: Lower frequency, asymmetric",
            ("pitch_skewness", "gyro_kurtosis"):
                "Backstroke: Negative skew (arching), high kurtosis | Freestyle: Balanced profile",
            ("acc_p2p", "stroke_rhythm_cv"):
                "Breaststroke: High P2P (explosive), low CV (rhythmic) | Freestyle: Moderate both",
            ("pitch_peak_count", "gyro_skewness"):
                "Butterfly: High peaks, symmetric | Backstroke: Few peaks, asymmetric",
            ("acc_rms", "gyro_rms"):
                "Shows energy distribution: Linear vs Rotational energy balance"
        }
    
    def _get_stroke_display_name(self, stroke_code):
        """Convert stroke code to display name."""
        return self.stroke_names.get(stroke_code, stroke_code.replace('_', ' ').title())
    
    def _get_stroke_color(self, stroke_code):
        """Get color for a stroke code."""
        return self.stroke_colors.get(stroke_code, 'gray')
    
    def create_feature_scatter(self, x_feature, y_feature, title=None, height=400):
        """
        Create a scatter plot between two features.
        
        Parameters:
        -----------
        x_feature : str
            Feature name for x-axis
        y_feature : str
            Feature name for y-axis
        title : str, optional
            Custom title for the plot
        height : int
            Plot height in pixels
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Scatter plot figure
        """
        if x_feature not in self.features_df.columns or y_feature not in self.features_df.columns:
            st.warning(f"Features {x_feature} or {y_feature} not found in dataframe")
            return go.Figure()
        
        # Create display names
        x_display = x_feature.replace('_', ' ').title()
        y_display = y_feature.replace('_', ' ').title()
        
        # Create unique title if not provided
        if title is None:
            title = f"{y_display} vs {x_display}"
        
        # Create figure
        fig = px.scatter(
            self.features_df,
            x=x_feature,
            y=y_feature,
            color="activity",
            color_discrete_map=self.stroke_colors,
            title=title,
            labels={
                x_feature: x_display,
                y_feature: y_display,
                "activity": "Stroke"
            },
            height=height,
            hover_data=["segment_start", "segment_end"]
        )
        
        # Add biomechanics insight if available
        if (x_feature, y_feature) in self.biomechanics_insights:
            fig.add_annotation(
                x=0.5,
                y=1.05,
                xref="paper",
                yref="paper",
                text=f"💡 {self.biomechanics_insights[(x_feature, y_feature)]}",
                showarrow=False,
                font=dict(size=11, color="gray"),
                align="center"
            )
        
        # Update layout for better readability
        fig.update_layout(
            hovermode="closest",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                title="Stroke"
            ),
            margin=dict(t=80 if (x_feature, y_feature) in self.biomechanics_insights else 50)
        )
        
        return fig
    
    def create_all_relationships_dashboard(self):
        """
        Create a complete dashboard with all feature relationship plots.
        
        Returns:
        --------
        dict
            Dictionary of plotly figures with descriptive keys
        """
        figures = {}
        
        # 1. Original: Acceleration Mean vs RMS
        figures["acc_mean_vs_rms"] = self.create_feature_scatter(
            "acc_mean", "acc_rms",
            title="Acceleration Mean vs RMS (Motion Energy)"
        )
        
        # 2. Original: Gyro Mean vs Std Dev
        figures["gyro_mean_vs_std"] = self.create_feature_scatter(
            "gyro_mean", "gyro_std",
            title="Gyroscope Mean vs Std Dev (Rotation Variability)"
        )
        
        # 3. NEW: Stroke Frequency vs Roll Asymmetry
        figures["freq_vs_roll_asym"] = self.create_feature_scatter(
            "stroke_frequency", "roll_asymmetry",
            title="Stroke Frequency vs Roll Asymmetry (Cadence vs Balance)"
        )
        
        # 4. NEW: Pitch Skewness vs Gyro Kurtosis
        figures["pitch_skew_vs_gyro_kurt"] = self.create_feature_scatter(
            "pitch_skewness", "gyro_kurtosis",
            title="Pitch Skewness vs Gyro Kurtosis (Angle vs Rotation Sharpness)"
        )
        
        # 5. NEW: Acceleration P2P vs Stroke Rhythm CV
        figures["acc_p2p_vs_rhythm_cv"] = self.create_feature_scatter(
            "acc_p2p", "stroke_rhythm_cv",
            title="Acceleration Range vs Stroke Rhythm CV (Power vs Consistency)"
        )
        
        # 6. NEW: Pitch Peak Count vs Gyro Skewness
        figures["pitch_peaks_vs_gyro_skew"] = self.create_feature_scatter(
            "pitch_peak_count", "gyro_skewness",
            title="Pitch Peaks vs Gyro Skewness (Undulations vs Rotation Asymmetry)"
        )
        
        # 7. NEW: Roll Asymmetry vs Stroke Frequency (alternative view)
        figures["roll_asym_vs_freq"] = self.create_feature_scatter(
            "roll_asymmetry", "stroke_frequency",
            title="Roll Asymmetry vs Stroke Frequency (Balance vs Cadence)"
        )
        
        # 8. NEW: Acceleration RMS vs Gyro RMS
        figures["acc_rms_vs_gyro_rms"] = self.create_feature_scatter(
            "acc_rms", "gyro_rms",
            title="Acceleration RMS vs Gyro RMS (Linear vs Rotational Energy)"
        )
        
        
        
        return figures
    
    def create_feature_correlation_heatmap(self):
        """
        Create a correlation heatmap between motion and shape features.
        
        Returns:
        --------
        plotly.graph_objects.Figure
            Correlation heatmap figure
        """
        # Select motion and shape features
        motion_features = ["acc_mean", "acc_rms", "acc_std", "acc_p2p",
                          "gyro_mean", "gyro_rms", "gyro_std", "gyro_p2p"]
        
        shape_features = ["pitch_kurtosis", "pitch_skewness", "pitch_peak_count",
                         "roll_asymmetry", "stroke_frequency", "stroke_rhythm_cv",
                         "gyro_kurtosis", "gyro_skewness"]
        
        # Check which features exist
        motion_features = [f for f in motion_features if f in self.features_df.columns]
        shape_features = [f for f in shape_features if f in self.features_df.columns]
        
        if not motion_features or not shape_features:
            return go.Figure()
        
        # Calculate correlation matrix
        all_features = motion_features + shape_features
        corr_matrix = self.features_df[all_features].corr().round(2)
        
        # Create feature type labels
        feature_types = []
        for feat in all_features:
            if feat in motion_features:
                feature_types.append("Motion")
            else:
                feature_types.append("Shape")
        
        # Display names
        display_names = [f.replace('_', ' ').title() for f in all_features]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=display_names,
            y=display_names,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            hoverongaps=False,
            colorbar=dict(title="Correlation")
        ))
        
        # Add feature type annotations
        for i, feat_type in enumerate(feature_types):
            fig.add_annotation(
                x=i,
                y=-1,
                xref="x",
                yref="y",
                text=feat_type,
                showarrow=False,
                font=dict(size=10, color="black"),
                bgcolor="lightgray" if feat_type == "Motion" else "lightblue"
            )
            
            fig.add_annotation(
                x=-1,
                y=i,
                xref="x",
                yref="y",
                text=feat_type,
                showarrow=False,
                font=dict(size=10, color="black"),
                bgcolor="lightgray" if feat_type == "Motion" else "lightblue"
            )
        
        fig.update_layout(
            title="Feature Correlation Matrix (Motion vs Shape Features)",
            height=500,
            xaxis_title="Features",
            yaxis_title="Features",
            margin=dict(b=80, l=80),
            xaxis=dict(tickangle=45),
            yaxis=dict(tickangle=0)
        )
        
        return fig
    
    def create_stroke_specific_insights(self):
        """
        Create specialized insights for each swimming stroke.
        
        Returns:
        --------
        dict
            Dictionary of figures with stroke-specific insights
        """
        stroke_insights = {}
        
        for stroke in self.strokes:
            stroke_df = self.features_df[self.features_df['activity'] == stroke]
            display_name = self._get_stroke_display_name(stroke)
            
            if len(stroke_df) < 3:  # Need minimum data
                continue
            
            # Create stroke-specific scatter matrix
            fig = self._create_stroke_scatter_matrix(stroke_df, display_name)
            stroke_insights[f"{stroke}_scatter_matrix"] = fig
            
            # Create stroke feature distribution
            fig_dist = self._create_stroke_distribution(stroke_df, display_name)
            stroke_insights[f"{stroke}_distribution"] = fig_dist
        
        return stroke_insights
    
    def _create_stroke_scatter_matrix(self, stroke_df, stroke_name):
        """Create scatter matrix for a specific stroke."""
        # Select key features for the matrix
        key_features = ["stroke_frequency", "roll_asymmetry", "acc_rms", "gyro_rms"]
        key_features = [f for f in key_features if f in stroke_df.columns]
        
        if len(key_features) < 2:
            return go.Figure()
        
        fig = px.scatter_matrix(
            stroke_df,
            dimensions=key_features,
            title=f"{stroke_name}: Feature Relationships",
            labels={col: col.replace('_', ' ').title() for col in key_features},
            height=500
        )
        
        fig.update_traces(diagonal_visible=False)
        fig.update_layout(
            hovermode="closest",
            showlegend=False
        )
        
        return fig
    
    def _create_stroke_distribution(self, stroke_df, stroke_name):
        """Create distribution plot for a specific stroke."""
        # Select features to show
        features = ["stroke_frequency", "roll_asymmetry", "acc_rms", "gyro_rms"]
        features = [f for f in features if f in stroke_df.columns]
        
        if not features:
            return go.Figure()
        
        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[f.replace('_', ' ').title() for f in features],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        for i, feature in enumerate(features[:4]):  # Limit to 4 features
            row = i // 2 + 1
            col = i % 2 + 1
            
            fig.add_trace(
                go.Histogram(
                    x=stroke_df[feature],
                    name=feature.replace('_', ' ').title(),
                    nbinsx=20,
                    marker_color=self._get_stroke_color(stroke_df['activity'].iloc[0])
                ),
                row=row,
                col=col
            )
            
            # Add mean line
            mean_val = stroke_df[feature].mean()
            fig.add_vline(
                x=mean_val,
                line_dash="dash",
                line_color="red",
                row=row,
                col=col
            )
            
            # Add annotation for mean
            fig.add_annotation(
                x=mean_val,
                y=0.9,
                xref=f"x{i+1}",
                yref=f"y{i+1}",
                text=f"Mean: {mean_val:.3f}",
                showarrow=False,
                font=dict(size=10, color="red"),
                bgcolor="white"
            )
        
        fig.update_layout(
            title=f"{stroke_name}: Key Feature Distributions",
            height=500,
            showlegend=False
        )
        
        return fig
    
    def render_dashboard(self):
        """
        Render complete feature relationships dashboard in Streamlit.
        
        This is the main method to call from your Streamlit app.
        """
        st.markdown("---")
        st.subheader("🔍 Feature Relationships Explorer")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 All Relationships", 
            "🎯 Key Biomechanics", 
            "🏊‍♂️ Stroke Analysis",
            "📈 Feature Matrix"
        ])
        
        with tab1:
            st.markdown("### Complete Feature Relationships")
            st.info("""
            **How to read these plots:**
            - Each point = one 2-second segment of swimming
            - Colors = different swimming strokes
            - Clusters = segments with similar biomechanical patterns
            - Outliers = unusual technique or data artifacts
            """)
            
            # Get all relationship figures
            figures = self.create_all_relationships_dashboard()
            
            # Display in a grid (2 per row)
            figure_keys = list(figures.keys())
            
            for i in range(0, len(figure_keys), 2):
                col1, col2 = st.columns(2)
                
                with col1:
                    if i < len(figure_keys):
                        fig = figures[figure_keys[i]]
                        if fig.data:  # Check if figure has data
                            # Add unique key using the figure key
                            st.plotly_chart(
                                fig, 
                                use_container_width=True,
                                key=f"fig_{figure_keys[i]}_{i}"
                            )
                
                with col2:
                    if i + 1 < len(figure_keys):
                        fig = figures[figure_keys[i + 1]]
                        if fig.data:
                            # Add unique key using the figure key
                            st.plotly_chart(
                                fig, 
                                use_container_width=True,
                                key=f"fig_{figure_keys[i+1]}_{i+1}"
                            )
        
        with tab2:
            st.markdown("### Key Biomechanical Relationships")
            st.info("""
            **These plots reveal swimming technique:**
            1. **Stroke Frequency vs Roll Asymmetry** - Cadence vs bilateral balance
            2. **Pitch Skewness vs Gyro Kurtosis** - Body angle vs rotation sharpness
            3. **Acceleration RMS vs Gyro RMS** - Linear vs rotational energy distribution
            """)
            
            # Select key relationships
            key_figures = {
                "Stroke Cadence & Balance": self.create_feature_scatter(
                    "stroke_frequency", "roll_asymmetry",
                    title="Stroke Frequency vs Roll Asymmetry"
                ),
                "Body Angle Dynamics": self.create_feature_scatter(
                    "pitch_skewness", "gyro_kurtosis",
                    title="Pitch Skewness vs Gyro Kurtosis"
                ),
                "Energy Distribution": self.create_feature_scatter(
                    "acc_rms", "gyro_rms",
                    title="Acceleration Energy vs Rotation Energy"
                )
            }
            
            for idx, (title, fig) in enumerate(key_figures.items()):
                if fig.data:
                    st.markdown(f"#### {title}")
                    # Add unique key for each plot
                    st.plotly_chart(
                        fig, 
                        use_container_width=True,
                        key=f"key_biomech_{idx}_{title.replace(' ', '_')}"
                    )
        
        with tab3:
            st.markdown("### Stroke-Specific Analysis")
            
            # Stroke selector
            selected_stroke = st.selectbox(
                "Select stroke to analyze:",
                [self._get_stroke_display_name(s) for s in self.strokes],
                key="stroke_selector_feature"
            )
            
            # Find stroke code from display name
            stroke_code = None
            for code in self.strokes:
                if self._get_stroke_display_name(code) == selected_stroke:
                    stroke_code = code
                    break
            
            if stroke_code:
                stroke_df = self.features_df[self.features_df['activity'] == stroke_code]
                
                # Create insights for this stroke
                col1, col2 = st.columns(2)
                
                with col1:
                    # Feature distribution
                    fig_dist = self._create_stroke_distribution(stroke_df, selected_stroke)
                    if fig_dist.data:
                        st.plotly_chart(
                            fig_dist, 
                            use_container_width=True,
                            key=f"stroke_dist_{stroke_code}"
                        )
                
                with col2:
                    # Statistics table
                    stats_df = stroke_df[["stroke_frequency", "roll_asymmetry", 
                                          "acc_rms", "gyro_rms"]].describe().round(3)
                    st.dataframe(stats_df, use_container_width=True, key=f"stats_table_{stroke_code}")
                    
                    # Biomechanics summary
                    st.markdown("**Biomechanics Summary:**")
                    
                    # Calculate stroke characteristics
                    avg_freq = stroke_df["stroke_frequency"].mean()
                    avg_roll_asym = stroke_df["roll_asymmetry"].mean()
                    acc_energy = stroke_df["acc_rms"].mean()
                    gyro_energy = stroke_df["gyro_rms"].mean()
                    
                    st.write(f"• Stroke Frequency: **{avg_freq:.2f} Hz**")
                    st.write(f"• Roll Asymmetry: **{abs(avg_roll_asym):.3f}** ({'Right' if avg_roll_asym > 0 else 'Left'} bias)")
                    st.write(f"• Energy Ratio (Acc/Gyro): **{acc_energy/gyro_energy:.2f}**")
        
        with tab4:
            st.markdown("### Feature Correlation Matrix")
            st.info("""
            **Understanding correlations:**
            - **Blue** = Positive correlation (features increase together)
            - **Red** = Negative correlation (one increases, other decreases)
            - **White** = No correlation
            """)
            
            fig = self.create_feature_correlation_heatmap()
            if fig.data:
                st.plotly_chart(
                    fig, 
                    use_container_width=True,
                    key="feature_correlation_heatmap"
                )
                
                # Add interpretation
                st.markdown("**Key Insights from Correlations:**")
                
                # Check for interesting correlations
                corr_threshold = 0.7
                
                # Motion-motion correlations
                motion_features = ["acc_mean", "acc_rms", "acc_std", "acc_p2p",
                                 "gyro_mean", "gyro_rms", "gyro_std", "gyro_p2p"]
                motion_features = [f for f in motion_features if f in self.features_df.columns]
                
                if motion_features:
                    motion_corr = self.features_df[motion_features].corr().abs()
                    high_corr_pairs = []
                    
                    for i in range(len(motion_features)):
                        for j in range(i+1, len(motion_features)):
                            if motion_corr.iloc[i, j] > corr_threshold:
                                high_corr_pairs.append((
                                    motion_features[i].replace('_', ' ').title(),
                                    motion_features[j].replace('_', ' ').title(),
                                    motion_corr.iloc[i, j]
                                ))
                    
                    if high_corr_pairs:
                        st.write("**High Motion Feature Correlations:**")
                        for idx, (feat1, feat2, corr) in enumerate(high_corr_pairs[:3]):  # Show top 3
                            st.write(f"• {feat1} ↔ {feat2}: {corr:.2f}")
    
    # Alias method for backward compatibility
    def render_feature_scatter_dashboard(self):
        """
        Alias for render_dashboard() for backward compatibility.
        """
        return self.render_dashboard()