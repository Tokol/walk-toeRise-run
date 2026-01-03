import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st


class FeatureRelationships:
    """
    Class for creating activity recognition feature relationship visualizations.
    
    This class generates various 2D scatter plots showing relationships between
    extracted features for activity recognition.
    """
    
    def __init__(self, features_df, activity_colors=None, activity_names=None):
        """
        Initialize the Feature Relationships visualizer.
        
        Parameters:
        -----------
        features_df : pandas DataFrame
            DataFrame containing extracted features with 'activity' column
        activity_colors : dict, optional
            Mapping of activity names to colors for consistent coloring
        activity_names : dict, optional
            Mapping of activity codes to display names
        """
        self.features_df = features_df
        self.activity_colors = activity_colors or {}
        self.activity_names = activity_names or {}
        
        # Convert numpy array to list and check if 'activity' column exists
        if 'activity' in features_df.columns:
            self.activities = list(features_df['activity'].unique())
        else:
            self.activities = []
        
        # Feature descriptions for tooltips
        self.feature_descriptions = {
            # Motion features (4)
            "mean_magnitude": "Average acceleration magnitude in the window",
            "rms_magnitude": "Root Mean Square - measures acceleration energy/power",
            "std_magnitude": "Standard deviation - measures acceleration variability",
            "p2p_magnitude": "Peak-to-peak - range from minimum to maximum acceleration",
            
            # Shape features (5)
            "kurtosis_magnitude": "Tailedness/sharpness of acceleration distribution",
            "skewness_magnitude": "Asymmetry of acceleration distribution",
            "median_magnitude": "Middle value of acceleration distribution",
            "iqr_magnitude": "Interquartile range - spread of middle 50% of data",
            "peak_count": "Number of local maxima in the window"
        }
        
        # Feature relationship descriptions (generic)
        self.feature_relationship_descriptions = {
            ("mean_magnitude", "rms_magnitude"): 
                "Shows the relationship between average acceleration and signal energy",
            ("std_magnitude", "p2p_magnitude"):
                "Compares variability (std dev) with total range (peak-to-peak)",
            ("kurtosis_magnitude", "skewness_magnitude"):
                "Shows the shape of the acceleration distribution",
            ("mean_magnitude", "peak_count"):
                "Compares intensity with frequency of peaks",
            ("rms_magnitude", "peak_count"):
                "Shows energy distribution vs periodicity"
        }
    
    def _get_activity_display_name(self, activity_code):
        """Convert activity code to display name."""
        return self.activity_names.get(activity_code, activity_code.replace('_', ' ').title())
    
    def _get_activity_color(self, activity_code):
        """Get color for an activity code."""
        return self.activity_colors.get(activity_code, 'gray')
    
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
            color_discrete_map=self.activity_colors,
            title=title,
            labels={
                x_feature: x_display,
                y_feature: y_display,
                "activity": "Activity"
            },
            height=height,
            hover_data=["segment_start", "segment_end"],
            opacity=0.7,
            size_max=8
        )
        
        # Add generic feature relationship description
        if (x_feature, y_feature) in self.feature_relationship_descriptions:
            fig.add_annotation(
                x=0.5,
                y=1.05,
                xref="paper",
                yref="paper",
                text=f"💡 {self.feature_relationship_descriptions[(x_feature, y_feature)]}",
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
                title="Activity",
                font=dict(size=10)
            ),
            margin=dict(t=80 if (x_feature, y_feature) in self.feature_relationship_descriptions else 50),
            xaxis_title_font=dict(size=12),
            yaxis_title_font=dict(size=12)
        )
        
        # Add trend lines for each activity
        for activity in self.activities:
            activity_data = self.features_df[self.features_df['activity'] == activity]
            if len(activity_data) > 1:
                try:
                    # Fit a linear regression
                    z = np.polyfit(activity_data[x_feature], activity_data[y_feature], 1)
                    p = np.poly1d(z)
                    
                    # Add trend line
                    x_range = [activity_data[x_feature].min(), activity_data[x_feature].max()]
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=p(x_range),
                        mode='lines',
                        line=dict(color=self._get_activity_color(activity), width=1, dash='dash'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                except:
                    pass
        
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
        
        # Motion feature relationships
        figures["mean_vs_rms"] = self.create_feature_scatter(
            "mean_magnitude", "rms_magnitude",
            title="Mean vs RMS Magnitude (Intensity vs Energy)"
        )
        
        figures["std_vs_p2p"] = self.create_feature_scatter(
            "std_magnitude", "p2p_magnitude",
            title="Std Dev vs Peak-to-Peak (Variability vs Range)"
        )
        
        # Shape feature relationships
        figures["kurtosis_vs_skewness"] = self.create_feature_scatter(
            "kurtosis_magnitude", "skewness_magnitude",
            title="Kurtosis vs Skewness (Distribution Shape)"
        )
        
        figures["median_vs_iqr"] = self.create_feature_scatter(
            "median_magnitude", "iqr_magnitude",
            title="Median vs IQR (Central Tendency vs Spread)"
        )
        
        # Motion vs Shape feature relationships
        figures["mean_vs_kurtosis"] = self.create_feature_scatter(
            "mean_magnitude", "kurtosis_magnitude",
            title="Mean vs Kurtosis (Average vs Tailedness)"
        )
        
        figures["rms_vs_peak_count"] = self.create_feature_scatter(
            "rms_magnitude", "peak_count",
            title="RMS vs Peak Count (Energy vs Periodicity)"
        )
        
        figures["std_vs_skewness"] = self.create_feature_scatter(
            "std_magnitude", "skewness_magnitude",
            title="Std Dev vs Skewness (Variability vs Asymmetry)"
        )
        
        figures["p2p_vs_median"] = self.create_feature_scatter(
            "p2p_magnitude", "median_magnitude",
            title="Peak-to-Peak vs Median (Range vs Central Value)"
        )
        
        # Additional useful combinations
        figures["mean_vs_peak_count"] = self.create_feature_scatter(
            "mean_magnitude", "peak_count",
            title="Mean vs Peak Count (Intensity vs Frequency)"
        )
        
        figures["std_vs_iqr"] = self.create_feature_scatter(
            "std_magnitude", "iqr_magnitude",
            title="Std Dev vs IQR (Two Measures of Spread)"
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
        motion_features = ["mean_magnitude", "rms_magnitude", "std_magnitude", "p2p_magnitude"]
        shape_features = ["kurtosis_magnitude", "skewness_magnitude", "median_magnitude", 
                         "iqr_magnitude", "peak_count"]
        
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
            colorbar=dict(title="Correlation"),
            hoverinfo="x+y+z"
        ))
        
        fig.update_layout(
            title="Feature Correlation Matrix",
            height=500,
            width=600,
            xaxis_title="Features",
            yaxis_title="Features",
            margin=dict(b=80, l=80, r=50),
            xaxis=dict(tickangle=45),
            yaxis=dict(tickangle=0)
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
            "🎯 Key Patterns", 
            "🏃 Activity Analysis",
            "📈 Correlation Matrix"
        ])
        
        with tab1:
            st.markdown("### Complete Feature Relationships")
            st.info("""
            **How to read these plots:**
            - Each point = one time segment (e.g., 2-second window)
            - Colors = different activities
            - Clusters = segments with similar movement patterns
            - Dashed lines = trend lines for each activity
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
                            st.plotly_chart(
                                fig, 
                                use_container_width=True,
                                key=f"fig_{figure_keys[i]}_{i}"
                            )
                
                with col2:
                    if i + 1 < len(figure_keys):
                        fig = figures[figure_keys[i + 1]]
                        if fig.data:
                            st.plotly_chart(
                                fig, 
                                use_container_width=True,
                                key=f"fig_{figure_keys[i+1]}_{i+1}"
                            )
        
        with tab2:
            st.markdown("### Key Feature Relationships")
            st.info("""
            **These plots reveal key movement characteristics:**
            1. **Mean vs RMS** - Shows intensity and energy of movement
            2. **Std Dev vs Peak-to-Peak** - Shows variability and range
            3. **Kurtosis vs Skewness** - Shows distribution shape
            4. **Mean vs Peak Count** - Shows intensity vs frequency
            """)
            
            # Create a simple 2-column layout for key patterns
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = self.create_feature_scatter(
                    "mean_magnitude", "rms_magnitude",
                    title="Intensity vs Energy"
                )
                if fig1.data:
                    st.plotly_chart(fig1, use_container_width=True, key="intensity_vs_energy")
            
            with col2:
                fig2 = self.create_feature_scatter(
                    "std_magnitude", "p2p_magnitude",
                    title="Variability vs Range"
                )
                if fig2.data:
                    st.plotly_chart(fig2, use_container_width=True, key="variability_vs_range")
            
            # Add second row
            col3, col4 = st.columns(2)
            
            with col3:
                fig3 = self.create_feature_scatter(
                    "kurtosis_magnitude", "skewness_magnitude",
                    title="Distribution Shape"
                )
                if fig3.data:
                    st.plotly_chart(fig3, use_container_width=True, key="distribution_shape")
            
            with col4:
                fig4 = self.create_feature_scatter(
                    "mean_magnitude", "peak_count",
                    title="Intensity vs Frequency"
                )
                if fig4.data:
                    st.plotly_chart(fig4, use_container_width=True, key="intensity_vs_frequency")
        
        with tab3:
            st.markdown("### Activity-Specific Analysis")
            
            # Check if activities list is not empty
            if self.activities:
                selected_activity = st.selectbox(
                    "Select activity to analyze:",
                    [self._get_activity_display_name(a) for a in self.activities],
                    key="activity_selector_feature"
                )
                
                # Find activity code from display name
                activity_code = None
                for code in self.activities:
                    if self._get_activity_display_name(code) == selected_activity:
                        activity_code = code
                        break
                
                if activity_code:
                    activity_df = self.features_df[self.features_df['activity'] == activity_code]
                    
                    if len(activity_df) > 0:
                        # Create columns for different views
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Feature distribution
                            fig_dist = self._create_activity_distribution(activity_df, selected_activity)
                            if fig_dist.data:
                                st.plotly_chart(
                                    fig_dist, 
                                    use_container_width=True,
                                    key=f"activity_dist_{activity_code}"
                                )
                        
                        with col2:
                            # Statistics table
                            stats_features = ["mean_magnitude", "rms_magnitude", "std_magnitude", 
                                            "p2p_magnitude", "peak_count", "kurtosis_magnitude"]
                            stats_features = [f for f in stats_features if f in activity_df.columns]
                            
                            if stats_features:
                                stats_df = activity_df[stats_features].describe().round(3)
                                st.dataframe(stats_df, use_container_width=True, key=f"stats_table_{activity_code}")
                                
                                # Activity summary
                                st.markdown("**Activity Summary:**")
                                
                                # Calculate key characteristics
                                if "mean_magnitude" in activity_df.columns:
                                    avg_mean = activity_df["mean_magnitude"].mean()
                                    st.write(f"• Average Intensity: **{avg_mean:.3f} m/s²**")
                                
                                if "std_magnitude" in activity_df.columns:
                                    avg_std = activity_df["std_magnitude"].mean()
                                    st.write(f"• Variability: **{avg_std:.3f} m/s²**")
                                
                                if "peak_count" in activity_df.columns:
                                    avg_peaks = activity_df["peak_count"].mean()
                                    st.write(f"• Peak Frequency: **{avg_peaks:.1f} peaks/segment**")
                                
                                if "kurtosis_magnitude" in activity_df.columns:
                                    avg_kurt = activity_df["kurtosis_magnitude"].mean()
                                    st.write(f"• Distribution Tailedness: **{avg_kurt:.3f}**")
                                    
                                    # Add interpretation
                                    if avg_kurt > 3:
                                        kurt_interpret = "Leptokurtic (peaked, heavy tails)"
                                    elif avg_kurt < 3:
                                        kurt_interpret = "Platykurtic (flat, light tails)"
                                    else:
                                        kurt_interpret = "Mesokurtic (normal-like)"
                                    
                                    st.write(f"• Kurtosis Type: **{kurt_interpret}**")
                    else:
                        st.warning(f"No data found for activity: {selected_activity}")
            else:
                st.warning("No activities found in the data.")
        
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
                available_features = ["mean_magnitude", "rms_magnitude", "std_magnitude", 
                                    "p2p_magnitude", "peak_count"]
                available_features = [f for f in available_features if f in self.features_df.columns]
                
                if len(available_features) >= 2:
                    corr_matrix = self.features_df[available_features].corr().abs()
                    
                    high_corr_threshold = 0.8
                    moderate_corr_threshold = 0.5
                    
                    # Find high correlations
                    high_corr_pairs = []
                    moderate_corr_pairs = []
                    
                    for i in range(len(available_features)):
                        for j in range(i+1, len(available_features)):
                            corr_value = corr_matrix.iloc[i, j]
                            feat1 = available_features[i].replace('_', ' ').title()
                            feat2 = available_features[j].replace('_', ' ').title()
                            
                            if corr_value > high_corr_threshold:
                                high_corr_pairs.append((feat1, feat2, corr_value))
                            elif corr_value > moderate_corr_threshold:
                                moderate_corr_pairs.append((feat1, feat2, corr_value))
                    
                    if high_corr_pairs:
                        st.write("**Highly Correlated Features (>0.8):**")
                        for feat1, feat2, corr in high_corr_pairs[:3]:
                            st.write(f"• {feat1} ↔ {feat2}: {corr:.2f}")
                        st.caption("These features may provide redundant information")
                    
                    if moderate_corr_pairs and not high_corr_pairs:
                        st.write("**Moderately Correlated Features (>0.5):**")
                        for feat1, feat2, corr in moderate_corr_pairs[:3]:
                            st.write(f"• {feat1} ↔ {feat2}: {corr:.2f}")
    
    def _create_activity_distribution(self, activity_df, activity_name):
        """Create distribution plot for a specific activity."""
        # Select features to show
        features = ["mean_magnitude", "std_magnitude", "rms_magnitude", "peak_count"]
        features = [f for f in features if f in activity_df.columns]
        
        if not features:
            return go.Figure()
        
        # Create subplots with descriptive titles that include statistics
        subplot_titles = []
        for f in features:
            mean_val = activity_df[f].mean()
            std_val = activity_df[f].std()
            subplot_titles.append(f"{f.replace('_', ' ').title()}<br>Mean: {mean_val:.3f}, Std: {std_val:.3f}")
        
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=subplot_titles,
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        for i, feature in enumerate(features[:4]):  # Limit to 4 features
            row = i // 2 + 1
            col = i % 2 + 1
            
            fig.add_trace(
                go.Histogram(
                    x=activity_df[feature],
                    name=feature.replace('_', ' ').title(),
                    nbinsx=15,
                    marker_color=self._get_activity_color(activity_df['activity'].iloc[0]),
                    opacity=0.7
                ),
                row=row,
                col=col
            )
            
            # Add mean line
            mean_val = activity_df[feature].mean()
            fig.add_vline(
                x=mean_val,
                line_dash="dash",
                line_color="red",
                row=row,
                col=col
            )
        
        fig.update_layout(
            title=f"{activity_name}: Key Feature Distributions (N={len(activity_df)})",
            height=500,
            showlegend=False
        )
        
        return fig
    
    def _create_activity_radar_chart(self, activity_df, activity_name):
        """Create radar chart showing activity profile."""
        # Normalize features for radar chart
        features = ["mean_magnitude", "rms_magnitude", "std_magnitude", 
                   "p2p_magnitude", "peak_count", "kurtosis_magnitude"]
        features = [f for f in features if f in activity_df.columns and f in self.features_df.columns]
        
        if len(features) < 3:  # Need at least 3 features for radar
            return None
        
        # Calculate mean values
        means = activity_df[features].mean()
        
        # Normalize between 0 and 1
        mins = self.features_df[features].min()
        maxs = self.features_df[features].max()
        
        # Handle cases where max == min to avoid division by zero
        normalized = []
        for i, feat in enumerate(features):
            if maxs.iloc[i] == mins.iloc[i]:
                normalized.append(0.5)  # Middle value if no variation
            else:
                normalized.append((means.iloc[i] - mins.iloc[i]) / (maxs.iloc[i] - mins.iloc[i]))
        
        # Create radar chart
        fig = go.Figure(data=go.Scatterpolar(
            r=normalized,
            theta=[f.replace('_', ' ').title() for f in features],
            fill='toself',
            name=activity_name,
            line_color=self._get_activity_color(activity_df['activity'].iloc[0])
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title=f"{activity_name}: Normalized Feature Profile",
            height=400,
            showlegend=False
        )
        
        return fig
    
    # Alias method for backward compatibility
    def render_feature_scatter_dashboard(self):
        """
        Alias for render_dashboard() for backward compatibility.
        """
        return self.render_dashboard()