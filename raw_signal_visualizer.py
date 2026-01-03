import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class SwimmingSignalVisualizer:
    """
    A class to handle comprehensive raw signal visualization for swimming stroke analysis.
    Provides interactive plots with stroke regions, signal filtering, and insights.
    """
    
    def __init__(self, df, features_df, time_ranges, stroke_mode, stroke_names_dict=None):
        """
        Initialize the visualizer with data and configuration.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Raw time series data with IMU signals
        features_df : pandas.DataFrame
            Extracted features dataframe
        time_ranges : dict
            Dictionary containing stroke time ranges
        stroke_mode : str
            Either "Default Strokes" or "Custom Strokes"
        stroke_names_dict : dict, optional
            Mapping of stroke IDs to display names
        """
        self.df = df.copy()
        self.features_df = features_df.copy()
        self.time_ranges = time_ranges
        self.stroke_mode = stroke_mode
        self.stroke_names_dict = stroke_names_dict or {}
        
        # Get time column
        self.time_col = "Time (s)" if "Time (s)" in self.df.columns else "seconds_elapsed"
        
        # Get unique strokes for color mapping
        self.strokes_from_features = list(self.features_df["activity"].unique())
        self.num_strokes = len(self.strokes_from_features)
        
        # Generate colors for swimming strokes
        self.swimming_colors_matplotlib = {
            "butterfly": "tab:blue",
            "backstroke": "tab:green",
            "breaststroke": "tab:orange",
            "freestyle": "tab:red"
        }
        
        self.swimming_colors_plotly = {
            "butterfly": "blue",
            "backstroke": "green",
            "breaststroke": "orange",
            "freestyle": "red"
        }
        
        # Initialize color mappings
        self.matplotlib_stroke_colors = {}
        self.plotly_stroke_colors = {}
        self.stroke_names = {}
        
        self._initialize_color_mappings()
        
        # Pre-compute magnitudes if needed
        self._compute_magnitudes()
        
        # Initialize stroke visibility
        self.stroke_visibility = {}
        self._initialize_stroke_visibility()
    
    def _initialize_color_mappings(self):
        """Initialize color mappings based on stroke mode."""
        if self.stroke_mode == "Default Strokes (butterfly, backstroke, breaststroke, freestyle)":
            # Use predefined colors for default strokes
            self.matplotlib_stroke_colors = self.swimming_colors_matplotlib.copy()
            self.plotly_stroke_colors = self.swimming_colors_plotly.copy()
            self.stroke_names = {
                "butterfly": "Butterfly",
                "backstroke": "Backstroke",
                "breaststroke": "Breaststroke",
                "freestyle": "Freestyle"
            }
        else:
            # For custom strokes, map each unique name to a color
            all_stroke_names = set()
            
            # Add from features_df if available
            all_stroke_names.update(self.features_df["activity"].unique())
            
            # Add from stroke names dict if provided
            if self.stroke_names_dict:
                all_stroke_names.update(self.stroke_names_dict.keys())
            
            # Create mapping
            all_stroke_names = list(all_stroke_names)
            colors = list(self.swimming_colors_matplotlib.values())
            plotly_colors = list(self.swimming_colors_plotly.values())
            
            for i, stroke_name in enumerate(all_stroke_names):
                mat_color = colors[i % len(colors)]
                plt_color = plotly_colors[i % len(plotly_colors)]
                
                self.matplotlib_stroke_colors[stroke_name] = mat_color
                self.plotly_stroke_colors[stroke_name] = plt_color
                
                # Create display name
                display_name = stroke_name.replace("_", " ").title()
                self.stroke_names[stroke_name] = display_name
    
    def _initialize_stroke_visibility(self):
        """Initialize stroke visibility settings."""
        for i in range(self.num_strokes):
            stroke_name, _, _ = self._get_stroke_times(i)
            if stroke_name:
                # Initialize with True (visible) if not already in session state
                key = f"show_stroke_{stroke_name}"
                if key not in st.session_state:
                    st.session_state[key] = True
                self.stroke_visibility[stroke_name] = st.session_state[key]
    
    def _compute_magnitudes(self):
        """Compute signal magnitudes if not already present."""
        # Acceleration magnitude
        if "acc_magnitude" not in self.df.columns:
            if all(col in self.df.columns for col in ["Acc_X", "Acc_Y", "Acc_Z"]):
                self.df["acc_magnitude"] = np.sqrt(
                    self.df["Acc_X"]**2 + self.df["Acc_Y"]**2 + self.df["Acc_Z"]**2
                )
            elif all(col in self.df.columns for col in ["accelerationX", "accelerationY", "accelerationZ"]):
                self.df["acc_magnitude"] = np.sqrt(
                    self.df["accelerationX"]**2 + self.df["accelerationY"]**2 + self.df["accelerationZ"]**2
                )
        
        # Gyroscope magnitude
        if "gyro_magnitude" not in self.df.columns:
            if all(col in self.df.columns for col in ["Gyro_X", "Gyro_Y", "Gyro_Z"]):
                self.df["gyro_magnitude"] = np.sqrt(
                    self.df["Gyro_X"]**2 + self.df["Gyro_Y"]**2 + self.df["Gyro_Z"]**2
                )
            elif all(col in self.df.columns for col in ["rotationRateX", "rotationRateY", "rotationRateZ"]):
                self.df["gyro_magnitude"] = np.sqrt(
                    self.df["rotationRateX"]**2 + self.df["rotationRateY"]**2 + self.df["rotationRateZ"]**2
                )
    
    def _get_stroke_visibility(self, stroke_name):
        """Get stroke visibility based on stroke name."""
        return st.session_state.get(f"show_stroke_{stroke_name}", True)
    
    def _get_stroke_color(self, stroke_name):
        """Get color for a stroke."""
        return self.plotly_stroke_colors.get(stroke_name, 'gray')
    
    def _get_stroke_display_name(self, stroke_name):
        """Get display name for a stroke."""
        return self.stroke_names.get(stroke_name, stroke_name.replace("_", " ").title())
    
    def _get_stroke_times(self, stroke_idx):
        """Get start and end times for a stroke by index."""
        start_key = f"stroke_{stroke_idx}_start"
        end_key = f"stroke_{stroke_idx}_end"
        name_key = f"stroke_{stroke_idx}_name"
        
        if all(k in self.time_ranges for k in [start_key, end_key, name_key]):
            stroke_name = str(self.time_ranges[name_key]).strip()
            start_time = self.time_ranges[start_key]
            end_time = self.time_ranges[end_key]
            
            if start_time < end_time:
                return stroke_name, start_time, end_time
        
        return None, None, None
    
    def _add_stroke_legend_traces(self, fig):
        """
        Add dummy traces for stroke regions to appear in legend.
        
        Parameters:
        -----------
        fig : plotly.graph_objects.Figure
            The figure to add legend traces to
        """
        # Track which strokes we've already added to legend
        added_to_legend = set()
        
        for i in range(self.num_strokes):
            stroke_name, start_time, end_time = self._get_stroke_times(i)
            if stroke_name and start_time is not None and end_time is not None:
                # Only add once per stroke
                if stroke_name not in added_to_legend:
                    color = self._get_stroke_color(stroke_name)
                    display_name = self._get_stroke_display_name(stroke_name)
                    
                    # Add dummy trace for legend (invisible but appears in legend)
                    fig.add_trace(go.Scatter(
                        x=[None],  # No actual data points
                        y=[None],
                        mode='markers',
                        marker=dict(
                            size=15,
                            color=color,
                            symbol='square',
                            opacity=0.7,
                            line=dict(width=1, color='white')
                        ),
                        name=f"Stroke: {display_name}",
                        showlegend=True,
                        legendgroup='strokes',
                        visible=True
                    ))
                    added_to_legend.add(stroke_name)
    
    def _add_stroke_regions_to_plot(self, fig, opacity=0.2, annotation_text=True, row=None, col=None):
        """
        Add shaded stroke regions to a plot.
        
        Parameters:
        -----------
        fig : plotly.graph_objects.Figure
            The figure to add stroke regions to
        opacity : float
            Opacity of the shaded regions (0-1)
        annotation_text : bool
            Whether to add text annotations
        row, col : int, optional
            For subplots only
        """
        for i in range(self.num_strokes):
            stroke_name, start_time, end_time = self._get_stroke_times(i)
            if stroke_name and start_time is not None and end_time is not None:
                # Check if stroke should be visible
                if self._get_stroke_visibility(stroke_name):
                    color = self._get_stroke_color(stroke_name)
                    display_name = self._get_stroke_display_name(stroke_name)
                    
                    # Create vrect parameters
                    vrect_params = {
                        'x0': start_time,
                        'x1': end_time,
                        'fillcolor': color,
                        'opacity': opacity,
                        'layer': 'below',
                        'line_width': 0,
                    }
                    
                    # Add annotation if requested
                    if annotation_text:  # Add to every other stroke to avoid clutter
                        vrect_params['annotation_text'] = f"<b>{display_name}</b>"
                        vrect_params['annotation_position'] = 'top left'
                        vrect_params['annotation_font'] = dict(color=color, size=11)
                    
                    # Add row/col for subplots
                    if row is not None and col is not None:
                        vrect_params['row'] = row
                        vrect_params['col'] = col
                    
                    # Add the stroke region
                    fig.add_vrect(**vrect_params)
    
    def render_enhanced_signal_dashboard(self):
        """Render the complete enhanced signal visualization dashboard."""

        # Create tabs for different signal types
        signal_tabs = st.tabs([
            "📊 Acceleration & Gyroscope", 
            "📐 Orientation", 
            "⚡ Motion Magnitude", 
            "🎨 All Signals"
        ])

        # Tab 1: Acceleration & Gyroscope
        with signal_tabs[0]:
            self._render_acceleration_gyroscope_tab()
        
        # Tab 2: Orientation
        with signal_tabs[1]:
            self._render_orientation_tab()
        
        # Tab 3: Motion Magnitude
        with signal_tabs[2]:
            self._render_motion_magnitude_tab()
        
        # Tab 4: All Signals Dashboard
        with signal_tabs[3]:
            self._render_all_signals_dashboard_tab()
    
    def _render_acceleration_gyroscope_tab(self):
        """Render the acceleration and gyroscope visualization tab."""
        # Create main container with 3 columns
        col_left, col_center, col_right = st.columns([1, 3, 1])
        
        with col_center:
            # Modern header with wave icon
            st.markdown("""
            <div style="text-align: center; padding: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        border-radius: 10px; margin-bottom: 20px;">
                <h2 style="color: white; margin: 0;">🌊 Swimming Motion Analysis</h2>
                <p style="color: white; opacity: 0.9; margin: 5px 0 0 0;">Real-time acceleration & gyroscope visualization</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Left sidebar: Controls & Info
        with col_left:
            show_acc_x, show_acc_y, show_acc_z, show_gyro_x, show_gyro_y, show_gyro_z = self._render_signal_controls()
        
        # Main visualization area
        with col_center:
            self._render_acceleration_visualization(show_acc_x, show_acc_y, show_acc_z)
            self._render_gyroscope_visualization(show_gyro_x, show_gyro_y, show_gyro_z)
            self._render_magnitude_comparison()
        
        # Right sidebar: Insights & Metrics
        with col_right:
            self._render_signal_insights()
    
    def _render_signal_controls(self):
        """Render signal visibility controls in left sidebar."""
        st.markdown("### ⚙️ Settings")
        
        # Signal visibility toggles
        st.markdown("**Signal Display:**")
        show_acc_x = st.checkbox("Acc X", value=True, key="show_acc_x")
        show_acc_y = st.checkbox("Acc Y", value=True, key="show_acc_y")
        show_acc_z = st.checkbox("Acc Z", value=True, key="show_acc_z")
        show_gyro_x = st.checkbox("Gyro X", value=True, key="show_gyro_x")
        show_gyro_y = st.checkbox("Gyro Y", value=True, key="show_gyro_y")
        show_gyro_z = st.checkbox("Gyro Z", value=True, key="show_gyro_z")
        
        # Stroke visibility
        st.markdown("---")
        st.markdown("**Stroke Overlays:**")
        
        # Update stroke visibility based on user input
        for i in range(self.num_strokes):
            stroke_name, start_time, end_time = self._get_stroke_times(i)
            if stroke_name and start_time is not None and end_time is not None:
                display_name = self._get_stroke_display_name(stroke_name)
                # Store the checkbox value in session state
                st.session_state[f"show_stroke_{stroke_name}"] = st.checkbox(
                    f"Show {display_name}",
                    value=st.session_state.get(f"show_stroke_{stroke_name}", True),
                    key=f"stroke_checkbox_{stroke_name}"
                )
        
        # Smoothing control
        st.markdown("---")
        smoothing = st.slider(
            "Smoothing",
            min_value=1,
            max_value=50,
            value=5,
            help="Moving average window size for smoother curves",
            key="smoothing_slider"
        )
        
        # Water-themed color palette
        st.markdown("---")
        st.markdown("### 🎨 Color Theme")
        theme = st.radio(
            "Color Scheme",
            ["Ocean Blue", "Coral Reef", "Deep Sea", "Sunset"],
            horizontal=True,
            key="color_theme"
        )
        
        return show_acc_x, show_acc_y, show_acc_z, show_gyro_x, show_gyro_y, show_gyro_z
    
    def _render_acceleration_visualization(self, show_acc_x, show_acc_y, show_acc_z):
        """Render acceleration signal visualization."""
        st.markdown("### 📈 Acceleration Signals")
        
        # Calculate smoothed signals
        smoothing = st.session_state.get("smoothing_slider", 5)
        
        if smoothing > 1:
            acc_x_smooth = self.df["Acc_X"].rolling(window=smoothing, center=True).mean() if "Acc_X" in self.df.columns else None
            acc_y_smooth = self.df["Acc_Y"].rolling(window=smoothing, center=True).mean() if "Acc_Y" in self.df.columns else None
            acc_z_smooth = self.df["Acc_Z"].rolling(window=smoothing, center=True).mean() if "Acc_Z" in self.df.columns else None
        else:
            acc_x_smooth = self.df["Acc_X"] if "Acc_X" in self.df.columns else None
            acc_y_smooth = self.df["Acc_Y"] if "Acc_Y" in self.df.columns else None
            acc_z_smooth = self.df["Acc_Z"] if "Acc_Z" in self.df.columns else None
        
        # Create acceleration figure
        fig_acc = go.Figure()
        
        # Add dummy traces for stroke regions in legend FIRST
        self._add_stroke_legend_traces(fig_acc)
        
        # Theme-based colors
        theme = st.session_state.get("color_theme", "Ocean Blue")
        if theme == "Ocean Blue":
            acc_colors = ['#4B9FFF', '#1E90FF', '#0077BE']
            stroke_opacity = 0.15
        elif theme == "Coral Reef":
            acc_colors = ['#FF6B6B', '#4ECDC4', '#FFD166']
            stroke_opacity = 0.12
        elif theme == "Deep Sea":
            acc_colors = ['#2C3E50', '#3498DB', '#1ABC9C']
            stroke_opacity = 0.2
        else:  # Sunset
            acc_colors = ['#FF9A76', '#FF6B6B', '#FFD166']
            stroke_opacity = 0.1
        
        # Add acceleration traces
        if show_acc_x and acc_x_smooth is not None:
            fig_acc.add_trace(go.Scatter(
                x=self.df[self.time_col],
                y=acc_x_smooth,
                mode='lines',
                name='Acceleration X',
                line=dict(color=acc_colors[0], width=2.5),
                hovertemplate='<b>Acceleration X</b><br>Time: %{x:.2f}s<br>Value: %{y:.3f} m/s²<br><extra></extra>',
                fill='tozeroy',
                fillcolor=f'rgba({int(acc_colors[0][1:3], 16)}, {int(acc_colors[0][3:5], 16)}, {int(acc_colors[0][5:7], 16)}, 0.1)'
            ))
        
        if show_acc_y and acc_y_smooth is not None:
            fig_acc.add_trace(go.Scatter(
                x=self.df[self.time_col],
                y=acc_y_smooth,
                mode='lines',
                name='Acceleration Y',
                line=dict(color=acc_colors[1], width=2.5),
                hovertemplate='<b>Acceleration Y</b><br>Time: %{x:.2f}s<br>Value: %{y:.3f} m/s²<br><extra></extra>'
            ))
        
        if show_acc_z and acc_z_smooth is not None:
            fig_acc.add_trace(go.Scatter(
                x=self.df[self.time_col],
                y=acc_z_smooth,
                mode='lines',
                name='Acceleration Z',
                line=dict(color=acc_colors[2], width=2.5),
                hovertemplate='<b>Acceleration Z</b><br>Time: %{x:.2f}s<br>Value: %{y:.3f} m/s²<br><extra></extra>'
            ))
        
        # Add stroke regions as wave-like patterns
        self._add_stroke_regions_to_plot(fig_acc, opacity=stroke_opacity, annotation_text=True)
        
        # Style acceleration plot
        fig_acc.update_layout(
            height=350,
            template="plotly_white",
            hovermode="x unified",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                title="Legend",
                title_font=dict(size=12),
                font=dict(size=10),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='lightgray',
                borderwidth=1,
                itemclick="toggleothers",  # Better legend interaction
                itemdoubleclick="toggle"
            ),
            margin=dict(l=40, r=20, t=50, b=40),
            xaxis=dict(
                title="Time (seconds)",
                gridcolor='lightgray',
                showgrid=True,
                dtick=5,
                tick0=0,
                tickformat=".0f",
                rangeslider=dict(visible=True, thickness=0.05)  # Add range slider
            ),
            yaxis=dict(
                title="Acceleration (m/s²)",
                gridcolor='lightgray',
                showgrid=True
            ),
            plot_bgcolor='white'
        )
        
        st.plotly_chart(fig_acc, use_container_width=True)
    
    def _render_gyroscope_visualization(self, show_gyro_x, show_gyro_y, show_gyro_z):
        """Render gyroscope signal visualization."""
        st.markdown("### 🌀 Gyroscope Signals")
        
        # Calculate smoothed gyroscope signals
        smoothing = st.session_state.get("smoothing_slider", 5)
        
        if smoothing > 1:
            gyro_x_smooth = self.df["Gyro_X"].rolling(window=smoothing, center=True).mean() if "Gyro_X" in self.df.columns else None
            gyro_y_smooth = self.df["Gyro_Y"].rolling(window=smoothing, center=True).mean() if "Gyro_Y" in self.df.columns else None
            gyro_z_smooth = self.df["Gyro_Z"].rolling(window=smoothing, center=True).mean() if "Gyro_Z" in self.df.columns else None
        else:
            gyro_x_smooth = self.df["Gyro_X"] if "Gyro_X" in self.df.columns else None
            gyro_y_smooth = self.df["Gyro_Y"] if "Gyro_Y" in self.df.columns else None
            gyro_z_smooth = self.df["Gyro_Z"] if "Gyro_Z" in self.df.columns else None
        
        # Create gyroscope figure
        fig_gyro = go.Figure()
        
        # Add dummy traces for stroke regions in legend
        self._add_stroke_legend_traces(fig_gyro)
        
        # Gyroscope colors
        gyro_colors = ['#FF9A76', '#6C5CE7', '#00B894']
        
        # Add gyroscope traces
        if show_gyro_x and gyro_x_smooth is not None:
            fig_gyro.add_trace(go.Scatter(
                x=self.df[self.time_col],
                y=gyro_x_smooth,
                mode='lines',
                name='Gyroscope X',
                line=dict(color=gyro_colors[0], width=2.5),
                hovertemplate='<b>Gyroscope X</b><br>Time: %{x:.2f}s<br>Value: %{y:.3f} rad/s<br><extra></extra>'
            ))
        
        if show_gyro_y and gyro_y_smooth is not None:
            fig_gyro.add_trace(go.Scatter(
                x=self.df[self.time_col],
                y=gyro_y_smooth,
                mode='lines',
                name='Gyroscope Y',
                line=dict(color=gyro_colors[1], width=2.5),
                hovertemplate='<b>Gyroscope Y</b><br>Time: %{x:.2f}s<br>Value: %{y:.3f} rad/s<br><extra></extra>'
            ))
        
        if show_gyro_z and gyro_z_smooth is not None:
            fig_gyro.add_trace(go.Scatter(
                x=self.df[self.time_col],
                y=gyro_z_smooth,
                mode='lines',
                name='Gyroscope Z',
                line=dict(color=gyro_colors[2], width=2.5),
                hovertemplate='<b>Gyroscope Z</b><br>Time: %{x:.2f}s<br>Value: %{y:.3f} rad/s<br><extra></extra>'
            ))
        
        # Add stroke regions to gyroscope plot
        self._add_stroke_regions_to_plot(fig_gyro, opacity=0.15, annotation_text=False)
        
        # Style gyroscope plot
        fig_gyro.update_layout(
            height=350,
            template="plotly_white",
            hovermode="x unified",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                title="Legend",
                title_font=dict(size=12),
                font=dict(size=10),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='lightgray',
                borderwidth=1
            ),
            margin=dict(l=40, r=20, t=50, b=40),
            xaxis=dict(
                title="Time (seconds)",
                gridcolor='lightgray',
                showgrid=True,
                dtick=5,
                tick0=0,
                tickformat=".0f",
                rangeslider=dict(visible=True, thickness=0.05)
            ),
            yaxis=dict(
                title="Angular Velocity (rad/s)",
                gridcolor='lightgray',
                showgrid=True
            ),
            plot_bgcolor='white'
        )
        
        st.plotly_chart(fig_gyro, use_container_width=True)
    
    def _render_magnitude_comparison(self):
        """Render motion magnitude comparison visualization."""
        st.markdown("### ⚡ Motion Magnitude Comparison")
        
        smoothing = st.session_state.get("smoothing_slider", 5)
        
        # Calculate magnitudes
        if "acc_magnitude" in self.df.columns:
            acc_mag_smooth = self.df["acc_magnitude"].rolling(window=smoothing, center=True).mean()
        else:
            if all(col in self.df.columns for col in ["Acc_X", "Acc_Y", "Acc_Z"]):
                acc_mag = np.sqrt(self.df["Acc_X"]**2 + self.df["Acc_Y"]**2 + self.df["Acc_Z"]**2)
                acc_mag_smooth = acc_mag.rolling(window=smoothing, center=True).mean()
            else:
                return
        
        if "gyro_magnitude" in self.df.columns:
            gyro_mag_smooth = self.df["gyro_magnitude"].rolling(window=smoothing, center=True).mean()
        else:
            if all(col in self.df.columns for col in ["Gyro_X", "Gyro_Y", "Gyro_Z"]):
                gyro_mag = np.sqrt(self.df["Gyro_X"]**2 + self.df["Gyro_Y"]**2 + self.df["Gyro_Z"]**2)
                gyro_mag_smooth = gyro_mag.rolling(window=smoothing, center=True).mean()
            else:
                return
        
        # Create magnitude figure
        fig_mag = go.Figure()
        
        # Add dummy traces for stroke regions in legend
        self._add_stroke_legend_traces(fig_mag)
        
        # Add magnitude traces
        fig_mag.add_trace(go.Scatter(
            x=self.df[self.time_col],
            y=acc_mag_smooth,
            mode='lines',
            name='Acceleration Magnitude',
            line=dict(color='#4B9FFF', width=3),
            hovertemplate='<b>Acceleration Magnitude</b><br>Time: %{x:.2f}s<br>Value: %{y:.3f}<br><extra></extra>',
            fill='tozeroy',
            fillcolor='rgba(75, 159, 255, 0.1)'
        ))
        
        fig_mag.add_trace(go.Scatter(
            x=self.df[self.time_col],
            y=gyro_mag_smooth,
            mode='lines',
            name='Gyroscope Magnitude',
            line=dict(color='#FF9A76', width=3),
            hovertemplate='<b>Gyroscope Magnitude</b><br>Time: %{x:.2f}s<br>Value: %{y:.3f}<br><extra></extra>'
        ))
        
        # Add stroke regions
        self._add_stroke_regions_to_plot(fig_mag, opacity=0.15, annotation_text=True)
        
        # Style magnitude plot
        fig_mag.update_layout(
            height=350,
            template="plotly_white",
            hovermode="x unified",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                title="Legend",
                title_font=dict(size=12),
                font=dict(size=10),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='lightgray',
                borderwidth=1
            ),
            margin=dict(l=40, r=20, t=50, b=40),
            xaxis=dict(
                title="Time (seconds)",
                gridcolor='lightgray',
                showgrid=True,
                dtick=5,
                tick0=0,
                tickformat=".0f",
                rangeslider=dict(visible=True, thickness=0.05)
            ),
            yaxis=dict(
                title="Magnitude",
                gridcolor='lightgray',
                showgrid=True
            ),
            plot_bgcolor='white'
        )
        
        st.plotly_chart(fig_mag, use_container_width=True)
    
    def _render_signal_insights(self):
        """Render insights and metrics in right sidebar."""
        st.markdown("### 📊 Insights")
        
        # Current view metrics
        st.markdown("**Signal Statistics:**")
        
        # Calculate basic stats
        if "Acc_X" in self.df.columns:
            acc_x_mean = self.df["Acc_X"].mean()
            acc_x_std = self.df["Acc_X"].std()
            st.metric("Acc X Mean", f"{acc_x_mean:.3f}", delta=f"±{acc_x_std:.3f}")
        
        if "Gyro_Z" in self.df.columns:
            gyro_z_mean = self.df["Gyro_Z"].mean()
            gyro_z_std = self.df["Gyro_Z"].std()
            st.metric("Gyro Z Mean", f"{gyro_z_mean:.3f}", delta=f"±{gyro_z_std:.3f}")
        
        # Stroke timing metrics
        st.markdown("---")
        st.markdown("**Stroke Timing:**")
        
        stroke_timings = []
        for i in range(self.num_strokes):
            stroke_name, start_time, end_time = self._get_stroke_times(i)
            if stroke_name and start_time is not None and end_time is not None:
                duration = end_time - start_time
                if duration > 0:
                    display_name = self._get_stroke_display_name(stroke_name)
                    stroke_timings.append((display_name, duration))
        
        # Display top 3 longest strokes
        if stroke_timings:
            stroke_timings.sort(key=lambda x: x[1], reverse=True)
            for name, duration in stroke_timings[:3]:
                st.progress(min(duration/30, 1.0), text=f"{name}: {duration:.1f}s")
        
        # Key observations
        st.markdown("---")
        st.markdown("**Observations:**")
        
        observations = []
        
        # Check for high acceleration periods
        if "acc_magnitude" in self.df.columns or all(c in self.df.columns for c in ["Acc_X", "Acc_Y", "Acc_Z"]):
            if "acc_magnitude" not in self.df.columns:
                self.df["acc_magnitude"] = np.sqrt(self.df["Acc_X"]**2 + self.df["Acc_Y"]**2 + self.df["Acc_Z"]**2)
            
            high_acc = self.df[self.df["acc_magnitude"] > self.df["acc_magnitude"].quantile(0.9)]
            if len(high_acc) > 0:
                observations.append(f"**High acceleration** detected in {len(high_acc)} segments")
        
        # Check stroke rhythm consistency
        if "stroke_frequency" in self.features_df.columns:
            freq_std = self.features_df.groupby("activity")["stroke_frequency"].std().mean()
            if freq_std < 0.5:
                observations.append("**Consistent stroke rhythm** across all strokes")
            else:
                observations.append("**Variable stroke rhythm** detected")
        
        # Display observations
        for obs in observations[:3]:  # Show only top 3
            st.info(obs)
        
        # Quick tips
        st.markdown("---")
        st.markdown("💡 **Tips:**")
        st.markdown("""
        • Click legend items to toggle visibility
        • Hover over signals for detailed values
        • Use smoothing for cleaner patterns
        • Compare stroke colors with signal patterns
        • Use range slider to zoom in/out
        """)
    
    def _render_orientation_tab(self):
        """Render the orientation visualization tab."""
        st.markdown("""
        <div style="text-align: center; padding: 10px; background: linear-gradient(135deg, #1D976C 0%, #93F9B9 100%);
                    border-radius: 10px; margin-bottom: 20px;">
            <h2 style="color: white; margin: 0;">🧭 Body Orientation Analysis</h2>
            <p style="color: white; opacity: 0.9; margin: 5px 0 0 0;">Pitch, Roll & Yaw angles during swimming</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create 2-column layout for orientation
        orient_col1, orient_col2 = st.columns([3, 1])
        
        with orient_col1:
            self._render_orientation_visualization()
        
        with orient_col2:
            self._render_orientation_controls()
    
    def _render_orientation_visualization(self):
        """Render orientation signal visualization."""
        # Create main orientation figure
        fig_orient = go.Figure()
        
        # Add dummy traces for stroke regions in legend
        self._add_stroke_legend_traces(fig_orient)
        
        # Orientation color scheme
        orient_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        # Orientation signals data
        orientation_data = []
        orientation_names = []
        
        if "Pitch" in self.df.columns or "pitch" in self.df.columns:
            pitch = self.df["Pitch"] if "Pitch" in self.df.columns else self.df["pitch"]
            orientation_data.append(pitch)
            orientation_names.append("Pitch")
        
        if "Roll" in self.df.columns or "roll" in self.df.columns:
            roll = self.df["Roll"] if "Roll" in self.df.columns else self.df["roll"]
            orientation_data.append(roll)
            orientation_names.append("Roll")
        
        if "Yaw" in self.df.columns or "yaw" in self.df.columns:
            yaw = self.df["Yaw"] if "Yaw" in self.df.columns else self.df["yaw"]
            orientation_data.append(yaw)
            orientation_names.append("Yaw")
        
        # Add orientation traces
        for i, (data, name) in enumerate(zip(orientation_data, orientation_names)):
            fig_orient.add_trace(go.Scatter(
                x=self.df[self.time_col],
                y=data,
                mode='lines',
                name=name,
                line=dict(color=orient_colors[i], width=2.5),
                hovertemplate=f'<b>{name}</b><br>Time: %{{x:.2f}}s<br>Angle: %{{y:.3f}} rad<br><extra></extra>',
                visible=True,
                fill='tozeroy' if i == 0 else None,
                fillcolor=f'rgba({int(orient_colors[i][1:3], 16)}, {int(orient_colors[i][3:5], 16)}, {int(orient_colors[i][5:7], 16)}, 0.1)' if i == 0 else None
            ))
        
        # Add stroke regions
        self._add_stroke_regions_to_plot(fig_orient, opacity=0.08, annotation_text=True)
        
        # Style orientation plot
        fig_orient.update_layout(
            height=500,
            template="plotly_white",
            hovermode="x unified",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.8)",
                title="Legend",
                title_font=dict(size=12),
                font=dict(size=10)
            ),
            margin=dict(l=50, r=30, t=80, b=50),
            xaxis=dict(
                title="<b>Time (seconds)</b>",
                gridcolor='rgba(200,200,200,0.3)',
                showgrid=True,
                dtick=5,
                tick0=0,
                tickformat=".0f",
                showline=True,
                linecolor='lightgray',
                linewidth=1,
                rangeslider=dict(visible=True, thickness=0.05)
            ),
            yaxis=dict(
                title="<b>Angle (radians)</b>",
                gridcolor='rgba(200,200,200,0.3)',
                showgrid=True,
                zeroline=True,
                zerolinecolor='lightgray',
                zerolinewidth=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig_orient, use_container_width=True)
        
        # Orientation phase analysis
        st.markdown("### 🔄 Orientation Phase Analysis")
        
        col_phase1, col_phase2, col_phase3 = st.columns(3)
        
        with col_phase1:
            if "Pitch" in self.df.columns or "pitch" in self.df.columns:
                pitch_data = self.df["Pitch"] if "Pitch" in self.df.columns else self.df["pitch"]
                pitch_range = pitch_data.max() - pitch_data.min()
                st.metric("Pitch Range", f"{pitch_range:.3f} rad", 
                         delta=f"±{pitch_range/2:.3f}")
        
        with col_phase2:
            if "Roll" in self.df.columns or "roll" in self.df.columns:
                roll_data = self.df["Roll"] if "Roll" in self.df.columns else self.df["roll"]
                roll_avg = roll_data.mean()
                st.metric("Roll Average", f"{roll_avg:.3f} rad",
                         delta="Neutral" if abs(roll_avg) < 0.1 else "Biased")
        
        with col_phase3:
            if "Yaw" in self.df.columns or "yaw" in self.df.columns:
                yaw_data = self.df["Yaw"] if "Yaw" in self.df.columns else self.df["yaw"]
                yaw_change = yaw_data.diff().abs().mean()
                st.metric("Yaw Change Rate", f"{yaw_change:.3f} rad/s")
    
    def _render_orientation_controls(self):
        """Render orientation controls and insights."""
        st.markdown("### ⚙️ Controls")
        
        # Orientation visibility
        st.markdown("**Show Angles:**")
        show_pitch = st.checkbox("Pitch", value=True, key="show_pitch_orient")
        show_roll = st.checkbox("Roll", value=True, key="show_roll_orient")
        show_yaw = st.checkbox("Yaw", value=True, key="show_yaw_orient")
        
        # Normalize option
        normalize = st.checkbox("Normalize Angles", value=False, 
                               help="Scale all angles to 0-1 range")
        
        # Smoothing for orientation
        orient_smooth = st.slider("Smoothing", 1, 20, 3, key="orient_smooth")
        
        st.markdown("---")
        st.markdown("### 📊 Insights")
        
        # Orientation stability
        orientation_signals = []
        orientation_names = []
        
        if "Pitch" in self.df.columns or "pitch" in self.df.columns:
            orientation_signals.append(self.df["Pitch"] if "Pitch" in self.df.columns else self.df["pitch"])
            orientation_names.append("Pitch")
        
        if "Roll" in self.df.columns or "roll" in self.df.columns:
            orientation_signals.append(self.df["Roll"] if "Roll" in self.df.columns else self.df["roll"])
            orientation_names.append("Roll")
        
        if orientation_signals:
            stability_scores = []
            for data, name in zip(orientation_signals, orientation_names):
                variability = data.diff().abs().mean()
                stability = 1 / (1 + variability) if variability > 0 else 1
                stability_scores.append((name, stability))
            
            st.markdown("**Orientation Stability:**")
            for name, score in stability_scores:
                st.progress(score, text=f"{name}: {score:.2%}")
        
        # Stroke orientation patterns
        st.markdown("---")
        st.markdown("**Stroke Patterns:**")
        
        # Quick observations
        observations = []
        if "Pitch" in self.df.columns or "pitch" in self.df.columns:
            pitch_data = self.df["Pitch"] if "Pitch" in self.df.columns else self.df["pitch"]
            if abs(pitch_data.mean()) > 0.5:
                observations.append("Strong pitch bias detected")
        
        if "Roll" in self.df.columns or "roll" in self.df.columns:
            roll_data = self.df["Roll"] if "Roll" in self.df.columns else self.df["roll"]
            if roll_data.std() > 1.0:
                observations.append("High roll variability")
        
        for obs in observations:
            st.info(obs)
        
        st.markdown("---")
        st.markdown("💡 **Interpretation:**")
        st.markdown("""
        • **Pitch**: Head position (up/down)
        • **Roll**: Body rotation (side-to-side)
        • **Yaw**: Direction/facing
        • **Stroke colors** show activity regions
        """)
    
    def _render_motion_magnitude_tab(self):
        """Render the motion magnitude visualization tab."""
        st.markdown("""
        <div style="text-align: center; padding: 10px; background: linear-gradient(135deg, #8E2DE2 0%, #4A00E0 100%);
                    border-radius: 10px; margin-bottom: 20px;">
            <h2 style="color: white; margin: 0;">⚡ Motion Power Analysis</h2>
            <p style="color: white; opacity: 0.9; margin: 5px 0 0 0;">Acceleration & Gyroscope magnitude patterns</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create 2-column layout for motion magnitude
        mag_col1, mag_col2 = st.columns([3, 1])
        
        with mag_col1:
            self._render_motion_magnitude_visualization()
        
        with mag_col2:
            self._render_magnitude_controls()
    
    def _render_motion_magnitude_visualization(self):
        """Render motion magnitude visualization."""
        # Create magnitude figure
        fig_mag = go.Figure()
        
        # Add dummy traces for stroke regions in legend
        self._add_stroke_legend_traces(fig_mag)
        
        # Magnitude colors
        mag_colors = ['#FF9A76', '#6C5CE7']
        
        # Add acceleration magnitude
        if "acc_magnitude" in self.df.columns:
            acc_mag = self.df["acc_magnitude"]
            fig_mag.add_trace(go.Scatter(
                x=self.df[self.time_col],
                y=acc_mag,
                mode='lines',
                name='Acceleration Magnitude',
                line=dict(color=mag_colors[0], width=3),
                hovertemplate='<b>Acceleration Magnitude</b><br>Time: %{x:.2f}s<br>Value: %{y:.3f} m/s²<br><extra></extra>',
                fill='tozeroy',
                fillcolor='rgba(255, 154, 118, 0.2)'
            ))
        
        # Add gyroscope magnitude
        if "gyro_magnitude" in self.df.columns:
            gyro_mag = self.df["gyro_magnitude"]
            fig_mag.add_trace(go.Scatter(
                x=self.df[self.time_col],
                y=gyro_mag,
                mode='lines',
                name='Gyroscope Magnitude',
                line=dict(color=mag_colors[1], width=3),
                hovertemplate='<b>Gyroscope Magnitude</b><br>Time: %{x:.2f}s<br>Value: %{y:.3f} rad/s<br><extra></extra>'
            ))
        
        # Add stroke power zones
        self._add_stroke_regions_to_plot(fig_mag, opacity=0.3, annotation_text=True)
        
        # Style magnitude plot
        fig_mag.update_layout(
            height=500,
            template="plotly_white",
            hovermode="x unified",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.8)",
                title="Legend",
                title_font=dict(size=12),
                font=dict(size=10)
            ),
            margin=dict(l=50, r=30, t=80, b=50),
            xaxis=dict(
                title="<b>Time (seconds)</b>",
                gridcolor='rgba(200,200,200,0.3)',
                showgrid=True,
                dtick=5,
                tick0=0,
                tickformat=".0f",
                showline=True,
                linecolor='lightgray',
                linewidth=1,
                rangeslider=dict(visible=True, thickness=0.05)
            ),
            yaxis=dict(
                title="<b>Magnitude</b>",
                gridcolor='rgba(200,200,200,0.3)',
                showgrid=True
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig_mag, use_container_width=True)
        
        # Power metrics
        st.markdown("### 🏋️ Power Statistics")
        
        if "acc_magnitude" in self.df.columns and "gyro_magnitude" in self.df.columns:
            col_power1, col_power2, col_power3, col_power4 = st.columns(4)
            
            with col_power1:
                avg_acc_power = self.df["acc_magnitude"].mean()
                st.metric("Avg Acceleration", f"{avg_acc_power:.3f}", 
                         delta="High" if avg_acc_power > 1 else "Low")
            
            with col_power2:
                avg_gyro_power = self.df["gyro_magnitude"].mean()
                st.metric("Avg Rotation", f"{avg_gyro_power:.3f}")
            
            with col_power3:
                peak_acc = self.df["acc_magnitude"].max()
                st.metric("Peak Acceleration", f"{peak_acc:.3f}")
            
            with col_power4:
                power_variation = self.df["acc_magnitude"].std() / self.df["acc_magnitude"].mean() if self.df["acc_magnitude"].mean() > 0 else 0
                st.metric("Power CV", f"{power_variation:.2f}",
                         delta="Consistent" if power_variation < 0.5 else "Variable")
    
    def _render_magnitude_controls(self):
        """Render magnitude controls and insights."""
        st.markdown("### ⚙️ Controls")
        
        st.markdown("**Display Options:**")
        show_acc_mag = st.checkbox("Acceleration", value=True, key="show_acc_mag_magnitude")
        show_gyro_mag = st.checkbox("Rotation", value=True, key="show_gyro_mag_magnitude")
        show_power = st.checkbox("Power Envelope", value=False, key="show_power_magnitude")
        
        # Normalization
        normalize_mag = st.checkbox("Normalize", value=False,
                                   help="Scale magnitudes to 0-1 range")
        
        # Log scale option
        log_scale = st.checkbox("Log Scale", value=False,
                               help="Use logarithmic scale for magnitude")
        
        st.markdown("---")
        st.markdown("### 📊 Power Insights")
        
        # Calculate stroke power ranking
        stroke_powers = []
        for i in range(self.num_strokes):
            stroke_name, start_time, end_time = self._get_stroke_times(i)
            if stroke_name and start_time is not None and end_time is not None:
                stroke_mask = (self.df[self.time_col] >= start_time) & (self.df[self.time_col] <= end_time)
                if stroke_mask.any():
                    acc_power = self.df.loc[stroke_mask, "acc_magnitude"].mean() if "acc_magnitude" in self.df.columns else 0
                    gyro_power = self.df.loc[stroke_mask, "gyro_magnitude"].mean() if "gyro_magnitude" in self.df.columns else 0
                    total_power = (acc_power + gyro_power * 3) / 2  # Weighted combination
                    
                    display_name = self._get_stroke_display_name(stroke_name)
                    stroke_powers.append((display_name, total_power))
        
        if stroke_powers:
            st.markdown("**Stroke Power Ranking:**")
            stroke_powers.sort(key=lambda x: x[1], reverse=True)
            max_power = max([p for _, p in stroke_powers]) if stroke_powers else 1
            
            for name, power in stroke_powers[:5]:  # Show top 5
                st.progress(power/max_power, text=f"{name}: {power:.2f}")
        
        st.markdown("---")
        st.markdown("💡 **Interpretation:**")
        st.markdown("""
        • **Acceleration**: Linear movement power
        • **Rotation**: Turning/spinning power
        • **Darker shades** = Higher intensity
        • **Numbers** = Total power score
        • **Stroke colors** show activity regions
        """)
    
    def _render_all_signals_dashboard_tab(self):
        """Render the comprehensive all signals dashboard tab."""
        st.markdown("""
        <div style="text-align: center; padding: 10px; background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%);
                    border-radius: 10px; margin-bottom: 20px;">
            <h2 style="color: white; margin: 0;">🎯 Comprehensive Signal Dashboard</h2>
            <p style="color: white; opacity: 0.9; margin: 5px 0 0 0;">Interactive exploration of all IMU signals</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create dashboard layout
        dash_col1, dash_col2 = st.columns([3, 1])
        
        with dash_col1:
            self._render_comprehensive_dashboard()
        
        with dash_col2:
            self._render_dashboard_controls()
    
    def _render_comprehensive_dashboard(self):
        """Render comprehensive dashboard with multiple subplots."""
        # Create comprehensive dashboard figure
        fig_dashboard = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                "<b>Acceleration Signals (m/s²)</b>",
                "<b>Gyroscope Signals (rad/s)</b>", 
                "<b>Motion Magnitude & Power</b>"
            ),
            row_heights=[0.35, 0.35, 0.3]
        )
        
        # 1. ACCELERATION SIGNALS
        acc_colors = ['#FF4B4B', '#4B9FFF', '#4BFF9F']
        acc_signals = ['Acc_X', 'Acc_Y', 'Acc_Z']
        acc_names = ['Acceleration X', 'Acceleration Y', 'Acceleration Z']
        
        for i, (sig, name, color) in enumerate(zip(acc_signals, acc_names, acc_colors)):
            if sig in self.df.columns:
                signal_data = self.df[sig]
                fig_dashboard.add_trace(
                    go.Scatter(
                        x=self.df[self.time_col],
                        y=signal_data,
                        mode='lines',
                        name=name,
                        line=dict(color=color, width=1.8),
                        hovertemplate=f'<b>{name}</b><br>Time: %{{x:.2f}}s<br>Value: %{{y:.3f}}<br><extra></extra>',
                        legendgroup='acceleration'
                    ),
                    row=1, col=1
                )
        
        # 2. GYROSCOPE SIGNALS
        gyro_colors = ['#FFA500', '#800080', '#008080']
        gyro_signals = ['Gyro_X', 'Gyro_Y', 'Gyro_Z']
        gyro_names = ['Gyroscope X', 'Gyroscope Y', 'Gyroscope Z']
        
        for i, (sig, name, color) in enumerate(zip(gyro_signals, gyro_names, gyro_colors)):
            if sig in self.df.columns:
                signal_data = self.df[sig]
                fig_dashboard.add_trace(
                    go.Scatter(
                        x=self.df[self.time_col],
                        y=signal_data,
                        mode='lines',
                        name=name,
                        line=dict(color=color, width=1.8),
                        hovertemplate=f'<b>{name}</b><br>Time: %{{x:.2f}}s<br>Value: %{{y:.3f}}<br><extra></extra>',
                        legendgroup='gyroscope'
                    ),
                    row=2, col=1
                )
        
        # 3. MOTION MAGNITUDE
        if "acc_magnitude" in self.df.columns:
            fig_dashboard.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df["acc_magnitude"],
                    mode='lines',
                    name='Acceleration Magnitude',
                    line=dict(color='#2C3E50', width=2.5),
                    hovertemplate='<b>Acceleration Magnitude</b><br>Time: %{x:.2f}s<br>Value: %{y:.3f}<br><extra></extra>',
                    fill='tozeroy',
                    fillcolor='rgba(44, 62, 80, 0.1)',
                    legendgroup='magnitude'
                ),
                row=3, col=1
            )
        
        if "gyro_magnitude" in self.df.columns:
            fig_dashboard.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df["gyro_magnitude"],
                    mode='lines',
                    name='Gyroscope Magnitude',
                    line=dict(color='#8E44AD', width=2.5),
                    hovertemplate='<b>Gyroscope Magnitude</b><br>Time: %{x:.2f}s<br>Value: %{y:.3f}<br><extra></extra>',
                    legendgroup='magnitude'
                ),
                row=3, col=1
            )
        
        # Add stroke regions across all subplots
        for i in range(self.num_strokes):
            stroke_name, start_time, end_time = self._get_stroke_times(i)
            if stroke_name and start_time is not None and end_time is not None:
                color = self._get_stroke_color(stroke_name)
                display_name = self._get_stroke_display_name(stroke_name)
                
                # Add to all three subplots
                for row in [1, 2, 3]:
                    fig_dashboard.add_vrect(
                        x0=start_time,
                        x1=end_time,
                        fillcolor=color,
                        opacity=0.08,
                        layer="below",
                        line_width=0,
                        row=row, col=1
                    )
                
                # Add annotation only on first row
                fig_dashboard.add_annotation(
                    x=(start_time + end_time) / 2,
                    y=1.12,
                    xref="x",
                    yref="paper",
                    text=f"<b>{display_name}</b>",
                    showarrow=False,
                    font=dict(size=9, color=color),
                    row=1, col=1
                )
        
        # FIXED: Add proper stroke legend (using the same approach as other tabs)
        # Track which strokes we've already added to legend
        added_to_legend = set()
        
        for i in range(self.num_strokes):
            stroke_name, start_time, end_time = self._get_stroke_times(i)
            if stroke_name and start_time is not None and end_time is not None:
                # Only add once per stroke
                if stroke_name not in added_to_legend:
                    color = self._get_stroke_color(stroke_name)
                    display_name = self._get_stroke_display_name(stroke_name)
                    
                    # Add dummy trace for legend (invisible but appears in legend)
                    fig_dashboard.add_trace(go.Scatter(
                        x=[None],  # No actual data points
                        y=[None],
                        mode='markers',
                        marker=dict(
                            size=15,
                            color=color,
                            symbol='square',
                            opacity=0.7,
                            line=dict(width=1, color='white')
                        ),
                        name=f"Stroke: {display_name}",
                        showlegend=True,
                        legendgroup='strokes',
                        visible=True
                    ), row=1, col=1)  # FIX: Added row and col parameters
                    added_to_legend.add(stroke_name)
        
        # Style the dashboard
        fig_dashboard.update_layout(
            height=900,
            template="plotly_white",
            hovermode="x unified",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="lightgray",
                borderwidth=1,
                groupclick="toggleitem",
                title="Legend",
                title_font=dict(size=12),
                font=dict(size=10)
            ),
            margin=dict(l=50, r=30, t=100, b=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Update axes
        fig_dashboard.update_xaxes(
            title="<b>Time (seconds)</b>",
            gridcolor='rgba(200,200,200,0.2)',
            showgrid=True,
            dtick=5,
            tick0=0,
            tickformat=".0f",
            row=3, col=1
        )
        
        fig_dashboard.update_yaxes(
            title="<b>Acceleration (m/s²)</b>",
            gridcolor='rgba(200,200,200,0.2)',
            showgrid=True,
            row=1, col=1
        )
        
        fig_dashboard.update_yaxes(
            title="<b>Angular Velocity (rad/s)</b>",
            gridcolor='rgba(200,200,200,0.2)',
            showgrid=True,
            row=2, col=1
        )
        
        fig_dashboard.update_yaxes(
            title="<b>Magnitude</b>",
            gridcolor='rgba(200,200,200,0.2)',
            showgrid=True,
            row=3, col=1
        )
        
        # Add range slider
        fig_dashboard.update_xaxes(rangeslider=dict(visible=True), row=3, col=1)
        
        st.plotly_chart(fig_dashboard, use_container_width=True)
    
    def _render_dashboard_controls(self):
        """Render dashboard controls and insights."""
        st.markdown("### 🎮 Dashboard Controls")
        
        st.markdown("**Signal Groups:**")
        show_acc_group = st.checkbox("Acceleration", value=True, key="show_acc_group_dash")
        show_gyro_group = st.checkbox("Gyroscope", value=True, key="show_gyro_group_dash")
        show_mag_group = st.checkbox("Magnitude", value=True, key="show_mag_group_dash")
        
        st.markdown("---")
        st.markdown("**Display Mode:**")
        display_mode = st.radio(
            "View Mode",
            ["Standard", "Centered", "Normalized"],
            horizontal=False,
            key="display_mode_dash"
        )
        
        # Time window selection
        st.markdown("---")
        st.markdown("**Time Window:**")
        
        if len(self.df) > 0:
            time_min = float(self.df[self.time_col].min())
            time_max = float(self.df[self.time_col].max())
            
            time_range = st.slider(
                "Select time range",
                min_value=time_min,
                max_value=time_max,
                value=(time_min, min(time_max, time_min + 30)),
                step=1.0,
                format="%.1f",
                key="time_range_dash"
            )
        
        # Signal statistics
        st.markdown("---")
        st.markdown("### 📈 Signal Stats")
        
        if "Acc_X" in self.df.columns:
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.metric("Acc X Max", f"{self.df['Acc_X'].max():.2f}")
            with col_stat2:
                st.metric("Acc X Min", f"{self.df['Acc_X'].min():.2f}")
        
        if "Gyro_Z" in self.df.columns:
            col_stat3, col_stat4 = st.columns(2)
            with col_stat3:
                st.metric("Gyro Z Avg", f"{self.df['Gyro_Z'].mean():.2f}")
            with col_stat4:
                st.metric("Gyro Z Std", f"{self.df['Gyro_Z'].std():.2f}")
        
        # Stroke efficiency
        st.markdown("---")
        st.markdown("### 🏊‍♂️ Stroke Efficiency")
        
        stroke_efficiency = []
        for i in range(self.num_strokes):
            stroke_name, start_time, end_time = self._get_stroke_times(i)
            if stroke_name and start_time is not None and end_time is not None:
                duration = end_time - start_time
                
                if duration > 0:
                    display_name = self._get_stroke_display_name(stroke_name)
                    
                    # Simple efficiency calculation
                    if "acc_magnitude" in self.df.columns:
                        stroke_mask = (self.df[self.time_col] >= start_time) & (self.df[self.time_col] <= end_time)
                        if stroke_mask.any():
                            stroke_power = self.df.loc[stroke_mask, "acc_magnitude"].mean()
                            efficiency = stroke_power / duration if duration > 0 else 0
                            stroke_efficiency.append((display_name, efficiency))
        
        if stroke_efficiency:
            st.markdown("**Power/Duration Ratio:**")
            for name, eff in sorted(stroke_efficiency, key=lambda x: x[1], reverse=True)[:3]:
                st.progress(min(eff/10, 1.0), text=f"{name}: {eff:.2f}")
        
        st.markdown("---")
        st.markdown("💡 **Dashboard Tips:**")
        st.markdown("""
        • Click legend to toggle signals
        • Use range slider to zoom
        • Hover for exact values
        • Compare stroke colors
        • Stroke colors show activity regions
        """)