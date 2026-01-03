import pandas as pd
import streamlit as st
from typing import Optional, Dict, List, Tuple

def load_and_process_csv(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Simple function to load and process CSV files with flexible column names.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Processed DataFrame with standardized column names, or None if failed
    """
    
    # Define patterns for each required column
    PATTERNS = {
        'time': ['time', 'timestamp', 't', 'time(s)', 'time (s)', 'time_sec'],
        'x': ['x', 'accel x', 'acc_x', 'acceleration x', 'x (m/s^2)', 'x-axis', 
              'x acceleration', 'accx', 'acc_x_mss', 'accelerometer_x', 'acc_x_mss'],
        'y': ['y', 'accel y', 'acc_y', 'acceleration y', 'y (m/s^2)', 'y-axis', 
              'y acceleration', 'accy', 'acc_y_mss', 'accelerometer_y', 'acc_y_mss'],
        'z': ['z', 'accel z', 'acc_z', 'acceleration z', 'z (m/s^2)', 'z-axis', 
              'z acceleration', 'accz', 'acc_z_mss', 'accelerometer_z', 'acc_z_mss']
    }
    
    # Standard column names we want
    STANDARD_NAMES = {
        'time': "Time (s)",
        'x': "X (m/s^2)",
        'y': "Y (m/s^2)",
        'z': "Z (m/s^2)"
    }
    
    try:
        # Read CSV file
        df = pd.read_csv(uploaded_file)
        original_columns = df.columns.tolist()
        
        # Find matching columns for each required type
        found_columns = {}
        column_mapping = {}
        
        for col_type, patterns in PATTERNS.items():
            found = False
            for col in df.columns:
                col_lower = str(col).lower().strip()
                for pattern in patterns:
                    if pattern.lower() in col_lower:
                        found_columns[col_type] = col
                        column_mapping[col] = STANDARD_NAMES[col_type]
                        found = True
                        break
                if found:
                    break
        
        # Check if all required columns were found
        missing = [col_type.upper() for col_type in PATTERNS.keys() 
                  if col_type not in found_columns]
        
        if missing:
            st.error(f"""
            ❌ Missing columns: {', '.join(missing)}
            
            **Available columns:** {', '.join(original_columns)}
            
            **Looking for patterns like:**
            - Time: {', '.join(PATTERNS['time'][:3])}...
            - X: {', '.join(PATTERNS['x'][:3])}...
            - Y: {', '.join(PATTERNS['y'][:3])}...
            - Z: {', '.join(PATTERNS['z'][:3])}...
            """)
            return None
        
        # Rename columns to standard names
        df = df.rename(columns=column_mapping)
        
        # Show what was detected
        st.success("✅ CSV loaded successfully!")
        st.info(f"""
        **Detected columns:**
        - `{found_columns['time']}` → **{STANDARD_NAMES['time']}**
        - `{found_columns['x']}` → **{STANDARD_NAMES['x']}**
        - `{found_columns['y']}` → **{STANDARD_NAMES['y']}**
        - `{found_columns['z']}` → **{STANDARD_NAMES['z']}**
        """)
        
        # Convert to numeric and clean
        for col in STANDARD_NAMES.values():
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with NaN in required columns
        df = df.dropna(subset=list(STANDARD_NAMES.values()))
        
        if df.empty:
            st.error("❌ No valid data after cleaning.")
            return None
        
        # Sort by time
        df = df.sort_values(by=STANDARD_NAMES['time'])
        
        st.info(f"📊 Loaded {len(df)} rows")
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading CSV: {str(e)}")
        return None


def manual_column_mapping(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Allow manual column mapping if auto-detection fails
    
    Args:
        df: Raw DataFrame with original column names
        
    Returns:
        DataFrame with standardized column names, or None if not completed
    """
    st.subheader("📋 Manual Column Mapping")
    
    STANDARD_NAMES = {
        'time': "Time (s)",
        'x': "X (m/s^2)",
        'y': "Y (m/s^2)",
        'z': "Z (m/s^2)"
    }
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        time_col = st.selectbox(
            "Time column:",
            df.columns.tolist(),
            key="manual_time"
        )
    
    with col2:
        x_col = st.selectbox(
            "X acceleration column:",
            df.columns.tolist(),
            key="manual_x"
        )
    
    with col3:
        y_col = st.selectbox(
            "Y acceleration column:",
            df.columns.tolist(),
            key="manual_y"
        )
    
    with col4:
        z_col = st.selectbox(
            "Z acceleration column:",
            df.columns.tolist(),
            key="manual_z"
        )
    
    if st.button("✅ Apply Manual Mapping", type="primary"):
        # Check for duplicates
        selected = [time_col, x_col, y_col, z_col]
        if len(set(selected)) != 4:
            st.error("❌ Please select different columns for each field.")
            return None
        
        # Create mapping
        column_mapping = {
            time_col: STANDARD_NAMES['time'],
            x_col: STANDARD_NAMES['x'],
            y_col: STANDARD_NAMES['y'],
            z_col: STANDARD_NAMES['z']
        }
        
        # Apply mapping
        df = df.rename(columns=column_mapping)
        
        # Convert and clean
        for col in STANDARD_NAMES.values():
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=list(STANDARD_NAMES.values()))
        
        if df.empty:
            st.error("❌ No valid data after cleaning.")
            return None
        
        df = df.sort_values(by=STANDARD_NAMES['time'])
        
        st.success(f"✅ Manual mapping applied! Loaded {len(df)} rows")
        
        # Show mapping
        st.info(f"""
        **Mapping applied:**
        - `{time_col}` → **{STANDARD_NAMES['time']}**
        - `{x_col}` → **{STANDARD_NAMES['x']}**
        - `{y_col}` → **{STANDARD_NAMES['y']}**
        - `{z_col}` → **{STANDARD_NAMES['z']}**
        """)
        
        return df
    
    return None


def show_data_preview(df: pd.DataFrame):
    """Display data preview with statistics"""
    if df is None or df.empty:
        st.warning("No data available for preview.")
        return
    
    st.subheader("📄 Data Preview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Data Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
    
    with col2:
        if "Time (s)" in df.columns:
            time_min = df["Time (s)"].min()
            time_max = df["Time (s)"].max()
            st.write(f"**Time Range:** {time_min:.1f}s to {time_max:.1f}s")
            st.write(f"**Duration:** {time_max - time_min:.1f}s")
        
        if "X (m/s^2)" in df.columns:
            # Estimate sampling rate
            if len(df) > 1 and "Time (s)" in df.columns:
                time_diff = df["Time (s)"].diff().mean()
                if time_diff > 0:
                    sampling_rate = 1.0 / time_diff
                    st.write(f"**Est. Sampling Rate:** {sampling_rate:.1f} Hz")
    
    # Display preview
    st.dataframe(df.head(10), use_container_width=True)