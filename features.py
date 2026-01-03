# features.py
import pandas as pd
import numpy as np


def compute_magnitude(df):
    """
    Compute magnitude from X, Y, Z acceleration columns.

    Parameters:
    -----------
    df : pandas DataFrame         containing 'X (m/s^2)', 'Y (m/s^2)', 'Z (m/s^2)' columns

    Returns:
    --------
    df : pandas DataFrame
        Original DataFrame with added 'magnitude' column
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Calculate magnitude: sqrt(x² + y² + z²)
    df['magnitude'] = np.sqrt(
        df['X (m/s^2)'] ** 2 +
        df['Y (m/s^2)'] ** 2 +
        df['Z (m/s^2)'] ** 2
    )

    return df