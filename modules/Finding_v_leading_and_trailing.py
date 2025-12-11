import numpy as np
import matplotlib.pyplot as plt


def fit_velocity_edges(t_array, V_array, t_start, t_end, plot=True, title=None):
    """
    Fit velocities using simple linear regression and extract leading/trailing edge velocities.
    
    Parameters
    ----------
    t_array : list of datetime objects
        Time array for the velocity measurements
    V_array : ndarray, shape (N, 3)
        Velocity vectors in HGI or other coordinates (km/s)
    t_start : datetime
        Start time (leading edge time)
    t_end : datetime
        End time (trailing edge time)
    plot : bool, optional
        Whether to create verification plot (default: True)
    title : str, optional
        Title for the plot (e.g., "Sheath Region" or "MO Region")
    
    Returns
    -------
    v_leading : ndarray, shape (3,)
        Velocity vector at leading edge (t_start) in km/s
    v_trailing : ndarray, shape (3,)
        Velocity vector at trailing edge (t_end) in km/s
    v_center : ndarray, shape (3,)
        Center velocity: (v_leading + v_trailing) / 2 in km/s
    v_expansion : ndarray, shape (3,)
        Expansion velocity: (v_leading - v_trailing) / 2 in km/s
    """
    
    # Find indices for the interval
    idx_start = np.argmin([abs(t - t_start) for t in t_array])
    idx_end = np.argmin([abs(t - t_end) for t in t_array])
    
    # Extract interval data
    t_interval = t_array[idx_start:idx_end+1]
    V_interval = V_array[idx_start:idx_end+1]
    
    # Convert times to seconds since the actual leading edge (t_start)
    t_seconds = np.array([(t - t_start).total_seconds() for t in t_interval])
    
    # Initialize result arrays
    v_leading = np.zeros(3)
    v_trailing = np.zeros(3)
    
    # Time values for predictions
    t_predict_start = t_seconds[0]
    t_predict_end = t_seconds[-1]

    print(f"t_predict_start = {t_predict_start}, t_predict_end = {t_predict_end}, {(t_end - t_start).total_seconds()}")
    
    # Store fit parameters for plotting
    fit_params = []
    
    # Fit each component with simple linear regression
    for i in range(3):
        # np.polyfit with deg=1 does linear regression, returns [slope, intercept]
        coeffs = np.polyfit(t_seconds, V_interval[:, i], deg=1)
        slope, intercept = coeffs[0], coeffs[1]
        fit_params.append((slope, intercept))
        
        # Calculate velocities at edges: v = slope * t + intercept
        v_leading[i] = slope * t_predict_start + intercept
        v_trailing[i] = slope * t_predict_end + intercept
    
    # Calculate center and expansion velocities
    v_center = (v_leading + v_trailing) / 2
    v_expansion = (v_leading - v_trailing) / 2
    
    # Create verification plot
    if plot:
        fig, axes = plt.subplots(3, 1, figsize=(10, 10))
        component_names = ['X', 'Y', 'Z']
        
        for i in range(3):
            slope, intercept = fit_params[i]
            
            # Prediction line
            t_line = np.linspace(t_seconds[0], t_seconds[-1], 100)
            v_line = slope * t_line + intercept
            
            # Plot data and fit
            axes[i].plot(t_seconds, V_interval[:, i], 'o', alpha=0.3, 
                        label='Measured data', markersize=4)
            axes[i].plot(t_line, v_line, 'r-', linewidth=2, label='Linear fit')
            
            # Mark edges
            axes[i].axvline(t_predict_start, color='g', linestyle='--', 
                           alpha=0.5, label='Leading edge')
            axes[i].axvline(t_predict_end, color='b', linestyle='--', 
                           alpha=0.5, label='Trailing edge')
            axes[i].plot(t_predict_start, v_leading[i], 'go', markersize=10, 
                        label=f'V_LE: {v_leading[i]:.1f} km/s')
            axes[i].plot(t_predict_end, v_trailing[i], 'bs', markersize=10, 
                        label=f'V_TE: {v_trailing[i]:.1f} km/s')
            
            # Labels and styling
            axes[i].set_ylabel(f'V_{component_names[i]} (km/s)', fontsize=11)
            axes[i].legend(fontsize=9, loc='best')
            axes[i].grid(True, alpha=0.3)
            
            # Add info text
            info_text = f'Slope: {slope:.6f} km/s²\nIntercept: {intercept:.2f} km/s'
            axes[i].text(0.02, 0.98, info_text, transform=axes[i].transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        axes[2].set_xlabel('Time (seconds since leading edge)', fontsize=11)
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    # Print summary
    print(f"\n{'='*60}")
    if title:
        print(f"{title}")
        print(f"{'='*60}")
    print(f"Leading Edge Velocity:  [{v_leading[0]:7.2f}, {v_leading[1]:7.2f}, {v_leading[2]:7.2f}] km/s")
    print(f"Trailing Edge Velocity: [{v_trailing[0]:7.2f}, {v_trailing[1]:7.2f}, {v_trailing[2]:7.2f}] km/s")
    print(f"Center Velocity:        [{v_center[0]:7.2f}, {v_center[1]:7.2f}, {v_center[2]:7.2f}] km/s")
    print(f"Expansion Velocity:     [{v_expansion[0]:7.2f}, {v_expansion[1]:7.2f}, {v_expansion[2]:7.2f}] km/s")
    print(f"{'='*60}\n")
    
    return v_leading, v_trailing, v_center, v_expansion