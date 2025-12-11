from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
from scipy.interpolate import PchipInterpolator

def lon_score_function(lon_diff: float, rad_diff: float, lat_diff: float) -> float:
    """
    Calculate a Gaussian-like score based on the difference in longitude and heliodistance and latitude.
    Lower score for larger lon_diff, higher score for larger rad_diff. 
    Score tends to 0 for no alignment and tends to 1 for perfect alignment. 
    """
    stddev_lon = 15
    stddev_lat = 15
    kappa = 180 #Push kappa to smaller values for lesser tolerance to small rad_diff
    score = np.exp(-float(lon_diff)**2 / (2 * stddev_lon*stddev_lon)) * np.exp(-float(lat_diff)**2 / (2 * stddev_lat*stddev_lat)) * (1 - np.exp(-float(kappa) * float(rad_diff)))
    return score.round(4)

def rad_score_function(lon_diff: float, rad_diff: float, lat_diff: float) -> float:
    """
    Calculate a Gaussian-like score based on the difference in heliodistance and longitude and latitude.
    Lower score for larger rad_diff, higher score for larger lon_diff. 
    Score tends to 0 for no alignment and tends to 1 for perfect alignment. 
    """
    stddev_rad = 0.1
    stddev_lat = 15
    kappa = 180 #Push kappa to smaller values for lesser tolerance to small lon_diff
    score = np.exp(-rad_diff**2 / (2*stddev_rad*stddev_rad)) * np.exp(-float(lat_diff)**2 / (2 * stddev_lat*stddev_lat)) * (1 - np.exp(-kappa * lon_diff))
    return score.round(4)

def lon_align_scores(lon: np.ndarray, rad: np.ndarray, lat: np.ndarray) -> np.ndarray: 
    """
    Returns an array of lon_align_scores for each spacecraft for a particular event.
    Score is based on the longitudes and heliodistances of the spacecrafts. Latitude is not considered. 
    Perfect alignment of n spacecraft gives a score tending to n. No alignment gives a score tending to 1. 
    """
    scores = np.array([])
    for i, (lon_i, rad_i, lat_i) in enumerate(zip(lon, rad, lat)):
        score = 1
        for (lon_j, rad_j, lat_j) in zip(lon, rad, lat):
            lon_diff = abs(lon_i - lon_j)
            rad_diff = abs(rad_i - rad_j)
            lat_diff = abs(lat_i - lat_j)
            score += lon_score_function(lon_diff, rad_diff, lat_diff)
        scores = np.append(scores, score)
    return scores

def rad_align_scores(lon: np.ndarray, rad: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """
    Returns an array of rad_align_scores for each spacecraft for a particular event.
    Score is based on the longitudes and heliodistances of the spacecrafts. Latitude is not considered. 
    Perfect alignment of n spacecraft gives a score tending to n. No alignment gives a score tending to 1. 
    """
    scores = np.array([])
    for i, (lon_i, rad_i, lat_i) in enumerate(zip(lon, rad, lat)):
        score = 1
        for (lon_j, rad_j, lat_j) in zip(lon, rad, lat):
            lon_diff = abs(lon_i - lon_j)
            rad_diff = abs(rad_i - rad_j)
            lat_diff = abs(lat_i - lat_j)
            score += rad_score_function(lon_diff, rad_diff, lat_diff)
        scores = np.append(scores, score)
    return scores

def lon_dbscan(lon: np.ndarray):
    """
    Clusters the spacecraft based on their longitudes using DBSCAN.
    """
    X = lon.reshape(-1, 1)
    db = DBSCAN(eps=15, min_samples=2).fit(X)
    return db.labels_

def rad_dbscan(rad: np.ndarray):
    """
    Clusters the spacecraft based on their heliodistances using DBSCAN.
    """
    X = rad.reshape(-1, 1)
    db = DBSCAN(eps=0.1, min_samples=2).fit(X)
    return db.labels_

import numpy as np

###################

def interpolate_vector_series(t_source, vectors, t_target):
    """
    Interpolates 3D vector time series to new timestamps.

    Parameters:
    - t_source: array of timestamps (datetime or float), original
    - vectors: array of shape (N, 3), vectors at t_source
    - t_target: array of timestamps to interpolate to

    Returns:
    - vectors_interp: array of shape (len(t_target), 3)
    """
    t_source = np.array([t.timestamp() for t in t_source])
    t_target = np.array([t.timestamp() for t in t_target])

    vectors = np.asarray(vectors)
    interpolators = [interp1d(t_source, vectors[:, i], kind='linear', fill_value='extrapolate')
                     for i in range(3)]
    
    vectors_interp = np.stack([interp(t_target) for interp in interpolators], axis=1)
    return vectors_interp


def pchip_vector_interp(t_source, vectors, t_target):
    t_source = np.array([t.timestamp() for t in t_source])
    t_target = np.array([t.timestamp() for t in t_target])

    interpolators = [PchipInterpolator(t_source, vectors[:, i]) for i in range(3)]
    vectors_interp = np.stack([interp(t_target) for interp in interpolators], axis=1)

    return vectors_interp


def rtn_to_hgi(velocity_rtn, position_hgi):
    """
    Transforms a 3D velocity vector from RTN to HGI coordinates.

    Parameters:
    - velocity_rtn: array-like, shape (3,), velocity vector in RTN coordinates [v_R, v_T, v_N]
    - position_hgi: array-like, shape (3,), spacecraft position vector in HGI coordinates [x, y, z]

    Returns:
    - velocity_hgi: ndarray, shape (3,), velocity vector in HGI coordinates
    """
    velocity_rtn = np.asarray(velocity_rtn)
    position_hgi = np.asarray(position_hgi)
    
    # Define Sun's rotation axis in HGI (unit vector along +Z)
    omega_sun = np.array([0.0, 0.0, 1.0])

    # Compute RTN basis vectors in HGI coordinates
    R_hat = position_hgi / np.linalg.norm(position_hgi)
    T_hat = np.cross(omega_sun, R_hat)
    T_hat /= np.linalg.norm(T_hat)
    N_hat = np.cross(R_hat, T_hat)

    # Construct rotation matrix from RTN to HGI (columns are R̂, T̂, N̂)
    M_rtn_to_hgi = np.column_stack((R_hat, T_hat, N_hat))

    # Apply rotation
    velocity_hgi = M_rtn_to_hgi @ velocity_rtn

    return velocity_hgi


def batch_rtn_to_hgi(V_rtn_interp, R_sc_interp):
    """
    Apply RTN→HGI transformation to arrays of vectors.

    Parameters:
    - V_rtn_interp: (N, 3) array of velocity (or anything else) vectors in RTN
    - R_sc_interp: (N, 3) array of spacecraft positions in HGI

    Returns:
    - V_hgi: (N, 3) array of vectors in HGI
    """
    V_hgi = np.zeros_like(V_rtn_interp)
    
    for i in range(len(V_rtn_interp)):
        V_hgi[i] = rtn_to_hgi(V_rtn_interp[i], R_sc_interp[i])
    
    return V_hgi


def mask_vectors_by_components(vectors, component_bounds):
    """
    Filters 3D vectors based on individual component bounds and removes NaNs.

    Args:
        vectors (np.ndarray): shape (N, 3) array.
        component_bounds (list of tuple): [(min_x, max_x), (min_y, max_y), (min_z, max_z)]

    Returns:
        mask (np.ndarray): boolean mask of shape (N,) indicating valid rows.
    """
    vectors = np.asarray(vectors)
    assert vectors.shape[1] == 3, "Expecting Nx3 array for vectors"
    
    # Start with all valid
    mask = np.ones(vectors.shape[0], dtype=bool)

    for i, (lo, hi) in enumerate(component_bounds):
        mask &= np.isfinite(vectors[:, i])  # Exclude NaNs or Infs
        mask &= (vectors[:, i] >= lo) & (vectors[:, i] <= hi)

    return mask


def spherical_to_cartesian(r, theta, phi):
    """
    Converts spherical HGI (r, theta, phi) to Cartesian HGI (x, y, z).
    Angles in radians.
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])


def batch_spherical_to_cartesian(r, theta, phi, degrees=False):
    """
    Converts a batch of spherical coordinates to Cartesian coordinates.

    Parameters:
    - r:      array-like, shape (N,) — radial distances
    - theta:  array-like, shape (N,) — colatitudes (angle from +Z), in radians or degrees
    - phi:    array-like, shape (N,) — longitudes (angle from +X), in radians or degrees
    - degrees: bool — if True, interpret theta and phi as degrees

    Returns:
    - cartesian: ndarray, shape (N, 3) — [x, y, z] coordinates
    """
    r = np.asarray(r)
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    if degrees:
        theta = np.radians(theta)
        phi = np.radians(phi)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.column_stack((x, y, z))


def batch_cartesian_to_spherical(cartesian, degrees=False):
    """
    Converts an array of Cartesian vectors to spherical coordinates.

    Parameters:
    - cartesian: ndarray of shape (N, 3), input [x, y, z]
    - degrees: bool — if True, return angles in degrees

    Returns:
    - r: ndarray of shape (N,), radial distances
    - theta: ndarray of shape (N,), colatitude (0 = +Z)
    - phi: ndarray of shape (N,), longitude (0 = +X, increases toward +Y)
    """
    x, y, z = cartesian[:, 0], cartesian[:, 1], cartesian[:, 2]
    
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    # Wrap phi to [0, 2π)
    phi = np.mod(phi, 2 * np.pi)

    if degrees:
        theta = np.degrees(theta)
        phi = np.degrees(phi)

    return r, theta, phi


def custom_rotate_frame(vectors_hgi, reference_vector):
    """
    Rotate vectors from HGI Cartesian frame to a new frame aligned with the initial position vector.

    Parameters:
    ----------
    vectors_hgi : ndarray of shape (N, 3)
        Array of 3D vectors in the HGI frame (e.g., positions or magnetic fields).

    reference_vector : ndarray of shape (3,)
        The initial position vector (R0) in HGI coordinates, used to define the new frame.

    Returns:
    -------
    vectors_rotated : ndarray of shape (N, 3)
        The input vectors rotated into the new trajectory-aligned frame.
    """

    # Normalize the reference vector to define the new X' axis
    X_unit = reference_vector / np.linalg.norm(reference_vector)

    # Define solar rotation axis in HGI (Z-axis in HGI)
    Omega = np.array([0, 0, 1])

    # Define Y' axis as orthogonal to both Omega and R0
    Y_unit = np.cross(Omega, X_unit)
    Y_unit /= np.linalg.norm(Y_unit)

    # Define Z' axis to complete right-handed system
    Z_unit = np.cross(X_unit, Y_unit)
    Z_unit /= np.linalg.norm(Z_unit)

    # Rotation matrix: columns are new basis vectors in HGI coordinates
    R_matrix = np.stack([X_unit, Y_unit, Z_unit], axis=1)  # shape (3, 3)

    # Rotate all vectors into the new frame
    vectors_rotated = (R_matrix.T @ vectors_hgi.T).T  # shape (N, 3)

    return vectors_rotated


import numpy as np

def draw_spherical_grid_clipped(
    ax,
    r_values=[0.30, 0.35, 0.4, 0.45, 0.5, 0.55],
    latitudes_deg=[85, 87.5, 90, 92.5, 95],
    longitudes_deg=[-5, -2.5, 0, 2.5, 5],
    shell_color="#D1D1D1",
    grid_color="#868686",
    lw=0.1, 
    alpha=0,
    resolution=200
):
    """
    Draws a spherical grid in a 3D matplotlib plot, with lat/lon and radial lines clipped to the visible volume.
    """
    # Axis bounds
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    def point_within_bounds(x, y, z):
        return (
            xlim[0] <= x <= xlim[1] and
            ylim[0] <= y <= ylim[1] and
            zlim[0] <= z <= zlim[1]
        )

    def segment_within_bounds(x1, y1, z1, x2, y2, z2):
        return point_within_bounds(x1, y1, z1) or point_within_bounds(x2, y2, z2)

    u = np.linspace(0, np.pi, resolution)
    v = np.linspace(0, 2 * np.pi, resolution)

    # Radial shells (optional, translucent)
    for r in r_values:
        x = r * np.outer(np.sin(u), np.cos(v))
        y = r * np.outer(np.sin(u), np.sin(v))
        z = r * np.outer(np.cos(u), np.ones_like(v))
        ax.plot_surface(x, y, z, color=shell_color, alpha=alpha, linewidth=0)

    # Latitude lines (θ)
    for theta_deg in latitudes_deg:
        theta = np.radians(theta_deg)
        phi = np.linspace(0, 2 * np.pi, resolution)
        for r in r_values:
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta) * np.ones_like(phi)
            for i in range(len(x) - 1):
                if segment_within_bounds(x[i], y[i], z[i], x[i+1], y[i+1], z[i+1]):
                    ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=grid_color, lw=lw)

    # Longitude lines (φ)
    for phi_deg in longitudes_deg:
        phi = np.radians(phi_deg)
        theta = np.linspace(0, np.pi, resolution)
        for r in r_values:
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            for i in range(len(x) - 1):
                if segment_within_bounds(x[i], y[i], z[i], x[i+1], y[i+1], z[i+1]):
                    ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=grid_color, lw=lw)

    # Radial lines (from origin to grid intersections)
    for theta_deg in latitudes_deg:
        theta = np.radians(theta_deg)
        for phi_deg in longitudes_deg:
            phi = np.radians(phi_deg)
            r = np.linspace(0, r_values[-1], resolution)
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            for i in range(len(x) - 1):
                if segment_within_bounds(x[i], y[i], z[i], x[i+1], y[i+1], z[i+1]):
                    ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=grid_color, lw=lw)




import numpy as np

def draw_spherical_grid_on_box_faces(
    ax,
    r_bounds=(0.3, 0.7),                    # radial range in AU or desired unit
    latitudes_deg=[-85, -87.5, -90, -92.5, -95],  # constant latitude lines (θ)
    longitudes_deg=[-5, -2.5, 0, 2.5, 5],  # constant longitude lines (φ)
    grid_color="#FFFFFF",  
    lw=0.1,
    resolution=200
):
    """
    Draws spherical grid lines (latitudes, longitudes, radial) projected only onto the faces of the 3D plot cuboid.

    Parameters:
    - ax: matplotlib 3D axis
    - r_bounds: (r_min, r_max), min/max radius in same units as plot
    - latitudes_deg: list of constant-latitude lines (θ)
    - longitudes_deg: list of constant-longitude lines (φ)
    - grid_color: line color
    - lw: line width
    - resolution: sampling resolution
    """

    # Face extents
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    # Latitude lines (θ) on side faces
    for theta_deg in latitudes_deg:
        theta = np.radians(theta_deg)
        phi_vals = np.linspace(0, 2 * np.pi, resolution)
        r_min, r_max = r_bounds
        for face in ['+x', '-x', '+y', '-y']:
            r = np.linspace(r_min, r_max, resolution)
            phi = phi_vals
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta) * np.ones_like(phi)

            if face == '+x': x[:] = xlim[1]
            if face == '-x': x[:] = xlim[0]
            if face == '+y': y[:] = ylim[1]
            if face == '-y': y[:] = ylim[0]

            ax.plot(x, y, z, color=grid_color, lw=lw)

    # Longitude lines (φ) on top/bottom faces
    for phi_deg in longitudes_deg:
        phi = np.radians(phi_deg)
        theta_vals = np.linspace(0, np.pi, resolution)
        for face in ['+z', '-z']:
            r = np.linspace(r_min, r_max, resolution)
            theta = theta_vals
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)

            if face == '+z': z[:] = zlim[1]
            if face == '-z': z[:] = zlim[0]

            ax.plot(x, y, z, color=grid_color, lw=lw)

    # Radial lines on corner edges
    for theta_deg in latitudes_deg:
        theta = np.radians(theta_deg)
        for phi_deg in longitudes_deg:
            phi = np.radians(phi_deg)
            r = np.linspace(r_min, r_max, resolution)
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)

            # Clamp one end of the radial line to the closest box face (approx.)
            if x[0] < (xlim[0] + xlim[1])/2: x[:] = xlim[0]
            else: x[:] = xlim[1]

            if y[0] < (ylim[0] + ylim[1])/2: y[:] = ylim[0]
            else: y[:] = ylim[1]

            if z[0] < (zlim[0] + zlim[1])/2: z[:] = zlim[0]
            else: z[:] = zlim[1]

            ax.plot(x, y, z, color=grid_color, lw=lw)


def draw_time_plane(ax, x_position, color='k', alpha=0.2):
    # Build a rectangular mesh spanning full Y,Z range at fixed X
    y = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 2)
    z = np.linspace(ax.get_zlim()[0], ax.get_zlim()[1], 2)
    Y, Z = np.meshgrid(y, z)
    X_plane = np.full_like(Y, x_position)

    ax.plot_surface(X_plane, Y, Z, color=color, alpha=alpha, zorder=0)


from sklearn.linear_model import HuberRegressor
import numpy as np

def fit_velocity_huber(times, V_hgi, t_start, t_end, clip=5):
    """
    Fit velocity components with Huber regression between two times,
    clipping a few edge points, and evaluate at the exact start and end times.
    
    Parameters
    ----------
    times : list of datetime.datetime
        Time array corresponding to V_hgi.
    V_hgi : ndarray of shape (N, 3)
        Velocity components in HGI frame (km/s).
    t_start : datetime.datetime
        Start time of interval (e.g. t_mo_start).
    t_end : datetime.datetime
        End time of interval (e.g. t_cme_end).
    clip : int, optional
        Number of points to discard from each side of the interval before fitting.
    
    Returns
    -------
    v_start : ndarray of shape (3,)
        Fitted velocity at t_start.
    v_end : ndarray of shape (3,)
        Fitted velocity at t_end.
    """
    
    # Restrict to interval
    mask = [(t_start <= t <= t_end) for t in times]
    times_sel = np.array(times)[mask]
    V_sel = np.array(V_hgi)[mask]

    if len(times_sel) <= 2*clip:
        raise ValueError("Not enough points in interval after clipping. Reduce 'clip' or check time range.")

    # Convert times to seconds since start of interval
    t0 = times_sel[0]
    t_sec = np.array([(t - t0).total_seconds() for t in times_sel])

    # Clip edge points
    t_fit = t_sec[clip:-clip]
    V_fit = V_sel[clip:-clip]

    # Prepare evaluation times
    t_eval = np.array([
        (t_start - t0).total_seconds(),
        (t_end   - t0).total_seconds()
    ]).reshape(-1, 1)

    # Fit each component separately
    v_start = []
    v_end = []
    for comp in range(3):
        huber = HuberRegressor().fit(t_fit.reshape(-1,1), V_fit[:, comp])
        v_pred = huber.predict(t_eval)
        v_start.append(v_pred[0])
        v_end.append(v_pred[1])

    return np.array(v_start), np.array(v_end)
