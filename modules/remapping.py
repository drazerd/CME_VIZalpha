import numpy as np

def calculate_cme_positions(t_array, r_sc_array, t0, t1, t2, 
                           v_cm_sheath, v_exp_sheath, 
                           v_cm_mo, v_exp_mo):
    """
    Calculate parcel positions for CME sheath and MO regions at reference time t0.
    
    This implements the CME-Viz tool logic for mapping time series measurements
    to spatial positions assuming self-similar expansion.
    
    Parameters
    ----------
    t_array : list of datetime objects
        Array of measurement times
    r_sc_array : ndarray, shape (N, 3)
        Array of spacecraft position vectors (in km) corresponding to each time in t_array
    t0 : datetime
        Time when leading edge of sheath strikes spacecraft
    t1 : datetime
        Time when trailing edge of sheath / leading edge of MO strikes spacecraft
    t2 : datetime
        Time when trailing edge of MO strikes spacecraft
    v_cm_sheath : ndarray, shape (3,)
        Center velocity vector of sheath region (km/s)
    v_exp_sheath : ndarray, shape (3,)
        Expansion velocity vector of sheath region (km/s)
    v_cm_mo : ndarray, shape (3,)
        Center velocity vector of MO region (km/s)
    v_exp_mo : ndarray, shape (3,)
        Expansion velocity vector of MO region (km/s)
    
    Returns
    -------
    r_positions : ndarray, shape (N, 3)
        Array of parcel position vectors at time t0 (in km), same length as t_array
    region_mask : ndarray
        Boolean array indicating which region each point belongs to
        True = sheath region, False = MO region, NaN for points outside [t0, t2]
    """
    
    t_array = list(t_array)  # Ensure it's a list
    r_sc_array = np.asarray(r_sc_array)
    v_cm_sheath = np.asarray(v_cm_sheath)
    v_exp_sheath = np.asarray(v_exp_sheath)
    v_cm_mo = np.asarray(v_cm_mo)
    v_exp_mo = np.asarray(v_exp_mo)
    
    # Get spacecraft positions at key times by interpolation
    t_seconds = np.array([(t - t_array[0]).total_seconds() for t in t_array])
    t0_sec = (t0 - t_array[0]).total_seconds()
    t1_sec = (t1 - t_array[0]).total_seconds()
    t2_sec = (t2 - t_array[0]).total_seconds()
    
    r_sc_t0 = np.array([np.interp(t0_sec, t_seconds, r_sc_array[:, i]) for i in range(3)])
    r_sc_t1 = np.array([np.interp(t1_sec, t_seconds, r_sc_array[:, i]) for i in range(3)])
    r_sc_t2 = np.array([np.interp(t2_sec, t_seconds, r_sc_array[:, i]) for i in range(3)])
    
    # Calculate sheath region parameters
    # l0_sheath = (v_cm - v_exp)(t1 - t0) - r_sc(t1) + r_sc(t0)
    l0_sheath = (v_cm_sheath - v_exp_sheath) * (t1_sec - t0_sec) - (r_sc_t1 - r_sc_t0)

    
    # r_LE_sheath = r_sc(t0)
    r_LE_sheath = r_sc_t0
    
    # r_cm_sheath = r_sc(t0) - l0_sheath/2
    r_cm_sheath = r_sc_t0 - l0_sheath / 2
    
    # Calculate MO region parameters
    # l0_MO = (v_cm - v_exp)(t2 - t0) - l0_sheath - r_sc(t2) + r_sc(t0)
    #l0_mo = (v_cm_mo - v_exp_mo) * (t2_sec - t0_sec) - l0_sheath - (r_sc_t2 - r_sc_t0)
    #l0_mo = 2*v_exp_mo*t0_sec + (v_cm_mo - v_exp_mo)*(t2_sec - t1_sec) - (r_sc_t2 - r_sc_t1)
    l0_mo = (v_cm_mo - v_exp_mo)*(t2_sec - t1_sec) - 2*v_exp_mo*(t1_sec - t0_sec) - (r_sc_t2 - r_sc_t1)
    
    # r_LE_MO = r_sc(t0) - l0_sheath
    #r_LE_mo = r_sc_t0 - l0_sheath
    
    # r_cm_MO = r_sc(t0) - l0_sheath - l0_MO/2
    #r_cm_mo = r_sc_t0 - l0_sheath - l0_mo / 2
    #All of the below is to find r_cm_mo correctly by ensuring continuity at t1. You forgot to divide by an l0, stupid!
    # deno_mo = 1 + 2*v_exp_mo*(t1_sec - t0_sec)/l0_mo
    # deno_sheath = 1 + 2*v_exp_sheath*(t1_sec - t0_sec)/l0_sheath
    # numo_sheath = r_sc_t1 - v_cm_sheath*(t1_sec - t0_sec) + 2*v_exp_sheath*r_cm_sheath*(t1_sec - t0_sec)
    # r_cm_mo = (deno_mo*numo_sheath/deno_sheath - r_sc_t1 + v_cm_mo*(t1_sec - t0_sec)) / (2*v_exp_mo*(t1_sec - t0_sec))
    # Calculate what r_sheath gives at t1
    dt1 = t1_sec - t0_sec
    r_sheath_at_t1 = (r_sc_t1 - v_cm_sheath*dt1 + 2*v_exp_sheath*r_cm_sheath*dt1/l0_sheath) / (1 + 2*v_exp_sheath*dt1/l0_sheath)

    # Solve for r_cm_mo component by component
    r_cm_mo = (r_sheath_at_t1 * (1 + 2*v_exp_mo*dt1/l0_mo) - r_sc_t1 + v_cm_mo*dt1) * l0_mo / (2*v_exp_mo*dt1)

    # # Create mask for sheath vs MO regions
    # sheath_mask = np.array([(t >= t0) and (t <= t1) for t in t_array])
    # mo_mask = np.array([(t > t1) and (t <= t2) for t in t_array])
    # Create mask for sheath vs MO regions based on indices
    # Since data is already clipped to CME interval, we work with the full array
    # Find the index closest to t1 (boundary between sheath and MO)
    idx_t1 = np.argmin([abs(t - t1) for t in t_array])

    sheath_mask = np.zeros(len(t_array), dtype=bool)
    mo_mask = np.zeros(len(t_array), dtype=bool)

    # Everything from start to t1 index is sheath
    sheath_mask[0:idx_t1+1] = True
    # Everything after t1 index is MO
    mo_mask[idx_t1+1:] = True

    print(f"\nMask creation check:")
    print(f"Total points: {len(t_array)}")
    print(f"Index of t1 (MO start): {idx_t1}")
    print(f"Sheath points (0 to {idx_t1}): {np.sum(sheath_mask)}")
    print(f"MO points ({idx_t1+1} to {len(t_array)-1}): {np.sum(mo_mask)}")
    print(f"First point in sheath? {sheath_mask[0]}")
    print(f"Last point in MO? {mo_mask[-1]}")
    
    # Initialize output array
    r_positions = np.zeros_like(r_sc_array, dtype=float)
    
    # Calculate positions for sheath region
    # r = [r_sc(t) - v_cm(t-t0) + 2*v_exp*r_cm*(t-t0)/l0] / [1 + 2*v_exp*(t-t0)/l0]
    if np.any(sheath_mask):
        for idx in np.where(sheath_mask)[0]:
            t = t_array[idx]
            r_sc = r_sc_array[idx]
            dt = (t - t0).total_seconds()
            
            # Element-wise operations for each component
            numerator = (r_sc - v_cm_sheath * dt + 
                        2 * v_exp_sheath * r_cm_sheath * dt / l0_sheath)
            denominator = 1 + 2 * v_exp_sheath * dt / l0_sheath
            
            r_positions[idx] = numerator / denominator
    
    # Calculate positions for MO region
    if np.any(mo_mask):
        for idx in np.where(mo_mask)[0]:
            t = t_array[idx]
            r_sc = r_sc_array[idx]
            dt = (t - t0).total_seconds()
            
            numerator = (r_sc - v_cm_mo * dt + 
                        2 * v_exp_mo * r_cm_mo * dt / l0_mo)
            denominator = 1 + 2 * v_exp_mo * dt / l0_mo
            
            r_positions[idx] = numerator / denominator
    
    # ADD THIS BLOCK HERE:
    print(f"\nBoundary check:")
    print(f"r_sc_t0 - l0_sheath = {r_sc_t0 - l0_sheath}")
    #print(f"r_LE_mo = {r_LE_mo}")
    if np.any(sheath_mask):
        print(f"Last sheath r_position = {r_positions[np.where(sheath_mask)[0][-1]]}")
    if np.any(mo_mask):
        print(f"First MO r_position = {r_positions[np.where(mo_mask)[0][0]]}")

    # Print diagnostic information
    print("\n" + "="*70)
    print("CME REMAPPING DIAGNOSTICS")
    print("="*70)
    print(f"Sheath dimensions (l0_sheath): [{l0_sheath[0]:.2e}, {l0_sheath[1]:.2e}, {l0_sheath[2]:.2e}] km")
    print(f"MO dimensions (l0_mo):         [{l0_mo[0]:.2e}, {l0_mo[1]:.2e}, {l0_mo[2]:.2e}] km")
    print(f"Sheath center (r_cm_sheath):   [{r_cm_sheath[0]:.2e}, {r_cm_sheath[1]:.2e}, {r_cm_sheath[2]:.2e}] km")
    print(f"MO center (r_cm_mo):           [{r_cm_mo[0]:.2e}, {r_cm_mo[1]:.2e}, {r_cm_mo[2]:.2e}] km")
    print(f"\nSheath region points: {np.sum(sheath_mask)}")
    print(f"MO region points:     {np.sum(mo_mask)}")
    print("="*70 + "\n")

    # --- ANALYTIC CHECK AT t1 ---
    # Use the interpolated spacecraft positions and the l0/rcm already computed
    # r_sc_t0, r_sc_t1, r_sc_t2, l0_sheath, r_cm_sheath, l0_mo, r_cm_mo already available

    # dt for evaluating at t1, measured from t0 (as your l0_mo was computed)
    dt_t1_from_t0 = (t1 - t0).total_seconds()

    # compute sheath mapping at the exact analytic t1
    term_s = 2.0 * v_exp_sheath * dt_t1_from_t0 / l0_sheath
    num_s = (r_sc_t1 - v_cm_sheath * dt_t1_from_t0 + term_s * r_cm_sheath)
    den_s = 1.0 + term_s
    r_sheath_t1_exact = num_s / den_s

    # compute MO mapping at exact t1 **using the same dt origin** (t0) like your l0_mo
    term_m = 2.0 * v_exp_mo * dt_t1_from_t0 / l0_mo
    num_m = (r_sc_t1 - v_cm_mo * dt_t1_from_t0 + term_m * r_cm_mo)
    den_m = 1.0 + term_m
    r_mo_t1_exact = num_m / den_m

    print("\n=== Exact- t1 analytic comparison ===")
    print("r_sheath(t1) exact =", r_sheath_t1_exact)
    print("r_mo(t1) exact     =", r_mo_t1_exact)
    print("difference          =", r_sheath_t1_exact - r_mo_t1_exact)
    print("\nSheath term / denom per component:", term_s, den_s)
    print("MO term / denom per component:    ", term_m, den_m)

    # Find exact interpolated r_sc at t1 (you already have r_sc_t1)
    print("\n=== Numerical condition check ===")
    for name, (vcm, vexp, rcm, l0) in [
        ("Sheath", (v_cm_sheath, v_exp_sheath, r_cm_sheath, l0_sheath)),
        ("MO",     (v_cm_mo,     v_exp_mo,     r_cm_mo,     l0_mo))
    ]:
        dt_t1 = (t1 - t0).total_seconds()   # how l0 was computed
        term = 2.0 * vexp * dt_t1 / l0
        denom = 1.0 + term
        print(f"\n{name}:")
        print("  term at t1      =", term)
        print("  denom at t1     =", denom)
        # Also check dt at the *first* MO sample and *last* sheath sample (discrete times)
        if np.any(sheath_mask):
            i_last = np.where(sheath_mask)[0][-1]
            dt_last = (t_array[i_last] - t0).total_seconds()
            t_last = t_array[i_last]
            term_last = 2.0 * vexp * dt_last / l0
            denom_last = 1.0 + term_last
            print("  last-sheath sample dt      =", dt_last)
            print("  last-sheath term/denom     =", term_last, denom_last)
        if np.any(mo_mask):
            i_first = np.where(mo_mask)[0][0]
            dt_first = (t_array[i_first] - t0).total_seconds()
            print("  first-MO sample dt         =", dt_first)
            term_first = 2.0 * vexp * dt_first / l0
            print("  first-MO term/denom        =", term_first, 1.0 + term_first)


    
    return r_positions, sheath_mask


# Example usage:
if __name__ == "__main__":
    print("This module provides calculate_cme_positions() for CME visualization.")
    print("Import it and use with your spacecraft data and velocity fits.")